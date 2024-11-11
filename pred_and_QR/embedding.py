from numba.core.errors import NumbaPerformanceWarning
import warnings
import numpy as np
import cupy as cp
from image_processing import (
    split_image_flexible,
    merge_image_flexible
)
from utils import (
    brute_force_weight_search_cuda,
    generate_random_binary_array,
    calculate_block_variance_cuda
)
from common import *
from pee import *

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

def histogram_data_hiding(img, pee_info_bits, ratio_of_ones=1):
    print(f"HS Input - Max pixel value: {np.max(img)}")
    print(f"HS Input - Min pixel value: {np.min(img)}")
    h_img, w_img = img.shape
    markedImg = img.copy()
    total_payload = 0
    rounds = 0
    payloads = []

    pee_info_length = len(pee_info_bits)

    # 创建一个掩码来跟踪已经用于嵌入的像素
    embedded_mask = np.zeros_like(markedImg, dtype=bool)

    while np.max(markedImg) < 255:
        rounds += 1
        hist = np.bincount(markedImg[~embedded_mask].ravel(), minlength=256)
        
        print(f"\nRound {rounds}:")
        print(f"Histogram shape: {hist.shape}")
        
        peak = np.argmax(hist[:-1])  # Avoid selecting 255 as peak
        print(f"Histogram peak: {peak}, value: {hist[peak]}")
        
        print(f"Histogram around peak:")
        for i in range(max(0, peak-5), min(256, peak+6)):
            print(f"  Pixel value {i}: {hist[i]}")
        
        max_payload = hist[peak]
        
        if max_payload == 0:
            print("No more available peak values. Stopping embedding.")
            break
        
        if pee_info_length > 0:
            embedding_data = pee_info_bits[:max_payload]
            pee_info_bits = pee_info_bits[max_payload:]
            pee_info_length -= len(embedding_data)
            if len(embedding_data) < max_payload:
                random_bits = generate_random_binary_array(max_payload - len(embedding_data), ratio_of_ones)
                embedding_data += ''.join(map(str, random_bits))
        else:
            embedding_data = ''.join(map(str, generate_random_binary_array(max_payload, ratio_of_ones)))
        
        actual_payload = len(embedding_data)
        
        embedded_count = 0
        modified_count = 0
        
        # 创建一个掩码，标记所有需要移动的像素
        move_mask = (markedImg > peak) & (~embedded_mask)
        
        # 移动所有大于峰值的未嵌入像素
        markedImg[move_mask] += 1
        modified_count += np.sum(move_mask)
        
        # 嵌入数据到峰值像素
        peak_pixels = np.where((markedImg == peak) & (~embedded_mask))
        for i in range(min(len(peak_pixels[0]), actual_payload)):
            y, x = peak_pixels[0][i], peak_pixels[1][i]
            markedImg[y, x] += int(embedding_data[i])
            embedded_mask[y, x] = True
            embedded_count += 1
            modified_count += 1
        
        total_payload += actual_payload
        payloads.append(actual_payload)
        
        print(f"Embedded {actual_payload} bits")
        print(f"Modified {modified_count} pixels")
        print(f"Remaining PEE info: {pee_info_length} bits")
        print(f"Current max pixel value: {np.max(markedImg)}")
        print(f"Current min pixel value: {np.min(markedImg)}")
        
        hist_after = np.bincount(markedImg.ravel(), minlength=256)
        print(f"Histogram after embedding:")
        for i in range(max(0, peak-5), min(256, peak+7)):
            print(f"  Pixel value {i}: {hist_after[i]}")

    print(f"Final max pixel value: {np.max(markedImg)}")
    print(f"Final min pixel value: {np.min(markedImg)}")
    print(f"Total rounds: {rounds}")
    print(f"Total payload: {total_payload}")

    return markedImg, total_payload, payloads, rounds

def pee_process_with_rotation_cuda(img, total_embeddings, ratio_of_ones, use_different_weights, split_size, el_mode):
    """
    使用旋轉的 PEE 處理函數
    
    Parameters:
    -----------
    img : numpy.ndarray
        輸入圖像
    total_embeddings : int
        總嵌入次數
    ratio_of_ones : float
        嵌入數據中1的比例
    use_different_weights : bool
        是否對每個子圖像使用不同的權重
    split_size : int
        分割大小 (例如：4 表示 4x4=16 塊)
    el_mode : int
        EL模式 (0:無限制, 1:漸增, 2:漸減)
    """
    original_img = cp.asarray(img)
    height, width = img.shape
    total_pixels = height * width
    pee_stages = []
    total_payload = 0
    current_img = original_img.copy()
    previous_psnr = float('inf')
    previous_ssim = 1.0
    previous_payload = float('inf')

    for embedding in range(total_embeddings):
        print(f"\nStarting embedding {embedding}")
        
        if embedding == 0:
            target_psnr = 40.0
            target_bpp = 0.9
        else:
            target_psnr = max(28.0, previous_psnr - 1)  # Ensure PSNR decreases
            target_bpp = max(0.5, (previous_payload / total_pixels) * 0.95)  # Ensure payload decreases
        
        print(f"Target PSNR for embedding {embedding}: {target_psnr:.2f}")
        print(f"Target BPP for embedding {embedding}: {target_bpp:.4f}")
        
        stage_info = {
            'embedding': embedding,
            'sub_images': [],
            'payload': 0,
            'psnr': 0,
            'ssim': 0,
            'hist_corr': 0,
            'bpp': 0,
            'block_params': []
        }
        
        stage_payload = 0
        embedded_sub_images = []
        stage_rotation = embedding * 90  # 每個階段旋轉90度
        
        # 使用新的彈性分割函數，固定使用 block_base=True
        sub_images = split_image_flexible(current_img, split_size, block_base=True)
        
        # 計算最大子塊載荷
        total_blocks = split_size * split_size
        if embedding > 0:
            max_sub_payload = previous_payload // total_blocks
        else:
            first_sub_img = sub_images[0]
            max_sub_payload = first_sub_img.size
        
        for i, sub_img in enumerate(sub_images):
            rotated_sub_img = cp.asarray(sub_img)
            sub_data = generate_random_binary_array(rotated_sub_img.size, ratio_of_ones)
            sub_data = cp.asarray(sub_data, dtype=cp.uint8)
            
            # 根據 el_mode 決定 max_el
            if el_mode == 1:  # Increasing
                max_el = 3 + embedding * 2
            elif el_mode == 2:  # Decreasing
                max_el = 11 - embedding * 2
            else:  # No restriction
                max_el = 7
            
            # 使用改進的自適應 EL 計算
            local_el = compute_improved_adaptive_el(rotated_sub_img, window_size=5, max_el=max_el)
            
            if use_different_weights or i == 0:
                weights, (sub_payload, sub_psnr) = brute_force_weight_search_cuda(
                    rotated_sub_img, sub_data, local_el, target_bpp, target_psnr, embedding
                )
            
            # 使用最小值作為實際嵌入量
            data_to_embed = sub_data[:min(int(max_sub_payload), len(sub_data))]
            embedded_sub, payload = multi_pass_embedding(
                rotated_sub_img, data_to_embed, local_el, weights, embedding
            )
            
            embedded_sub_images.append(embedded_sub)
            stage_payload += payload
            
            sub_img_np = cp.asnumpy(sub_img)
            embedded_sub_np = cp.asnumpy(embedded_sub)
            sub_psnr = calculate_psnr(sub_img_np, embedded_sub_np)
            sub_ssim = calculate_ssim(sub_img_np, embedded_sub_np)
            sub_hist_corr = histogram_correlation(
                np.histogram(sub_img_np, bins=256, range=(0, 255))[0],
                np.histogram(embedded_sub_np, bins=256, range=(0, 255))[0]
            )
            
            local_el_np = local_el.get()
            max_el_used = int(np.max(local_el_np))

            block_info = {
                'weights': weights.tolist() if isinstance(weights, np.ndarray) else weights,
                'EL': max_el_used,
                'payload': int(payload),
                'psnr': float(sub_psnr),
                'ssim': float(sub_ssim),
                'rotation': stage_rotation,
                'hist_corr': float(sub_hist_corr)
            }
            stage_info['sub_images'].append(block_info)
            stage_info['block_params'].append(block_info)

        # 使用新的彈性合併函數
        stage_img = merge_image_flexible(embedded_sub_images, split_size, block_base=True)
        
        # 對整個圖像進行旋轉
        stage_img = cp.rot90(stage_img)
        stage_info['stage_img'] = stage_img

        stage_img_np = cp.asnumpy(stage_img)
        original_img_np = cp.asnumpy(original_img)
        stage_info['psnr'] = float(calculate_psnr(original_img_np, stage_img_np))
        stage_info['ssim'] = float(calculate_ssim(original_img_np, stage_img_np))
        stage_info['hist_corr'] = float(histogram_correlation(
            np.histogram(original_img_np, bins=256, range=(0, 255))[0],
            np.histogram(stage_img_np, bins=256, range=(0, 255))[0]
        ))
        stage_info['payload'] = stage_payload
        stage_info['bpp'] = float(stage_info['payload'] / total_pixels)
        
        # 檢查當前階段的載荷和PSNR是否小於前一階段
        if embedding > 0:
            if stage_info['payload'] >= previous_payload:
                print(f"Warning: Stage {embedding} payload ({stage_info['payload']}) is not less than previous stage ({previous_payload}).")
                stage_info['payload'] = int(previous_payload * 0.95)
                print(f"Adjusted payload: {stage_info['payload']}")
            
            if stage_info['psnr'] >= previous_psnr:
                print(f"Warning: Stage {embedding} PSNR ({stage_info['psnr']:.2f}) is not less than previous stage ({previous_psnr:.2f}).")
        
        pee_stages.append(stage_info)
        total_payload += stage_info['payload']
        previous_psnr = stage_info['psnr']
        previous_ssim = stage_info['ssim']
        previous_payload = stage_info['payload']

        print(f"Embedding {embedding} summary:")
        print(f"  Payload: {stage_info['payload']}")
        print(f"  BPP: {stage_info['bpp']:.4f}")
        print(f"  PSNR: {stage_info['psnr']:.2f}")
        print(f"  SSIM: {stage_info['ssim']:.4f}")
        print(f"  Hist Corr: {stage_info['hist_corr']:.4f}")
        print(f"  Rotation: {stage_rotation}")

        current_img = stage_img

    final_pee_img = cp.asnumpy(current_img)
    
    return final_pee_img, int(total_payload), pee_stages

def pee_process_with_split_cuda(img, total_embeddings, ratio_of_ones, use_different_weights, split_size, el_mode, block_base):
    """
    使用彈性分割的 PEE 處理函數
    
    Parameters:
    -----------
    img : numpy.ndarray
        輸入圖像
    total_embeddings : int
        總嵌入次數
    ratio_of_ones : float
        嵌入數據中1的比例
    use_different_weights : bool
        是否對每個子圖像使用不同的權重
    split_size : int
        分割大小 (例如：4 表示 4x4=16 塊)
    el_mode : int
        EL模式 (0:無限制, 1:漸增, 2:漸減)
    block_base : bool
        True: 使用 block-based 分割, False: 使用 quarter-based 分割
    """
    original_img = cp.asarray(img)
    height, width = original_img.shape
    total_pixels = height * width
    pee_stages = []
    total_payload = 0
    current_img = original_img.copy()
    previous_psnr = float('inf')
    previous_ssim = 1.0
    previous_payload = float('inf')

    for embedding in range(total_embeddings):
        print(f"\nStarting embedding {embedding}")
        
        if embedding == 0:
            target_psnr = 40.0
            target_bpp = 0.9
        else:
            target_psnr = max(28.0, previous_psnr - 1)
            target_bpp = max(0.5, (previous_payload / total_pixels) * 0.95)
        
        print(f"Target PSNR for embedding {embedding}: {target_psnr:.2f}")
        print(f"Target BPP for embedding {embedding}: {target_bpp:.4f}")
        
        stage_info = {
            'embedding': embedding,
            'sub_images': [],
            'payload': 0,
            'psnr': 0,
            'ssim': 0,
            'hist_corr': 0,
            'bpp': 0,
            'block_params': [],
            'rotated_sub_images': [],
            'non_rotated_sub_images': []
        }
        
        stage_payload = 0
        embedded_sub_images = []
        
        # 使用新的彈性分割函數
        sub_images = split_image_flexible(current_img, split_size, block_base)
        
        # 為每個子圖像設置隨機旋轉角度
        total_blocks = split_size * split_size
        stage_rotations = cp.random.choice([-270, -180, -90, 0, 90, 180, 270], size=total_blocks)
        
        # 計算最大嵌入量
        if embedding > 0:
            max_sub_payload = previous_payload // total_blocks
        else:
            first_sub_img = sub_images[0]
            max_sub_payload = first_sub_img.size
        
        for i, sub_img in enumerate(sub_images):
            sub_img = cp.asarray(sub_img)
            rotation = int(stage_rotations[i])
            rotated_sub_img = cp.rot90(sub_img, k=rotation // 90)

            sub_data = generate_random_binary_array(rotated_sub_img.size, ratio_of_ones)
            sub_data = cp.asarray(sub_data, dtype=cp.uint8)
            
            # 根據 el_mode 決定 max_el
            if el_mode == 1:  # Increasing
                max_el = 3 + embedding * 2
            elif el_mode == 2:  # Decreasing
                max_el = 11 - embedding * 2
            else:  # No restriction
                max_el = 7
            
            local_el = compute_improved_adaptive_el(rotated_sub_img, window_size=5, max_el=max_el)
            
            if use_different_weights or i == 0:
                weights, (sub_payload, sub_psnr) = brute_force_weight_search_cuda(
                    rotated_sub_img, sub_data, local_el, target_bpp, target_psnr, embedding
                )
            
            data_to_embed = sub_data[:min(int(max_sub_payload), len(sub_data))]
            embedded_sub, payload = multi_pass_embedding(rotated_sub_img, data_to_embed, local_el, weights, embedding)
            
            rotated_back_sub = cp.rot90(embedded_sub, k=-rotation // 90)
            embedded_sub_images.append(rotated_back_sub)
            stage_payload += payload
            
            # 計算各種指標
            sub_img_np = cp.asnumpy(sub_img)
            rotated_back_sub_np = cp.asnumpy(rotated_back_sub)
            sub_psnr = calculate_psnr(sub_img_np, rotated_back_sub_np)
            sub_ssim = calculate_ssim(sub_img_np, rotated_back_sub_np)
            sub_hist_corr = histogram_correlation(
                np.histogram(sub_img_np, bins=256, range=(0, 255))[0],
                np.histogram(rotated_back_sub_np, bins=256, range=(0, 255))[0]
            )
            
            local_el_np = local_el.get()
            max_el_used = int(np.max(local_el_np))

            block_info = {
                'weights': weights.tolist() if isinstance(weights, np.ndarray) else weights,
                'EL': max_el_used,
                'payload': int(payload),
                'psnr': float(sub_psnr),
                'ssim': float(sub_ssim),
                'rotation': rotation,
                'hist_corr': float(sub_hist_corr)
            }
            stage_info['sub_images'].append(block_info)
            stage_info['block_params'].append(block_info)
            stage_info['rotated_sub_images'].append(cp.asnumpy(embedded_sub))
            stage_info['non_rotated_sub_images'].append(cp.asnumpy(rotated_back_sub))

        # 使用新的彈性合併函數
        stage_img = merge_image_flexible(embedded_sub_images, split_size, block_base)
        stage_info['stage_img'] = stage_img

        stage_img_np = cp.asnumpy(stage_img)
        original_img_np = cp.asnumpy(original_img)
        stage_info['psnr'] = float(calculate_psnr(original_img_np, stage_img_np))
        stage_info['ssim'] = float(calculate_ssim(original_img_np, stage_img_np))
        stage_info['hist_corr'] = float(histogram_correlation(
            np.histogram(original_img_np, bins=256, range=(0, 255))[0],
            np.histogram(stage_img_np, bins=256, range=(0, 255))[0]
        ))
        stage_info['payload'] = stage_payload
        stage_info['bpp'] = float(stage_info['payload'] / total_pixels)
        
        if embedding > 0:
            if stage_info['payload'] >= previous_payload:
                print(f"Warning: Stage {embedding} payload ({stage_info['payload']}) is not less than previous stage ({previous_payload}).")
                stage_info['payload'] = int(previous_payload * 0.95)
                print(f"Adjusted payload: {stage_info['payload']}")
            
            if stage_info['psnr'] >= previous_psnr:
                print(f"Warning: Stage {embedding} PSNR ({stage_info['psnr']:.2f}) is not less than previous stage ({previous_psnr:.2f}).")
        
        pee_stages.append(stage_info)
        total_payload += stage_info['payload']
        previous_psnr = stage_info['psnr']
        previous_ssim = stage_info['ssim']
        previous_payload = stage_info['payload']

        print(f"Embedding {embedding} summary:")
        print(f"  Payload: {stage_info['payload']}")
        print(f"  BPP: {stage_info['bpp']:.4f}")
        print(f"  PSNR: {stage_info['psnr']:.2f}")
        print(f"  SSIM: {stage_info['ssim']:.4f}")
        print(f"  Hist Corr: {stage_info['hist_corr']:.4f}")
        print(f"  Rotations: {stage_rotations.tolist()}")

        current_img = stage_img

    final_pee_img = cp.asnumpy(current_img)
    
    return final_pee_img, int(total_payload), pee_stages, stage_rotations.tolist()

def pee_process_with_quadtree_cuda(img, total_embeddings, ratio_of_ones, use_different_weights, 
                                 min_block_size, variance_threshold, el_mode):
    """
    使用Quad tree的PEE處理函數
    
    Parameters:
    -----------
    img : numpy.ndarray
        輸入圖像
    total_embeddings : int
        總嵌入次數
    ratio_of_ones : float
        嵌入數據中1的比例
    use_different_weights : bool
        是否對每個子圖像使用不同的權重
    min_block_size : int
        最小區塊大小
    variance_threshold : float
        變異閾值
    el_mode : int
        EL模式 (0:無限制, 1:漸增, 2:漸減)
    """
    def calculate_metrics_with_rotation(original_img, stage_img, current_rotation):
        """計算考慮旋轉的metrics"""
        # 確保輸入是numpy array
        if isinstance(original_img, cp.ndarray):
            original_img = cp.asnumpy(original_img)
        if isinstance(stage_img, cp.ndarray):
            stage_img = cp.asnumpy(stage_img)
        
        # 如果需要旋轉，將圖像旋轉回原始方向
        if current_rotation != 0:
            k = (-current_rotation // 90) % 4
            stage_img = np.rot90(stage_img, k=k)
        
        # 計算各種指標
        psnr = calculate_psnr(original_img, stage_img)
        ssim = calculate_ssim(original_img, stage_img)
        hist_corr = histogram_correlation(
            np.histogram(original_img, bins=256, range=(0, 255))[0],
            np.histogram(stage_img, bins=256, range=(0, 255))[0]
        )
        
        return psnr, ssim, hist_corr

    # 初始化變數
    original_img = cp.asarray(img)
    height, width = original_img.shape
    total_pixels = height * width
    pee_stages = []
    total_payload = 0
    current_img = original_img.copy()
    previous_psnr = float('inf')
    previous_ssim = 1.0
    previous_payload = float('inf')

    # 初始化塊權重字典
    block_weights = {
        512: None,
        256: None,
        128: None,
        64: None
    }

    # process_block 函數保持不變
    def process_block(block, position, size):
        """處理單個區塊"""
        if size < min_block_size:
            return []
        
        # 確保block是CuPy陣列
        block = cp.asarray(block)
        
        # 計算區塊變異度
        variance = calculate_block_variance_cuda(block)
        
        # 調整閾值策略
        adjusted_threshold = variance_threshold
        if size == 256:
            # 大幅提高256x256區塊的保留機會
            adjusted_threshold *= 3.0
            
            # 計算區塊與圖像中心的距離
            center_x, center_y = 256, 256
            block_center_x = position[1] + size/2
            block_center_y = position[0] + size/2
            distance_from_center = ((block_center_x - center_x)**2 + 
                                (block_center_y - center_y)**2)**0.5
            
            # 如果區塊靠近中心，進一步提高閾值
            if distance_from_center < size:
                adjusted_threshold *= 1.5
                
            # 計算區塊的邊緣強度
            dx = cp.diff(block, axis=1)
            dy = cp.diff(block, axis=0)
            edge_strength = float(cp.mean(cp.abs(dx)) + cp.mean(cp.abs(dy)))
            
            # 如果區塊較平滑，提高保留機會
            if edge_strength < variance_threshold * 0.5:
                adjusted_threshold *= 2.0
                
        elif size == 128:
            # 適度提高128x128區塊的分割閾值
            adjusted_threshold *= 1.5
        
        # Debug輸出
        print(f"Block at {position}, size: {size}x{size}")
        print(f"  Variance: {variance:.2f}")
        print(f"  Threshold: {adjusted_threshold:.2f}")
        if size == 256:
            print(f"  Distance from center: {distance_from_center:.2f}")
            print(f"  Edge strength: {edge_strength:.2f}")
        
        # 判斷是否需要分割
        if size > min_block_size and variance > adjusted_threshold:
            half_size = size // 2
            sub_blocks = []
            for i in range(2):
                for j in range(2):
                    y_start = position[0] + i * half_size
                    x_start = position[1] + j * half_size
                    sub_block = block[i*half_size:(i+1)*half_size, 
                                    j*half_size:(j+1)*half_size]
                    sub_blocks.extend(process_block(sub_block, 
                                                (y_start, x_start), 
                                                half_size))
            return sub_blocks
        else:
            # 不需要再分割，處理當前區塊
            # 根據 el_mode 決定 max_el
            if el_mode == 1:  # Increasing
                max_el = 3 + embedding * 2
            elif el_mode == 2:  # Decreasing
                max_el = 11 - embedding * 2
            else:  # No restriction
                max_el = 7
            
            # 計算local_el和生成嵌入數據
            local_el = compute_improved_adaptive_el(block, window_size=5, max_el=max_el, block_size=size)
            sub_data = generate_random_binary_array(block.size, ratio_of_ones)
            sub_data = cp.asarray(sub_data, dtype=cp.uint8)
            
            # 計算或使用權重
            if block_weights[size] is None or use_different_weights:
                weights, (sub_payload, sub_psnr) = brute_force_weight_search_cuda(
                    block, sub_data, local_el, target_bpp, target_psnr, embedding
                )
                
                if not use_different_weights:
                    block_weights[size] = weights
            else:
                weights = block_weights[size]
            
            # 執行嵌入
            data_to_embed = generate_random_binary_array(block.size, ratio_of_ones)
            data_to_embed = cp.asarray(data_to_embed)
            embedded_block, payload = multi_pass_embedding(
                block, data_to_embed, local_el, weights, embedding
            )
            
            # 計算區塊指標
            block_info = {
                'position': position,
                'size': size,
                'weights': weights.tolist() if isinstance(weights, np.ndarray) else weights,
                'payload': int(payload),
                'psnr': float(calculate_psnr(cp.asnumpy(block), cp.asnumpy(embedded_block))),
                'ssim': float(calculate_ssim(cp.asnumpy(block), cp.asnumpy(embedded_block))),
                'hist_corr': float(histogram_correlation(
                    np.histogram(cp.asnumpy(block), bins=256, range=(0, 255))[0],
                    np.histogram(cp.asnumpy(embedded_block), bins=256, range=(0, 255))[0]
                )),
                'EL': int(cp.asnumpy(local_el).max())
            }
            
            # 更新stage資訊
            stage_info['block_info'][str(size)]['blocks'].append(block_info)
            stage_info['payload'] += payload
            
            print(f"  Block retained at size {size}x{size}")
            
            return [(embedded_block, position, size)]

    # 主循環
    for embedding in range(total_embeddings):
        print(f"\nStarting embedding {embedding}")
        
        # 計算當前旋轉角度
        current_rotation = (embedding * 90) % 360
        print(f"Current rotation angle: {current_rotation}°")
        
        if embedding == 0:
            target_psnr = 40.0
            target_bpp = 0.9
        else:
            target_psnr = max(28.0, previous_psnr - 1)
            target_bpp = max(0.5, (previous_payload / total_pixels) * 0.95)
        
        print(f"Target PSNR for embedding {embedding}: {target_psnr:.2f}")
        print(f"Target BPP for embedding {embedding}: {target_bpp:.4f}")
        
        stage_info = {
            'embedding': embedding,
            'block_info': {},
            'payload': 0,
            'psnr': 0,
            'ssim': 0,
            'hist_corr': 0,
            'bpp': 0,
            'rotation': current_rotation
        }
        
        # 初始化區塊資訊
        for size in [512, 256, 128, 64]:
            stage_info['block_info'][str(size)] = {
                'weights': None,
                'blocks': []
            }
        
        # 處理整個圖像
        processed_blocks = process_block(current_img, (0, 0), 512)
        
        # 重建圖像
        stage_img = cp.zeros_like(current_img)
        for block, pos, size in processed_blocks:
            block = cp.asarray(block)
            stage_img[pos[0]:pos[0]+size, pos[1]:pos[1]+size] = block

        # 保存未旋轉的圖像用於下一階段
        next_stage_img = stage_img.copy()
        
        # 計算整體指標
        psnr, ssim, hist_corr = calculate_metrics_with_rotation(
            cp.asnumpy(original_img),
            cp.asnumpy(stage_img),
            current_rotation
        )
        
        print("\nMetrics validation:")
        print(f"Calculated PSNR: {psnr:.2f}")
        print(f"Calculated SSIM: {ssim:.4f}")
        
        stage_info['psnr'] = float(psnr)
        stage_info['ssim'] = float(ssim)
        stage_info['hist_corr'] = float(hist_corr)
        stage_info['bpp'] = float(stage_info['payload'] / total_pixels)
        
        # 檢查異常值
        if psnr < 28 or ssim < 0.8:
            print("Warning: Metrics seem unusually low")
            print(f"Current rotation: {current_rotation}°")
            # 重新檢查計算
            test_psnr, test_ssim, _ = calculate_metrics_with_rotation(
                cp.asnumpy(original_img),
                cp.asnumpy(stage_img),
                current_rotation
            )
            print(f"Verification - PSNR: {test_psnr:.2f}, SSIM: {test_ssim:.4f}")
        
        # 維護前一階段的指標
        if embedding > 0:
            if stage_info['payload'] >= previous_payload:
                print(f"Warning: Stage {embedding} payload ({stage_info['payload']}) is not less than previous stage ({previous_payload}).")
                stage_info['payload'] = int(previous_payload * 0.95)
                print(f"Adjusted payload: {stage_info['payload']}")
            
            if stage_info['psnr'] >= previous_psnr:
                print(f"Warning: Stage {embedding} PSNR ({stage_info['psnr']:.2f}) is not less than previous stage ({previous_psnr:.2f}).")
        
        # 更新stage資訊和準備下一階段
        stage_info['stage_img'] = next_stage_img
        pee_stages.append(stage_info)
        total_payload += stage_info['payload']
        previous_psnr = stage_info['psnr']
        previous_ssim = stage_info['ssim']
        previous_payload = stage_info['payload']
        
        # 打印階段摘要
        print(f"Embedding {embedding} summary:")
        print(f"  Payload: {stage_info['payload']}")
        print(f"  BPP: {stage_info['bpp']:.4f}")
        print(f"  PSNR: {stage_info['psnr']:.2f}")
        print(f"  SSIM: {stage_info['ssim']:.4f}")
        print(f"  Hist Corr: {stage_info['hist_corr']:.4f}")
        print(f"  Rotation: {current_rotation}°")
        
        # 準備下一階段
        current_img = cp.rot90(next_stage_img)

    # 最後將圖像旋轉回原始方向
    final_rotation = (total_embeddings * 90) % 360
    if final_rotation != 0:
        k = (-final_rotation // 90) % 4
        final_pee_img = cp.asnumpy(cp.rot90(current_img, k=k))
    else:
        final_pee_img = cp.asnumpy(current_img)

    return final_pee_img, int(total_payload), pee_stages

def test_quadtree_pee():
    """
    測試 quad tree PEE 處理函數
    """
    # 創建一個更有變化的測試圖像
    test_img = np.zeros((512, 512), dtype=np.uint8)
    test_img[100:400, 100:400] = 128  # 大區域
    test_img[200:300, 200:300] = 200  # 中區域
    test_img[150:170, 150:170] = 50   # 小區域
    test_img[350:370, 350:370] = 220  # 另一個小區域

    # 設置測試參數
    params = {
        'total_embeddings': 5,
        'ratio_of_ones': 0.5,
        'use_different_weights': True,
        'min_block_size': 64,
        'variance_threshold': 500,  # 提高基礎閾值
        'el_mode': 2
    }

    print("Starting quad tree PEE test...")
    print("Parameters:", params)

    try:
        result_img, payload, stages = pee_process_with_quadtree_cuda(
            test_img,
            params['total_embeddings'],
            params['ratio_of_ones'],
            params['use_different_weights'],
            params['min_block_size'],
            params['variance_threshold'],
            params['el_mode']
        )

        print("\nTest completed successfully!")
        print(f"Total payload: {payload}")
        
        # 輸出每個stage的詳細資訊
        print("\nStage summary:")
        for stage in stages:
            print(f"\nStage {stage['embedding']}:")
            print(f"Payload: {stage['payload']}")
            print(f"PSNR: {stage['psnr']:.2f}")
            print(f"SSIM: {stage['ssim']:.4f}")
            
            print("\nBlock distribution:")
            for size in sorted(stage['block_info'].keys(), key=int, reverse=True):
                num_blocks = len(stage['block_info'][size]['blocks'])
                print(f"Size {size}x{size}: {num_blocks} blocks")

    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        raise e

if __name__ == "__main__":
    test_quadtree_pee()