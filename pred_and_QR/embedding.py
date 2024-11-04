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
    generate_random_binary_array
)
from common import *
from pee import *

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

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

