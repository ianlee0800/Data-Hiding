import numpy as np
import cupy as cp
from pee import (
    multi_pass_embedding,
    compute_improved_adaptive_el,
    brute_force_weight_search_cuda
)

from utils import (
    generate_random_binary_array
)

from common import *

from image_processing import (
    PredictionMethod
)

def cleanup_quadtree_resources():
    """
    清理 quadtree 處理過程中使用的資源
    """
    try:
        # 清理 GPU 記憶體
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as e:
        print(f"Error cleaning up quadtree resources: {str(e)}")
        
def process_current_block(block, position, size, stage_info, embedding, ratio_of_ones,
                         target_bpp, target_psnr, el_mode, prediction_method=PredictionMethod.PROPOSED,
                         remaining_target=None, verbose=False):
    """
    處理當前區塊的 PEE 嵌入，支援多種預測方法
    """
    try:
        # 確保 block 是 CuPy 數組
        if not isinstance(block, cp.ndarray):
            block = cp.asarray(block)
        
        # 計算區塊大小和目標容量
        block_size = block.size
        if remaining_target is not None:
            block_size = min(block_size, remaining_target)
        
        # 首先根據 el_mode 決定初始 max_el 值
        if el_mode == 1:  # Increasing
            max_el = 3 + embedding * 2
        elif el_mode == 2:  # Decreasing
            max_el = 11 - embedding * 2
        else:  # No restriction
            max_el = 7
            
        # 然後根據區塊大小調整參數
        if size <= 32:
            max_el = min(max_el, 5)  # 現在可以安全地使用 max_el
            local_target_bpp = target_bpp * 0.8
            local_target_psnr = target_psnr + 2
        else:
            local_target_bpp = target_bpp
            local_target_psnr = target_psnr
        
        # 計算改進的 local_el
        local_el = compute_improved_adaptive_el(
            block, 
            window_size=min(5, size//4),
            max_el=max_el,  # 使用已經正確初始化的 max_el
            block_size=size
        )
        
        # 生成嵌入數據
        data_to_embed = generate_random_binary_array(block_size, ratio_of_ones)
        data_to_embed = cp.asarray(data_to_embed, dtype=cp.uint8)
        
        # 根據預測方法進行不同的處理
        if prediction_method == PredictionMethod.PROPOSED:
            # PROPOSED 方法需要計算權重
            weights, (sub_payload, sub_psnr) = brute_force_weight_search_cuda(
                block, data_to_embed, local_el, local_target_bpp, local_target_psnr, 
                embedding, block_size=size
            )
        else:
            # MED 和 GAP 方法不需要權重
            weights = None
            
        # 執行數據嵌入
        embedded_block, payload = multi_pass_embedding(
            block,
            data_to_embed,
            local_el,
            weights,
            embedding,
            prediction_method=prediction_method,
            remaining_target=remaining_target
        )
        
        # 確保結果是 CuPy 數組
        if not isinstance(embedded_block, cp.ndarray):
            embedded_block = cp.asarray(embedded_block)
        
        # 計算並記錄區塊資訊
        block_info = {
            'position': position,
            'size': size,
            'weights': (weights.tolist() if weights is not None and hasattr(weights, 'tolist') 
                       else "N/A" if prediction_method in [PredictionMethod.MED, PredictionMethod.GAP] 
                       else None),
            'payload': int(payload),
            'psnr': float(calculate_psnr(cp.asnumpy(block), cp.asnumpy(embedded_block))),
            'ssim': float(calculate_ssim(cp.asnumpy(block), cp.asnumpy(embedded_block))),
            'hist_corr': float(histogram_correlation(
                np.histogram(cp.asnumpy(block), bins=256, range=(0, 255))[0],
                np.histogram(cp.asnumpy(embedded_block), bins=256, range=(0, 255))[0]
            )),
            'EL': int(to_numpy(local_el).max()),
            'prediction_method': prediction_method.value
        }
        
        # 更新階段資訊
        stage_info['block_info'][str(size)]['blocks'].append(block_info)
        stage_info['payload'] += payload
        
        if verbose:
            print(f"  Block processed at size {size}x{size}")
            print(f"  Prediction method: {prediction_method.value}")
            print(f"  Payload: {payload}")
            print(f"  PSNR: {block_info['psnr']:.2f}")
        
        return [(embedded_block, position, size)]
            
    except Exception as e:
        print(f"Error in block processing: {str(e)}")
        raise

def process_block(block, position, size, stage_info, embedding, variance_threshold, 
                 ratio_of_ones, target_bpp, target_psnr, el_mode, verbose=False,
                 rotation_mode='none', prediction_method=PredictionMethod.PROPOSED,
                 remaining_target=None):
    """
    遞迴處理區塊，決定是否需要進一步分割，支援多種預測方法
    
    Parameters:
    -----------
    block : cupy.ndarray
        輸入區塊
    position : tuple
        區塊在原圖中的位置 (y, x)
    size : int
        區塊大小
    stage_info : dict
        當前階段的資訊
    embedding : int
        當前嵌入階段
    variance_threshold : float
        變異度閾值
    ratio_of_ones : float
        嵌入數據中 1 的比例
    target_bpp : float
        目標 BPP
    target_psnr : float
        目標 PSNR
    el_mode : int
        EL模式
    verbose : bool
        是否輸出詳細資訊
    rotation_mode : str
        'none': 原始的 quadtree 方法
        'random': 使用隨機旋轉的新方法
    prediction_method : PredictionMethod
        預測方法選擇 (PROPOSED, MED, GAP)
    remaining_target : int, optional
        剩餘需要嵌入的數據量
    """
    try:
        if size < 16:  # 最小區塊大小限制
            return []
        
        # 確保 block 是 CuPy 數組
        if not isinstance(block, cp.ndarray):
            block = cp.asarray(block)
        
        # 計算區塊變異度
        variance = calculate_block_variance_cuda(block)
        
        # 根據區塊大小調整閾值
        adjusted_threshold = variance_threshold
        if size >= 128:
            adjusted_threshold *= 1.2
        elif size >= 64:
            adjusted_threshold *= 1.1
        elif size >= 32:
            adjusted_threshold *= 1.0
        else:  # 16x16 區塊
            adjusted_threshold *= 0.9
        
        # 計算邊緣強度以優化分割決策
        dx = cp.diff(block, axis=1)
        dy = cp.diff(block, axis=0)
        edge_strength = float(cp.mean(cp.abs(dx)) + cp.mean(cp.abs(dy)))
        
        if edge_strength > variance_threshold * 0.3:
            adjusted_threshold *= 0.9  # 邊緣區域更容易被分割
        
        if verbose:
            print(f"Block at {position}, size: {size}x{size}")
            print(f"  Variance: {variance:.2f}")
            print(f"  Edge strength: {edge_strength:.2f}")
            print(f"  Adjusted threshold: {adjusted_threshold:.2f}")
        
        # 根據變異度決定是否分割
        if size > 16 and variance > adjusted_threshold:
            # 繼續分割為四個子區塊
            half_size = size // 2
            sub_blocks = []
            
            for i in range(2):
                for j in range(2):
                    y_start = position[0] + i * half_size
                    x_start = position[1] + j * half_size
                    sub_block = block[i*half_size:(i+1)*half_size, 
                                    j*half_size:(j+1)*half_size]
                    
                    # 遞迴處理子區塊，傳遞所有必要參數
                    sub_blocks.extend(process_block(
                        sub_block, (y_start, x_start), half_size,
                        stage_info, embedding, variance_threshold,
                        ratio_of_ones, target_bpp, target_psnr, el_mode,
                        verbose=verbose,
                        rotation_mode=rotation_mode,
                        prediction_method=prediction_method,
                        remaining_target=remaining_target
                    ))
            return sub_blocks
        else:
            # 處理當前區塊
            if rotation_mode == 'random' and 'block_rotations' in stage_info:
                rotation = stage_info['block_rotations'][size]
                if rotation != 0:
                    k = rotation // 90
                    block = cp.rot90(block, k=k)
            
            # 使用process_current_block處理當前區塊
            return process_current_block(
                block, position, size, stage_info, embedding,
                ratio_of_ones, target_bpp, target_psnr, el_mode,
                prediction_method=prediction_method,
                remaining_target=remaining_target,
                verbose=verbose
            )
            
    except Exception as e:
        print(f"Error in block processing at position {position}, size {size}: {str(e)}")
        raise

def pee_process_with_quadtree_cuda(img, total_embeddings, ratio_of_ones, use_different_weights, 
                                 min_block_size, variance_threshold, el_mode, 
                                 prediction_method=PredictionMethod.PROPOSED,
                                 rotation_mode='none',
                                 target_payload_size=-1):
    """
    使用Quad tree的PEE處理函數，支援多種預測方法和payload控制
    
    Parameters:
    -----------
    img : numpy.ndarray
        輸入圖像
    total_embeddings : int
        總嵌入次數
    ratio_of_ones : float
        嵌入數據中1的比例
    use_different_weights : bool
        是否對每個子圖像使用不同的權重 (僅用於 PROPOSED 方法)
    min_block_size : int
        最小區塊大小 (支援到16x16)
    variance_threshold : float
        變異閾值
    el_mode : int
        EL模式 (0:無限制, 1:漸增, 2:漸減)
    prediction_method : PredictionMethod
        預測方法選擇 (PROPOSED, MED, GAP)
    rotation_mode : str
        'none': 原始的 quadtree 方法
        'random': 使用隨機旋轉的新方法
    target_payload_size : int
        目標總payload大小，設為-1時使用最大容量
        
    Returns:
    --------
    tuple
        (final_pee_img, total_payload, pee_stages)
        final_pee_img: 最終處理後的圖像
        total_payload: 總嵌入容量
        pee_stages: 包含每個階段詳細資訊的列表
    """
    try:
        # 參數合法性檢查
        if min_block_size < 16:
            raise ValueError("min_block_size must be at least 16")
        if min_block_size & (min_block_size - 1) != 0:
            raise ValueError("min_block_size must be a power of 2")
        if not 0 <= ratio_of_ones <= 1:
            raise ValueError("ratio_of_ones must be between 0 and 1")
        if rotation_mode not in ['none', 'random']:
            raise ValueError("rotation_mode must be 'none' or 'random'")
        
        # 預測方法相關設置
        if prediction_method in [PredictionMethod.MED, PredictionMethod.GAP]:
            use_different_weights = False
            print(f"Note: Weight optimization disabled for {prediction_method.value} prediction method")

        # 初始化基本變數
        original_img = cp.asarray(img)
        height, width = original_img.shape
        total_pixels = height * width
        pee_stages = []
        total_payload = 0
        current_img = original_img.copy()
        
        # 追蹤變數初始化
        previous_psnr = float('inf')
        previous_ssim = 1.0
        previous_payload = float('inf')
        
        # Payload 控制設置
        remaining_target = target_payload_size if target_payload_size > 0 else None
        
        # GPU 記憶體管理
        mem_pool = cp.get_default_memory_pool()

        try:
            # 逐階段處理
            for embedding in range(total_embeddings):
                # 檢查是否達到目標 payload
                if remaining_target is not None and remaining_target <= 0:
                    print(f"Reached target payload ({target_payload_size} bits) at stage {embedding}")
                    break
                    
                # 輸出階段開始資訊
                print(f"\nStarting embedding {embedding}")
                print(f"Memory usage before embedding {embedding}: {mem_pool.used_bytes()/1024/1024:.2f} MB")
                if remaining_target is not None:
                    print(f"Remaining target payload: {remaining_target}")

                # 設定目標品質參數
                if embedding == 0:
                    target_psnr = 40.0
                    target_bpp = 0.9
                else:
                    target_psnr = max(28.0, previous_psnr - 1)
                    target_bpp = max(0.5, (previous_payload / total_pixels) * 0.95)

                print(f"Target PSNR for embedding {embedding}: {target_psnr:.2f}")
                print(f"Target BPP for embedding {embedding}: {target_bpp:.4f}")

                # 初始化階段資訊
                stage_info = {
                    'embedding': embedding,
                    'block_info': {str(size): {'blocks': []} for size in [512, 256, 128, 64, 32, 16]},
                    'payload': 0,
                    'psnr': 0,
                    'ssim': 0,
                    'hist_corr': 0,
                    'bpp': 0,
                    'rotation_mode': rotation_mode,
                    'prediction_method': prediction_method.value
                }

                # 旋轉模式設置
                if rotation_mode == 'random':
                    # 為每個區塊大小生成隨機旋轉角度
                    block_rotations = {
                        size: np.random.choice([-270, -180, -90, 0, 90, 180, 270])
                        for size in [512, 256, 128, 64, 32, 16]
                    }
                    stage_info['block_rotations'] = block_rotations
                    print("\nBlock rotation angles for this stage:")
                    for size, angle in sorted(block_rotations.items(), reverse=True):
                        print(f"  {size}x{size}: {angle}°")

                # 處理整個圖像
                processed_blocks = process_block(
                    current_img, (0, 0), 512, stage_info, embedding,
                    variance_threshold, ratio_of_ones, target_bpp, target_psnr, el_mode,
                    verbose=False, 
                    rotation_mode=rotation_mode,
                    prediction_method=prediction_method,
                    remaining_target=remaining_target
                )

                # 初始化輸出圖像
                stage_img = cp.zeros_like(current_img)
                if rotation_mode == 'random':
                    rotated_stage_img = cp.zeros_like(current_img)

                # 重建圖像
                for block, pos, size in processed_blocks:
                    block = cp.asarray(block)
                    
                    if rotation_mode == 'random':
                        # 保存旋轉狀態的圖像
                        rotated_stage_img[pos[0]:pos[0]+size, pos[1]:pos[1]+size] = block
                        # 將區塊旋轉回原始方向
                        rotation = stage_info['block_rotations'][size]
                        if rotation != 0:
                            k = (-rotation // 90) % 4
                            block = cp.rot90(block, k=k)
                    
                    # 將區塊放回最終圖像
                    stage_img[pos[0]:pos[0]+size, pos[1]:pos[1]+size] = block

                # 保存圖像狀態
                if rotation_mode == 'random':
                    stage_info['rotated_stage_img'] = rotated_stage_img
                stage_info['stage_img'] = stage_img

                # 計算品質指標
                if rotation_mode == 'random':
                    # 使用轉回0度的圖像計算指標
                    stage_img_np = cp.asnumpy(stage_img)
                    reference_img_np = cp.asnumpy(original_img)
                    psnr = calculate_psnr(reference_img_np, stage_img_np)
                    ssim = calculate_ssim(reference_img_np, stage_img_np)
                    hist_corr = histogram_correlation(
                        np.histogram(reference_img_np, bins=256, range=(0, 255))[0],
                        np.histogram(stage_img_np, bins=256, range=(0, 255))[0]
                    )
                else:
                    # 使用原始旋轉狀態計算指標
                    psnr, ssim, hist_corr = calculate_metrics_with_rotation(
                        current_img,
                        stage_img,
                        original_img,
                        embedding
                    )

                # 更新階段品質指標
                stage_info['psnr'] = float(psnr)
                stage_info['ssim'] = float(ssim)
                stage_info['hist_corr'] = float(hist_corr)
                stage_info['bpp'] = float(stage_info['payload'] / total_pixels)

                # 品質檢查和警告
                if psnr < 28 or ssim < 0.8:
                    print("Warning: Metrics seem unusually low")

                # 與前一階段比較
                if embedding > 0:
                    if stage_info['payload'] >= previous_payload:
                        print(f"Warning: Stage {embedding} payload ({stage_info['payload']}) is not less than previous stage ({previous_payload}).")
                        stage_info['payload'] = int(previous_payload * 0.95)
                        print(f"Adjusted payload: {stage_info['payload']}")

                    if stage_info['psnr'] >= previous_psnr:
                        print(f"Warning: Stage {embedding} PSNR ({stage_info['psnr']:.2f}) is not less than previous stage ({previous_psnr:.2f}).")

                # 更新總體資訊
                pee_stages.append(stage_info)
                total_payload += stage_info['payload']
                previous_psnr = stage_info['psnr']
                previous_ssim = stage_info['ssim']
                previous_payload = stage_info['payload']
                
                # 更新剩餘目標payload
                if remaining_target is not None:
                    remaining_target -= stage_info['payload']

                # 輸出階段摘要
                print(f"\nEmbedding {embedding} summary:")
                print(f"Prediction Method: {prediction_method.value}")
                print(f"Payload: {stage_info['payload']}")
                print(f"BPP: {stage_info['bpp']:.4f}")
                print(f"PSNR: {stage_info['psnr']:.2f}")
                print(f"SSIM: {stage_info['ssim']:.4f}")
                print(f"Hist Corr: {stage_info['hist_corr']:.4f}")

                # 準備下一階段
                current_img = stage_img.copy()

                # 清理當前階段的記憶體
                mem_pool.free_all_blocks()
                print(f"Memory usage after embedding {embedding}: {mem_pool.used_bytes()/1024/1024:.2f} MB")

            # 返回最終結果
            return cp.asnumpy(current_img), int(total_payload), pee_stages

        except Exception as e:
            print(f"Error in embedding process: {str(e)}")
            raise

    except Exception as e:
        print(f"Error in quadtree processing: {str(e)}")
        raise

    finally:
        # 確保清理所有記憶體
        cleanup_quadtree_resources()

