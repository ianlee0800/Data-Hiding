import numpy as np
import cupy as cp
import math
from pee import (
    multi_pass_embedding,
    compute_improved_adaptive_el,
    brute_force_weight_search_cuda
)

from utils import (
    generate_random_binary_array,
    ensure_dir
)

from common import *

from image_processing import (
    PredictionMethod,
    save_image
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

def estimate_block_capacity(block, size, variance=None, edge_strength=None):
    """
    更精確地估計區塊的嵌入容量
    
    Parameters:
    -----------
    block : numpy.ndarray or cupy.ndarray
        區塊圖像數據
    size : int
        區塊大小
    variance : float, optional
        已計算的變異度，如不提供將重新計算
    edge_strength : float, optional
        已計算的邊緣強度，如不提供將重新計算
    """
    # 確保區塊是NumPy數組
    if isinstance(block, cp.ndarray):
        block_np = cp.asnumpy(block)
    else:
        block_np = block
    
    # 如未提供，計算變異度
    if variance is None:
        variance = np.var(block_np)
    
    # 如未提供，計算邊緣強度
    if edge_strength is None:
        dx = np.diff(block_np, axis=1)
        dy = np.diff(block_np, axis=0)
        edge_strength = np.mean(np.abs(dx)) + np.mean(np.abs(dy))
    
    # 區塊面積
    area = size * size
    
    # 計算紋理複雜度 (熵)
    hist, _ = np.histogram(block_np, bins=32, range=(0, 255))
    hist = hist / hist.sum()
    non_zero = hist > 0
    entropy = -np.sum(hist[non_zero] * np.log2(hist[non_zero]))
    entropy_normalized = entropy / 5.0  # 標準化，5.0是大約的最大熵
    
    # 基礎容量計算 - 考慮熵、變異度和邊緣
    # 高熵、高變異度和高邊緣區域容量較低
    base_rate = 0.7 - 0.2 * entropy_normalized - 0.1 * min(1.0, variance/500) - 0.1 * min(1.0, edge_strength/30)
    base_rate = max(0.1, min(0.7, base_rate))  # 限制在0.1-0.7之間
    
    # 基本容量
    estimated_capacity = int(area * base_rate)
    
    # 區塊大小調整
    if size <= 16:
        estimated_capacity = int(estimated_capacity * 0.8)  # 小區塊容量較低
    elif size >= 256:
        estimated_capacity = int(estimated_capacity * 1.2)  # 大區塊容量較高
    
    return max(1, estimated_capacity)  # 確保至少返回1

def adjust_quadtree_params(target_payload, max_payload, base_variance_threshold=300, base_min_block_size=16):
    """
    根據目標 payload 動態調整四叉樹參數
    
    Parameters:
    -----------
    target_payload : int
        目標 payload 大小，單位為 bits
    max_payload : int
        最大可能的 payload 大小
    base_variance_threshold : int
        基礎變異度閾值
    base_min_block_size : int
        基礎最小區塊大小
        
    Returns:
    --------
    tuple
        (adjusted_variance_threshold, adjusted_min_block_size)
    """
    # 計算目標與最大容量的比例
    if max_payload <= 0:
        ratio = 1.0  # 預設值
    else:
        ratio = min(1.0, target_payload / max_payload)
    
    # 調整變異度閾值 - 較小的目標使用較大的閾值（減少分割）
    adjusted_variance = base_variance_threshold * (1 + 1.0 * (1 - ratio))
    
    # 調整最小區塊大小 - 較小的目標使用較大的最小區塊大小
    if ratio < 0.2:  # 低 payload (< 20%)
        adjusted_min_block_size = 32
    elif ratio < 0.4:  # 中低 payload (20-40%)
        adjusted_min_block_size = 24  # 這需要修改代碼以支援非2的冪次區塊大小，否則使用32
    elif ratio < 0.6:  # 中等 payload (40-60%)
        adjusted_min_block_size = base_min_block_size  # 通常是16
    elif ratio < 0.8:  # 中高 payload (60-80%)
        adjusted_min_block_size = base_min_block_size  # 保持16
    else:  # 高 payload (> 80%)
        adjusted_min_block_size = base_min_block_size  # 保持16，或考慮降到8以增加容量
    
    # 確保最小區塊大小是2的冪次
    # 如果不支援非2的冪次大小，取最接近的2的冪次
    power = math.log2(adjusted_min_block_size)
    if power != int(power):
        adjusted_min_block_size = 2 ** int(round(power))
    
    # 確保合理範圍
    adjusted_variance = max(50, min(1000, adjusted_variance))
    adjusted_min_block_size = max(8, min(64, adjusted_min_block_size))
    
    return adjusted_variance, adjusted_min_block_size

def process_current_block(block, position, size, stage_info, embedding, ratio_of_ones,
                         target_bpp, target_psnr, el_mode, prediction_method=PredictionMethod.PROPOSED,
                         remaining_target=None, verbose=False, embed_data=True):
    """
    Process current block PEE embedding with enhanced control based on remaining target
    """
    try:
        # Store original block for verification
        original_block = None
        if isinstance(block, cp.ndarray):
            original_block = block.copy()
        else:
            original_block = cp.asarray(block).copy()
            
        # 在正式嵌入前檢查剩餘目標
        # 如果沒有足夠的剩餘容量需求，可以減少嵌入強度或完全跳過嵌入
        if not embed_data or (remaining_target is not None and remaining_target <= 0):
            block_info = {
                'position': position,
                'size': size,
                'weights': "N/A" if prediction_method in [PredictionMethod.MED, PredictionMethod.GAP, PredictionMethod.RHOMBUS] else None,
                'payload': 0,  # No embedding, so payload is 0
                'psnr': float('inf'),
                'ssim': 1.0,
                'hist_corr': 1.0,
                'EL': 0,
                'prediction_method': prediction_method.value,
                'original_block': original_block,
                'embedded_block': original_block
            }
            
            # Update stage info - don't increase payload
            stage_info['block_info'][str(size)]['blocks'].append(block_info)
            
            # Return original block
            return [(block, position, size)]
        
        # 確保區塊是 CuPy 數組
        if not isinstance(block, cp.ndarray):
            block = cp.asarray(block)
        
        # 計算區塊大小和目標容量
        block_size = block.size
        
        # 改進的分層目標控制
        if remaining_target is not None:
            # 1. 容量分層控制
            if remaining_target <= 0:
                # 完全跳過嵌入
                target_for_block = 0
            elif remaining_target < block_size * 0.1:
                # 極小目標，嚴格限制
                target_for_block = min(remaining_target, int(block_size * 0.05))
                reduction_factor = 0.4
            elif remaining_target < block_size * 0.3:
                # 小目標，中度限制
                target_for_block = min(remaining_target, int(block_size * 0.2))
                reduction_factor = 0.6
            elif remaining_target < block_size * 0.6:
                # 中等目標，輕度限制
                target_for_block = min(remaining_target, int(block_size * 0.4))
                reduction_factor = 0.8
            else:
                # 大目標，幾乎不限制
                target_for_block = min(remaining_target, block_size)
                reduction_factor = 0.95
                
            # 應用安全係數，避免超額嵌入
            target_for_block = int(target_for_block * reduction_factor)
            
            if verbose:
                print(f"區塊 {position}: 大小={block_size}，目標嵌入={target_for_block} (原始={remaining_target})")
        else:
            # 無目標限制
            target_for_block = block_size
        
        # 根據嵌入目標動態調整嵌入層級
        if target_for_block <= 0:
            # 不嵌入
            max_el = 0
        elif target_for_block < block_size * 0.1:
            # 極小嵌入量，使用最小層級
            max_el = 1
        elif target_for_block < block_size * 0.3:
            # 小嵌入量，使用較低層級
            max_el = max(1, min(3, max_el - 4))
        elif target_for_block < block_size * 0.6:
            # 中等嵌入量，適度降低層級
            max_el = max(3, min(5, max_el - 2))
        # 其他情況使用原始 max_el
        
        # 根據區塊大小進一步調整
        if size <= 32:
            max_el = min(max_el, 5)  # 限制小區塊的嵌入層級
            local_target_bpp = target_bpp * 0.8
            local_target_psnr = target_psnr + 2
        else:
            local_target_bpp = target_bpp
            local_target_psnr = target_psnr
        
        # 計算改進的 local_el
        local_el = compute_improved_adaptive_el(
            block, 
            window_size=min(5, size//4),
            max_el=max_el,
            block_size=size
        )
        
        # 生成嵌入數據
        data_to_embed = generate_random_binary_array(target_for_block, ratio_of_ones)
        data_to_embed = cp.asarray(data_to_embed, dtype=cp.uint8)
        
        # 根據預測方法處理
        if prediction_method == PredictionMethod.PROPOSED:
            if 'use_different_weights' in stage_info and stage_info['use_different_weights']:
                weights, (sub_payload, sub_psnr) = brute_force_weight_search_cuda(
                    block, data_to_embed, local_el, local_target_bpp, local_target_psnr, 
                    embedding, block_size=size
                )
            else:
                size_str = str(size)
                if size_str in stage_info['block_size_weights']:
                    weights = np.array(stage_info['block_size_weights'][size_str], dtype=np.int32)
                else:
                    weights, (sub_payload, sub_psnr) = brute_force_weight_search_cuda(
                        block, data_to_embed, local_el, local_target_bpp, local_target_psnr, 
                        embedding, block_size=size
                    )
                    if hasattr(weights, 'tolist'):
                        stage_info['block_size_weights'][size_str] = weights.tolist()
                    else:
                        stage_info['block_size_weights'][size_str] = weights
        else:
            weights = None
            
        # 執行數據嵌入
        embedded_block, payload = multi_pass_embedding(
            block,
            data_to_embed,
            local_el,
            weights,
            embedding,
            prediction_method=prediction_method,
            remaining_target=target_for_block  # 使用調整後的目標
        )
        
        # 確保結果是 CuPy 數組
        if not isinstance(embedded_block, cp.ndarray):
            embedded_block = cp.asarray(embedded_block)
        
        # 計算和記錄區塊資訊
        block_np = cp.asnumpy(block)
        embedded_block_np = cp.asnumpy(embedded_block)
        
        # 計算指標
        mse = np.mean((block_np.astype(np.float64) - embedded_block_np.astype(np.float64)) ** 2)
        if mse == 0:
            block_psnr = float('inf')
        else:
            block_psnr = 10 * np.log10((255.0 ** 2) / mse)
            
        block_ssim = calculate_ssim(block_np, embedded_block_np)
        block_hist_corr = histogram_correlation(
            np.histogram(block_np, bins=256, range=(0, 255))[0],
            np.histogram(embedded_block_np, bins=256, range=(0, 255))[0]
        )
        
        # 建立區塊資訊
        block_info = {
            'position': position,
            'size': size,
            'weights': (weights.tolist() if weights is not None and hasattr(weights, 'tolist') 
                       else "N/A" if prediction_method in [PredictionMethod.MED, PredictionMethod.GAP] 
                       else None),
            'payload': int(payload),
            'psnr': float(block_psnr),
            'ssim': float(block_ssim),
            'hist_corr': float(block_hist_corr),
            'EL': int(to_numpy(local_el).max()),
            'prediction_method': prediction_method.value,
            'original_block': original_block,
            'embedded_block': embedded_block
        }
        
        # 更新階段資訊
        stage_info['block_info'][str(size)]['blocks'].append(block_info)
        stage_info['payload'] += payload
        
        # 更新剩餘目標
        if remaining_target is not None and 'remaining_target' in stage_info:
            stage_info['remaining_target'] -= payload
        
        if verbose:
            print(f"  Block processed at size {size}x{size}")
            print(f"  Prediction method: {prediction_method.value}")
            print(f"  Payload: {payload}")
            print(f"  PSNR: {block_info['psnr']:.2f}")
        
        return [(embedded_block, position, size)]
            
    except Exception as e:
        print(f"Error in block processing: {str(e)}")
        import traceback
        traceback.print_exc()
        # 發生錯誤時返回原始區塊
        return [(block, position, size)]

def process_block(block, position, size, stage_info, embedding, variance_threshold, 
                 ratio_of_ones, target_bpp, target_psnr, el_mode, verbose=False,
                 rotation_mode='none', prediction_method=PredictionMethod.PROPOSED,
                 remaining_target=None, max_block_size=1024, embed_data=True):
    """
    遞迴處理區塊，決定是否需要進一步分割，支援多種預測方法，並具有改進的容量控制和優先級機制
    
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
    max_block_size : int
        最大區塊大小，默認為1024
    embed_data : bool
        是否嵌入數據，當達到目標 payload 後設為 False
    """
    try:
        # 初始化 current_remaining 變數
        current_remaining = remaining_target
        
        # 檢查是否已達到目標 payload
        if current_remaining is not None and current_remaining <= 0:
            # 已達到目標，不再處理此區塊
            if verbose:
                print(f"已達到目標 payload，跳過處理位置 {position} 的區塊")
            # 返回原始區塊，不進行嵌入或分割
            return [(block, position, size)]
        
        # 檢查區塊大小限制
        if size < 16:  # 最小區塊大小限制
            return []
        
        # 確保 block 是 CuPy 數組
        if not isinstance(block, cp.ndarray):
            block = cp.asarray(block)
        
        # 計算區塊變異度
        variance = calculate_block_variance_cuda(block)
        
        # 計算邊緣強度以優化分割決策
        dx = cp.diff(block, axis=1)
        dy = cp.diff(block, axis=0)
        edge_strength = float(cp.mean(cp.abs(dx)) + cp.mean(cp.abs(dy)))
        
        # 根據區塊大小調整閾值
        adjusted_threshold = variance_threshold
        if size >= 512:
            adjusted_threshold *= 1.3
        elif size >= 256:
            adjusted_threshold *= 1.2
        elif size >= 128:
            adjusted_threshold *= 1.1
        elif size >= 64:
            adjusted_threshold *= 1.0
        elif size >= 32:
            adjusted_threshold *= 0.9
        else:  # 16x16 區塊
            adjusted_threshold *= 0.8
        
        if edge_strength > variance_threshold * 0.3:
            adjusted_threshold *= 0.9  # 邊緣區域更容易被分割
        
        if verbose:
            print(f"Block at {position}, size: {size}x{size}")
            print(f"  Variance: {variance:.2f}")
            print(f"  Edge strength: {edge_strength:.2f}")
            print(f"  Adjusted threshold: {adjusted_threshold:.2f}")
        
        # 新增優先級評分系統
        if current_remaining is not None and 'remaining_target' in stage_info:
            original_target = stage_info.get('original_target', current_remaining)
            progress_ratio = 1.0 - (current_remaining / original_target) if original_target > 0 else 0
            
            # 計算區塊優先級分數 - 綜合考慮多種因素
            # 1. 變異度評分 - 低變異度區塊優先
            variance_score = 1.0 - min(1.0, variance / 1000)
            
            # 2. 大小評分 - 大區塊在初期優先，小區塊在後期優先
            if progress_ratio < 0.3:  # 前30%嵌入
                size_score = min(1.0, size / 256)  # 大區塊得高分
            elif progress_ratio < 0.7:  # 30-70%嵌入
                size_score = 0.5  # 中性
            else:  # 後30%嵌入
                size_score = 1.0 - min(1.0, size / 256)  # 小區塊得高分
                
            # 3. 邊緣評分 - 低邊緣強度區塊優先
            edge_score = 1.0 - min(1.0, edge_strength / 50)
            
            # 綜合評分，權重可以根據實驗調整
            block_priority = (variance_score * 0.5 + size_score * 0.3 + edge_score * 0.2)
            
            # 以嵌入進度為依據調整臨界點
            threshold_modifier = 0.5 + progress_ratio * 0.5  # 0.5-1.0
            priority_threshold = 0.6 * threshold_modifier  # 隨進度增加而提高
            
            # 估計當前區塊的容量
            estimated_capacity = estimate_block_capacity(block, size, variance, edge_strength)
            
            if block_priority < priority_threshold and current_remaining < estimated_capacity * 2:
                # 優先級不足且剩餘目標不大，跳過處理
                if verbose:
                    print(f"區塊 {position} 優先級過低 ({block_priority:.2f} < {priority_threshold:.2f})，跳過處理")
                return [(block, position, size)]
        
        # 新增估計當前區塊可能的容量，用於決策是否處理
        estimated_capacity = estimate_block_capacity(block, size, variance, edge_strength)
        
        # 決定是否分割
        if current_remaining is not None and estimated_capacity > current_remaining * 2:
            # 如果估計容量遠超過剩餘目標，考慮是否分割為更小區塊
            # 這有助於更精確地控制嵌入量
            should_split = True
        else:
            # 原有的分割邏輯
            should_split = size > 16 and variance > adjusted_threshold
        
        # 根據變異度和目標決定是否分割
        if should_split:
            # 繼續分割為四個子區塊
            half_size = size // 2
            sub_blocks = []
            
            for i in range(2):
                for j in range(2):
                    y_start = position[0] + i * half_size
                    x_start = position[1] + j * half_size
                    sub_block = block[i*half_size:(i+1)*half_size, 
                                    j*half_size:(j+1)*half_size]
                    
                    # 遞迴處理子區塊
                    sub_blocks.extend(process_block(
                        sub_block, (y_start, x_start), half_size,
                        stage_info, embedding, variance_threshold,
                        ratio_of_ones, target_bpp, target_psnr, el_mode,
                        verbose=verbose,
                        rotation_mode=rotation_mode,
                        prediction_method=prediction_method,
                        remaining_target=current_remaining,
                        max_block_size=max_block_size,
                        embed_data=embed_data
                    ))
            return sub_blocks
        else:
            # 處理當前區塊，確保 rotation 執行
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
                remaining_target=current_remaining,
                verbose=verbose,
                embed_data=embed_data
            )
            
    except Exception as e:
        print(f"Error in block processing at position {position}, size {size}: {str(e)}")
        import traceback
        traceback.print_exc()
        # 發生錯誤時，返回原始區塊
        return [(block, position, size)]

def pee_process_with_quadtree_cuda(img, total_embeddings, ratio_of_ones, use_different_weights, 
                                 min_block_size, variance_threshold, el_mode, 
                                 rotation_mode='none',
                                 prediction_method=None,
                                 target_payload_size=-1,
                                 max_block_size=None,
                                 imgName=None,
                                 output_dir=None):
    """
    Using Quad tree's PEE processing function with dynamic parameters adjustment,
    supporting both grayscale and color images.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image (grayscale or color)
    total_embeddings : int
        Total number of embedding stages
    ratio_of_ones : float
        Ratio of ones in embedding data
    use_different_weights : bool
        Whether to use different weights for each sub-image (only for PROPOSED method)
    min_block_size : int
        Minimum block size (supported down to 16x16)
    variance_threshold : float
        Variance threshold
    el_mode : int
        EL mode (0:no restriction, 1:increasing, 2:decreasing)
    prediction_method : PredictionMethod
        Prediction method (PROPOSED, MED, GAP)
    rotation_mode : str
        'none': original quadtree method
        'random': new method using random rotation
    target_payload_size : int
        Target total payload size, set to -1 for maximum capacity
    max_block_size : int, optional
        Maximum block size, defaults to image size half or 512, whichever is larger
    imgName : str, optional
        Image name for saving visualizations
    output_dir : str, optional
        Output directory for saving visualizations
        
    Returns:
    --------
    tuple
        (final_pee_img, total_payload, pee_stages)
        final_pee_img: Final processed image
        total_payload: Total embedded capacity
        pee_stages: List containing detailed information for each stage
    """
    # Check if this is a color image by examining image shape
    if len(img.shape) == 3 and img.shape[2] == 3:
        # For color images, redirect to color image processing function
        print("Detected color image, redirecting to color image processing...")
        return pee_process_color_image_quadtree_cuda(
            img, total_embeddings, ratio_of_ones, use_different_weights,
            min_block_size, variance_threshold, el_mode, rotation_mode,
            prediction_method, target_payload_size, max_block_size,
            imgName, output_dir
        )
    try:
        # Import necessary modules
        import os
        import cv2
        import numpy as np
        import cupy as cp
        import math
        from datetime import datetime
        
        # Create a timestamp for debug folders
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define ensure_dir function
        def ensure_dir(file_path):
            """Ensure directory exists, create if it doesn't"""
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Parameter validation and default values
        # Check and set image name
        if imgName is None:
            if rotation_mode == 'random':
                # Image name needed when using random rotation mode
                imgName = "unknown_image"  # Use default name
                print("Warning: No image name provided. Using 'unknown_image' for saving visualizations.")
            else:
                # Image name not required for non-rotation mode
                imgName = "temp"
        
        # Check and set output directory
        if output_dir is None:
            output_dir = "./Prediction_Error_Embedding/outcome"
            print(f"Using default output directory: {output_dir}")
        
        # Default visualization paths
        image_dir = f"{output_dir}/image/{imgName}/quadtree"
        debug_dir = f"{output_dir}/debug/{imgName}/quadtree_{timestamp}"
        
        # Ensure output directories exist
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(f"{image_dir}/rotated_blocks", exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save original image for verification
        orig_img_copy = img.copy()
        cv2.imwrite(f"{debug_dir}/original_image.png", img)
        
        # Parameter validity check
        if min_block_size < 16:
            raise ValueError("min_block_size must be at least 16")
        if min_block_size & (min_block_size - 1) != 0:
            raise ValueError("min_block_size must be a power of 2")
        if not 0 <= ratio_of_ones <= 1:
            raise ValueError("ratio_of_ones must be between 0 and 1")
        if rotation_mode not in ['none', 'random']:
            raise ValueError("rotation_mode must be 'none' or 'random'")
        
        # Determine maximum block size
        height, width = img.shape
        orig_height, orig_width = height, width  # Keep track of original dimensions
        if max_block_size is None:
            max_block_size = max(512, min(1024, max(height, width)))
        
        # Check if image size is a multiple of max_block_size
        if height % max_block_size != 0 or width % max_block_size != 0:
            # Pad image to suitable size
            new_height = ((height + max_block_size - 1) // max_block_size) * max_block_size
            new_width = ((width + max_block_size - 1) // max_block_size) * max_block_size
            
            # Create new image and copy original data
            padded_img = np.zeros((new_height, new_width), dtype=np.uint8)
            padded_img[:height, :width] = img
            
            # Fill remaining with edge pixels
            if height < new_height:
                padded_img[height:, :width] = padded_img[height-1:height, :width]
            if width < new_width:
                padded_img[:, width:] = padded_img[:, width-1:width]
            
            # Update image and dimensions
            img = padded_img
            print(f"Image resized from original size to {new_height}x{new_width} for quadtree processing")
        
        # Get method name for logging
        method_name = prediction_method.value if hasattr(prediction_method, 'value') else "unknown"
        
        # Store original parameters for later reference
        original_min_block_size = min_block_size
        original_variance_threshold = variance_threshold
        
        # ========== Dynamic Parameter Adjustment (New) ==========
        if target_payload_size > 0:
            # Create data directory for caching max payload info
            data_dir = f"{output_dir}/data/{imgName}"
            os.makedirs(data_dir, exist_ok=True)
            
            max_payload_cache_file = f"{data_dir}/quadtree_{method_name}_max_payload.npy"
            
            # Try to load max payload from cache if available
            estimated_max_payload = 0
            if os.path.exists(max_payload_cache_file):
                try:
                    max_payload_data = np.load(max_payload_cache_file, allow_pickle=True).item()
                    estimated_max_payload = max_payload_data.get('max_payload', 0)
                    print(f"Loaded maximum payload estimate from cache: {estimated_max_payload} bits")
                except Exception as e:
                    print(f"Error loading max payload cache: {str(e)}")
                    estimated_max_payload = 0
            
            # If no valid cache, use conservative estimate
            if estimated_max_payload <= 0:
                # Conservative estimate based on image size
                height, width = img.shape[:2]
                estimated_max_payload = int(height * width * 1.1)  # Assume ~1.1 bits per pixel
                print(f"Using conservative maximum payload estimate: {estimated_max_payload} bits")
            
            # Calculate target to max ratio for parameter adjustment
            ratio = min(1.0, target_payload_size / estimated_max_payload)
            
            # Adjust variance threshold - smaller target uses larger threshold (less splitting)
            adjusted_variance = variance_threshold * (1 + 1.0 * (1 - ratio))
            
            # Adjust minimum block size - smaller target uses larger minimum block
            if ratio < 0.2:  # Low payload (< 20%)
                adjusted_min_block_size = 32
            elif ratio < 0.4:  # Medium-low payload (20-40%)
                adjusted_min_block_size = 32  # Using power of 2
            elif ratio < 0.6:  # Medium payload (40-60%)
                adjusted_min_block_size = original_min_block_size
            elif ratio < 0.8:  # Medium-high payload (60-80%)
                adjusted_min_block_size = original_min_block_size
            else:  # High payload (> 80%)
                adjusted_min_block_size = original_min_block_size
            
            # Ensure minimum block size is power of 2
            power = math.log2(adjusted_min_block_size)
            if power != int(power):
                adjusted_min_block_size = 2 ** int(round(power))
            
            # Ensure parameters are within reasonable range
            adjusted_variance = max(50, min(1000, adjusted_variance))
            adjusted_min_block_size = max(8, min(64, adjusted_min_block_size))
            
            # Output adjusted parameters
            print(f"\n=== Dynamic Parameter Adjustment for Target Payload ({target_payload_size} bits) ===")
            print(f"Original variance threshold: {variance_threshold} -> Adjusted to: {adjusted_variance}")
            print(f"Original minimum block size: {min_block_size} -> Adjusted to: {adjusted_min_block_size}")
            print("=" * 60 + "\n")
            
            # Apply adjusted parameters
            variance_threshold = adjusted_variance
            min_block_size = adjusted_min_block_size
        
        # Prediction method related settings
        if prediction_method in [PredictionMethod.MED, PredictionMethod.GAP]:
            use_different_weights = False
            print(f"Note: Weight optimization disabled for {method_name} prediction method")

        # Initialize basic variables
        original_img = cp.asarray(img)
        height, width = original_img.shape
        total_pixels = height * width
        pee_stages = []
        total_payload = 0
        current_img = original_img.copy()
        
        # Tracking variables
        previous_psnr = float('inf')
        previous_ssim = 1.0
        previous_payload = float('inf')
        
        # Payload control setup
        remaining_target = target_payload_size if target_payload_size > 0 else None
        
        # Store weights for each block size
        block_size_weights = {}
        
        # GPU memory management
        mem_pool = cp.get_default_memory_pool()

        try:
            # Process each stage
            for embedding in range(total_embeddings):
                # Check if target payload reached
                if remaining_target is not None and remaining_target <= 0:
                    print(f"Reached target payload ({target_payload_size} bits) at stage {embedding}")
                    break
                    
                # Output stage start information
                print(f"\nStarting embedding {embedding}")
                print(f"Memory usage before embedding {embedding}: {mem_pool.used_bytes()/1024/1024:.2f} MB")
                if remaining_target is not None:
                    print(f"Remaining target payload: {remaining_target}")

                # Set target quality parameters
                if embedding == 0:
                    target_psnr = 40.0
                    target_bpp = 0.9
                else:
                    target_psnr = max(28.0, previous_psnr - 1)
                    target_bpp = max(0.5, (previous_payload / total_pixels) * 0.95)

                print(f"Target PSNR for embedding {embedding}: {target_psnr:.2f}")
                print(f"Target BPP for embedding {embedding}: {target_bpp:.4f}")

                # Initialize stage information including all possible block sizes
                all_block_sizes = [1024, 512, 256, 128, 64, 32, 16]
                stage_info = {
                    'embedding': embedding,
                    'block_info': {str(size): {'blocks': []} for size in all_block_sizes},
                    'payload': 0,
                    'psnr': 0,
                    'ssim': 0,
                    'hist_corr': 0,
                    'bpp': 0,
                    'rotation_mode': rotation_mode,
                    'prediction_method': method_name,
                    'use_different_weights': use_different_weights,
                    'block_size_weights': {},  # Store unified weights for each block size
                    'remaining_target': remaining_target,  # Track remaining target
                    'imgName': imgName  # Store image name for debugging
                }
                
                # 【新增】: 保存原始目標用於優先級計算
                if target_payload_size > 0:
                    stage_info['original_target'] = target_payload_size
                    stage_info['remaining_target'] = remaining_target
            
                # Rotation mode setup
                if rotation_mode == 'random':
                    # Generate random rotation angle for each block size
                    block_rotations = {
                        size: np.random.choice([-270, -180, -90, 0, 90, 180, 270])
                        for size in all_block_sizes
                    }
                    stage_info['block_rotations'] = block_rotations
                    print("\nBlock rotation angles for this stage:")
                    for size, angle in sorted(block_rotations.items(), reverse=True):
                        print(f"  {size}x{size}: {angle}°")

                # Calculate blocks to process
                num_blocks_horizontal = width // max_block_size
                num_blocks_vertical = height // max_block_size
                print(f"Processing {num_blocks_horizontal}x{num_blocks_vertical} blocks of size {max_block_size}x{max_block_size}")
                
                # Initialize output image
                stage_img = cp.zeros_like(current_img)
                if rotation_mode == 'random':
                    rotated_stage_img = cp.zeros_like(current_img)
                
                # Process image block by block
                processed_blocks = []
                remaining_at_start = remaining_target  # Store initial remaining target
                
                for i in range(num_blocks_vertical):
                    for j in range(num_blocks_horizontal):
                        # Extract current block position
                        y_start = i * max_block_size
                        x_start = j * max_block_size
                        
                        # Get current remaining target
                        if remaining_target is not None:
                            current_remaining = stage_info.get('remaining_target', remaining_target)
                            # Check if we've already reached the target
                            embed_data = current_remaining > 0
                        else:
                            current_remaining = None
                            embed_data = True
                            
                        # Extract current block
                        current_block = current_img[y_start:y_start+max_block_size, x_start:x_start+max_block_size]
                        
                        # Process current block with improved logic for early stopping
                        try:
                            block_results = process_block(
                                current_block, (y_start, x_start), max_block_size, stage_info, embedding,
                                variance_threshold, ratio_of_ones, target_bpp, target_psnr, el_mode,
                                verbose=False, 
                                rotation_mode=rotation_mode,
                                prediction_method=prediction_method,
                                remaining_target=current_remaining,
                                max_block_size=max_block_size,
                                embed_data=embed_data  # Pass embedding control flag
                            )
                            
                            processed_blocks.extend(block_results)
                        except Exception as e:
                            print(f"Error processing block at ({y_start}, {x_start}): {str(e)}")
                            # Return original block on error
                            processed_blocks.append((current_block, (y_start, x_start), max_block_size))
                        
                        # Periodically clean memory
                        if (i * num_blocks_horizontal + j + 1) % 4 == 0:
                            mem_pool.free_all_blocks()

                # Standard image reconstruction
                for block, pos, size in processed_blocks:
                    y, x = pos
                    
                    if rotation_mode == 'random':
                        # Save block in rotated state for visualization
                        if isinstance(block, cp.ndarray):
                            rotated_stage_img[y:y+size, x:x+size] = block
                        else:
                            rotated_stage_img[y:y+size, x:x+size] = cp.asarray(block)
                        
                        # If using rotation, need to rotate back
                        if size in stage_info['block_rotations']:
                            rotation = stage_info['block_rotations'][size]
                            if rotation != 0:
                                # Calculate inverse rotation
                                k = (-rotation // 90) % 4
                                # Check if block is CuPy array
                                if isinstance(block, cp.ndarray):
                                    block = cp.rot90(block, k=k)
                                else:
                                    block = np.rot90(block, k=k)
                                    block = cp.asarray(block)
                    
                    # Place block in the final image
                    if isinstance(block, cp.ndarray):
                        stage_img[y:y+size, x:x+size] = block
                    else:
                        stage_img[y:y+size, x:x+size] = cp.asarray(block)

                # Save rotated block visualization if using rotation mode
                if rotation_mode == 'random':
                    # Store rotated stage image in stage info
                    stage_info['rotated_stage_img'] = rotated_stage_img
                    
                    # Create visualization of rotated block structure
                    rotated_block_visualization = np.zeros_like(cp.asnumpy(original_img))
                    
                    # Define colors for different block sizes
                    block_colors = {
                        1024: 200,  # Light gray
                        512: 180,   # Slightly darker gray
                        256: 160,   # Medium gray
                        128: 140,   # Dark gray
                        64: 120,    # Darker gray
                        32: 100,    # Very dark gray
                        16: 80      # Almost black
                    }
                    
                    # Create visualization - draw blocks with rotation content
                    for size_str in stage_info['block_info']:
                        size = int(size_str)
                        blocks = stage_info['block_info'][size_str]['blocks']
                        
                        # Skip empty block sizes
                        if not blocks:
                            continue
                            
                        for block_info in blocks:
                            y, x = block_info['position']
                            
                            # Create border width based on block size
                            border_width = max(1, size // 64)
                            
                            # Fill block interior with original content
                            block_area = rotated_stage_img[y:y+size, x:x+size]
                            if isinstance(block_area, cp.ndarray):
                                block_area = cp.asnumpy(block_area)
                                
                            rotated_block_visualization[y:y+size, x:x+size] = block_area
                            
                            # Draw border around block
                            rotated_block_visualization[y:y+border_width, x:x+size] = block_colors[size]  # Top border
                            rotated_block_visualization[y+size-border_width:y+size, x:x+size] = block_colors[size]  # Bottom border
                            rotated_block_visualization[y:y+size, x:x+border_width] = block_colors[size]  # Left border
                            rotated_block_visualization[y:y+size, x+size-border_width:x+size] = block_colors[size]  # Right border
                    
                    # Save visualization
                    rotated_viz_path = f"{image_dir}/rotated_blocks/stage_{embedding}_blocks.png"
                    ensure_dir(rotated_viz_path)
                    cv2.imwrite(rotated_viz_path, rotated_block_visualization)
                    
                    # Add to stage info
                    stage_info['rotated_block_visualization'] = rotated_block_visualization
                    
                    print(f"Saved rotated block visualization to {rotated_viz_path}")
                    
                    # Create color-coded visualization for rotation angles
                    rotation_colors = {
                        0: [200, 200, 200],      # Gray for no rotation
                        90: [200, 100, 100],     # Reddish for 90°
                        180: [100, 200, 100],    # Greenish for 180°
                        270: [100, 100, 200],    # Bluish for 270°
                        -90: [200, 200, 100],    # Yellowish for -90°
                        -180: [200, 100, 200],   # Purplish for -180°
                        -270: [100, 200, 200]    # Cyan-ish for -270°
                    }
                    
                    # Create RGB visualization
                    rotated_block_visualization_color = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
                    
                    # Fill image with original content (darkened)
                    gray_img = cp.asnumpy(original_img)
                    for i in range(3):
                        rotated_block_visualization_color[:,:,i] = gray_img // 2  # Darken original image to make borders more visible
                    
                    # Draw borders with rotation color coding
                    for size_str in stage_info['block_info']:
                        size = int(size_str)
                        blocks = stage_info['block_info'][size_str]['blocks']
                        
                        # Skip empty block sizes
                        if not blocks:
                            continue
                            
                        for block_info in blocks:
                            y, x = block_info['position']
                            
                            # Get rotation angle for this block size
                            rotation = stage_info['block_rotations'][size]
                            color = rotation_colors.get(rotation, [150, 150, 150])  # Default gray if rotation not found
                            
                            # Border width proportional to block size
                            border_width = max(1, size // 64)
                            
                            # Draw colored border
                            rotated_block_visualization_color[y:y+border_width, x:x+size, :] = color  # Top
                            rotated_block_visualization_color[y+size-border_width:y+size, x:x+size, :] = color  # Bottom
                            rotated_block_visualization_color[y:y+size, x:x+border_width, :] = color  # Left
                            rotated_block_visualization_color[y:y+size, x+size-border_width:x+size, :] = color  # Right
                    
                    # Save colored visualization
                    color_viz_path = f"{image_dir}/rotated_blocks/stage_{embedding}_color.png"
                    ensure_dir(color_viz_path)
                    cv2.imwrite(color_viz_path, rotated_block_visualization_color)
                    
                    # Also save to debug directory
                    cv2.imwrite(f"{debug_dir}/stage_{embedding}_color.png", rotated_block_visualization_color)
                    
                    # Add rotation angle legend
                    legend_img = np.ones((200, 400, 3), dtype=np.uint8) * 255  # White background
                    legend_title = "Rotation Angles Legend"
                    cv2.putText(legend_img, legend_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    
                    # Add color samples for each rotation angle
                    y_offset = 60
                    for angle, color in rotation_colors.items():
                        # Draw color square
                        cv2.rectangle(legend_img, (10, y_offset), (40, y_offset+20), color, -1)
                        # Add text label
                        cv2.putText(legend_img, f"{angle}°", (50, y_offset+15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                        y_offset += 30
                    
                    # Save legend
                    legend_path = f"{image_dir}/rotated_blocks/legend.png"
                    ensure_dir(legend_path)
                    cv2.imwrite(legend_path, legend_img)
                    
                    # Add to stage info
                    stage_info['rotated_block_visualization_color'] = rotated_block_visualization_color
                    
                    print(f"Saved colored rotated block visualization to {color_viz_path}")
                
                # Store stage image in stage info
                stage_info['stage_img'] = stage_img
                
                # Save stage image for debug
                cv2.imwrite(f"{debug_dir}/stage_{embedding}_result.png", cp.asnumpy(stage_img))

                # Calculate metrics
                stage_img_np = cp.asnumpy(stage_img)
                original_img_np = cp.asnumpy(original_img)
                
                # Check for shape mismatch
                if stage_img_np.shape != original_img_np.shape:
                    print(f"WARNING: Shape mismatch - stage img: {stage_img_np.shape}, original img: {original_img_np.shape}")
                
                # Calculate metrics using library functions
                lib_psnr = calculate_psnr(original_img_np, stage_img_np)
                lib_ssim = calculate_ssim(original_img_np, stage_img_np)
                lib_hist_corr = histogram_correlation(
                    np.histogram(original_img_np, bins=256, range=(0, 255))[0],
                    np.histogram(stage_img_np, bins=256, range=(0, 255))[0]
                )
                
                # Direct calculation for verification
                mse = np.mean((original_img_np.astype(np.float64) - stage_img_np.astype(np.float64)) ** 2)
                if mse == 0:
                    direct_psnr = float('inf')
                else:
                    direct_psnr = 10 * np.log10((255.0 ** 2) / mse)
                
                # Use the most reliable PSNR value
                if direct_psnr > 30 and direct_psnr > lib_psnr:
                    psnr = direct_psnr
                    print(f"Using direct PSNR calculation: {direct_psnr:.2f} dB (library: {lib_psnr:.2f} dB)")
                else:
                    psnr = lib_psnr
                
                ssim = lib_ssim
                hist_corr = lib_hist_corr

                # Update stage quality metrics
                stage_info['psnr'] = float(psnr)
                stage_info['ssim'] = float(ssim)
                stage_info['hist_corr'] = float(hist_corr)
                stage_info['bpp'] = float(stage_info['payload'] / total_pixels)
                
                # Add stage to overall list
                pee_stages.append(stage_info)
                total_payload += stage_info['payload']
                previous_psnr = stage_info['psnr']
                previous_ssim = stage_info['ssim']
                previous_payload = stage_info['payload']

                # Output stage summary
                print(f"\nEmbedding {embedding} summary:")
                print(f"Prediction Method: {method_name}")
                print(f"Payload: {stage_info['payload']}")
                print(f"BPP: {stage_info['bpp']:.4f}")
                print(f"PSNR: {stage_info['psnr']:.2f}")
                print(f"SSIM: {stage_info['ssim']:.4f}")
                print(f"Hist Corr: {stage_info['hist_corr']:.4f}")
                
                # Output block size distribution
                print("\nBlock size distribution:")
                for size_str in sorted(stage_info['block_info'].keys(), key=int, reverse=True):
                    block_count = len(stage_info['block_info'][size_str]['blocks'])
                    if block_count > 0:
                        if rotation_mode == 'random' and 'block_rotations' in stage_info:
                            rotation = stage_info['block_rotations'][int(size_str)]
                            print(f"  {size_str}x{size_str}: {block_count} blocks, Rotation: {rotation}°")
                        else:
                            print(f"  {size_str}x{size_str}: {block_count} blocks")

                # Prepare for next stage
                current_img = stage_img.copy()

                # Clean up memory
                mem_pool.free_all_blocks()
                print(f"Memory usage after embedding {embedding}: {mem_pool.used_bytes()/1024/1024:.2f} MB")

                # Check if target reached
                if remaining_target is not None and remaining_target <= 0:
                    print(f"Reached target payload ({target_payload_size} bits) at stage {embedding}")
                    break

            # Crop image back to original size if it was padded
            final_pee_img = cp.asnumpy(current_img)
            if (orig_height, orig_width) != (height, width):
                final_pee_img = final_pee_img[:orig_height, :orig_width]
            
            # 【新增】: Save max payload info for future runs if this was a full capacity run
            if target_payload_size <= 0:
                # Create data directory
                data_dir = f"{output_dir}/data/{imgName}"
                os.makedirs(data_dir, exist_ok=True)
                
                # Store max payload information
                max_payload_data = {
                    'max_payload': total_payload,
                    'method': 'quadtree',
                    'prediction_method': method_name,
                    'min_block_size': original_min_block_size,
                    'variance_threshold': original_variance_threshold,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                
                # Save to npy file
                max_payload_cache_file = f"{data_dir}/quadtree_{method_name}_max_payload.npy"
                np.save(max_payload_cache_file, max_payload_data)
                print(f"Saved maximum payload information: {total_payload} bits for future use")
            
            # Return final result
            return final_pee_img, int(total_payload), pee_stages

        except Exception as e:
            print(f"Error in embedding process: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    except Exception as e:
        print(f"Error in quadtree processing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        # Ensure all memory is cleaned up
        cleanup_quadtree_resources()
        
def pee_process_color_image_quadtree_cuda(img, total_embeddings, ratio_of_ones, use_different_weights, 
                                       min_block_size, variance_threshold, el_mode, 
                                       rotation_mode='random',
                                       prediction_method=None,
                                       target_payload_size=-1,
                                       max_block_size=None,
                                       imgName=None,
                                       output_dir=None):
    """
    Process a color image using quadtree PEE method.
    
    This function splits a color image into its RGB channels, processes each channel
    independently using the existing quadtree PEE method, and then recombines the
    channels into a final color image.
    
    Parameters:
    -----------
    Same as pee_process_with_quadtree_cuda, but img is now a color image
        
    Returns:
    --------
    tuple
        (final_color_img, total_payload, color_pee_stages)
    """
    import os
    import cv2
    import numpy as np
    import cupy as cp
    from color import split_color_channels, combine_color_channels
    from common import cleanup_memory
    
    if prediction_method is None:
        # Import PredictionMethod if not provided to maintain compatibility
        from image_processing import PredictionMethod
        prediction_method = PredictionMethod.PROPOSED
    
    # Split color image into channels
    b_channel, g_channel, r_channel = split_color_channels(img)
    
    # Track total payload across all channels
    total_payload = 0
    
    # Create directory structure for channel outputs if imgName is provided
    if imgName and output_dir:
        channels_dir = f"{output_dir}/image/{imgName}/quadtree/channels"
        os.makedirs(channels_dir, exist_ok=True)
    
    color_pee_stages = []
    
    # Process each channel separately
    print("\nProcessing blue channel...")
    final_b_img, b_payload, b_stages = pee_process_with_quadtree_cuda(
        b_channel, total_embeddings, ratio_of_ones, use_different_weights,
        min_block_size, variance_threshold, el_mode, rotation_mode,
        prediction_method=prediction_method,
        target_payload_size=target_payload_size // 3 if target_payload_size > 0 else -1,
        max_block_size=max_block_size,
        imgName=f"{imgName}_blue" if imgName else None,
        output_dir=output_dir
    )
    total_payload += b_payload
    
    # Save blue channel output if imgName is provided
    if imgName and output_dir:
        cv2.imwrite(f"{channels_dir}/{imgName}_blue_final.png", final_b_img)
    
    # Clean GPU memory between channel processing
    cleanup_memory()
    
    print("\nProcessing green channel...")
    final_g_img, g_payload, g_stages = pee_process_with_quadtree_cuda(
        g_channel, total_embeddings, ratio_of_ones, use_different_weights,
        min_block_size, variance_threshold, el_mode, rotation_mode,
        prediction_method=prediction_method,
        target_payload_size=target_payload_size // 3 if target_payload_size > 0 else -1,
        max_block_size=max_block_size,
        imgName=f"{imgName}_green" if imgName else None,
        output_dir=output_dir
    )
    total_payload += g_payload
    
    # Save green channel output if imgName is provided
    if imgName and output_dir:
        cv2.imwrite(f"{channels_dir}/{imgName}_green_final.png", final_g_img)
    
    # Clean GPU memory between channel processing
    cleanup_memory()
    
    print("\nProcessing red channel...")
    final_r_img, r_payload, r_stages = pee_process_with_quadtree_cuda(
        r_channel, total_embeddings, ratio_of_ones, use_different_weights,
        min_block_size, variance_threshold, el_mode, rotation_mode,
        prediction_method=prediction_method,
        target_payload_size=target_payload_size // 3 if target_payload_size > 0 else -1,
        max_block_size=max_block_size,
        imgName=f"{imgName}_red" if imgName else None,
        output_dir=output_dir
    )
    total_payload += r_payload
    
    # Save red channel output if imgName is provided
    if imgName and output_dir:
        cv2.imwrite(f"{channels_dir}/{imgName}_red_final.png", final_r_img)
    
    # Combine channels back into a color image
    final_color_img = combine_color_channels(final_b_img, final_g_img, final_r_img)
    
    # Create combined stages information for all 3 channels
    for i in range(min(len(b_stages), len(g_stages), len(r_stages))):
        # Get stage info from each channel
        b_stage = b_stages[i]
        g_stage = g_stages[i]
        r_stage = r_stages[i]
        
        # Initialize combined stage info
        combined_stage = {
            'embedding': b_stage['embedding'],  # All should be the same
            'payload': b_stage['payload'] + g_stage['payload'] + r_stage['payload'],
            'channel_payloads': {
                'blue': b_stage['payload'],
                'green': g_stage['payload'],
                'red': r_stage['payload']
            },
            'bpp': (b_stage['bpp'] + g_stage['bpp'] + r_stage['bpp']) / 3,  # Average BPP
            'channel_metrics': {
                'blue': {'psnr': b_stage['psnr'], 'ssim': b_stage['ssim'], 'hist_corr': b_stage['hist_corr']},
                'green': {'psnr': g_stage['psnr'], 'ssim': g_stage['ssim'], 'hist_corr': g_stage['hist_corr']},
                'red': {'psnr': r_stage['psnr'], 'ssim': r_stage['ssim'], 'hist_corr': r_stage['hist_corr']}
            }
        }
        
        # Combine stage images if available
        if 'stage_img' in b_stage and 'stage_img' in g_stage and 'stage_img' in r_stage:
            b_stage_img = cp.asnumpy(b_stage['stage_img']) if isinstance(b_stage['stage_img'], cp.ndarray) else b_stage['stage_img']
            g_stage_img = cp.asnumpy(g_stage['stage_img']) if isinstance(g_stage['stage_img'], cp.ndarray) else g_stage['stage_img']
            r_stage_img = cp.asnumpy(r_stage['stage_img']) if isinstance(r_stage['stage_img'], cp.ndarray) else r_stage['stage_img']
            
            combined_stage['stage_img'] = combine_color_channels(b_stage_img, g_stage_img, r_stage_img)
        
        # Calculate combined metrics directly from the combined image
        if 'stage_img' in combined_stage:
            # For simplicity, we'll use the channel averages
            psnr = (b_stage['psnr'] + g_stage['psnr'] + r_stage['psnr']) / 3
            ssim = (b_stage['ssim'] + g_stage['ssim'] + r_stage['ssim']) / 3
            hist_corr = (b_stage['hist_corr'] + g_stage['hist_corr'] + r_stage['hist_corr']) / 3
            
            combined_stage['psnr'] = psnr
            combined_stage['ssim'] = ssim
            combined_stage['hist_corr'] = hist_corr
            
        # Save combined block info - store each channel's block_info separately
        combined_stage['block_info'] = {
            'blue': b_stage['block_info'],
            'green': g_stage['block_info'],
            'red': r_stage['block_info']
        }
            
        # Add stage to combined stages
        color_pee_stages.append(combined_stage)
        
        # Save stage image if imgName is provided
        if imgName and output_dir and 'stage_img' in combined_stage:
            stage_dir = f"{output_dir}/image/{imgName}/quadtree"
            os.makedirs(stage_dir, exist_ok=True)
            cv2.imwrite(f"{stage_dir}/color_stage_{i}_result.png", combined_stage['stage_img'])
    
    return final_color_img, total_payload, color_pee_stages