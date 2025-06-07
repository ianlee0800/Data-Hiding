import numpy as np
import cupy as cp
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
    save_image,
    predict_image_cuda
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
    處理當前區塊的 PEE 嵌入，支援多種預測方法和統一權重
    
    Parameters:
    -----------
    block : numpy.ndarray or cupy.ndarray
        輸入區塊
    position : tuple
        區塊在原圖中的位置 (y, x)
    size : int
        區塊大小
    stage_info : dict
        當前階段的資訊
    embedding : int
        當前嵌入階段
    ratio_of_ones : float
        嵌入數據中 1 的比例
    target_bpp : float
        目標 BPP
    target_psnr : float
        目標 PSNR
    el_mode : int
        EL模式
    prediction_method : PredictionMethod
        預測方法
    remaining_target : list or None
        剩餘需要嵌入的數據量的可變容器 [target_value]
    verbose : bool
        是否輸出詳細資訊
        
    Returns:
    --------
    list
        [embedded_block, position, size, block_was_rotated]
    """
    try:
        # 確保 block 是 CuPy 數組
        if not isinstance(block, cp.ndarray):
            block = cp.asarray(block)
        
        # 檢查是否已達到目標容量
        if remaining_target is not None and remaining_target[0] <= 0:
            # 已達到目標，直接返回原圖
            if verbose:
                print(f"  Target reached. Skipping block at {position} (size: {size}x{size})")
            return [(block, position, size, False)]
        
        # 計算區塊大小和目標容量
        block_size = block.size
        current_target = None
        if remaining_target is not None:
            # 如果剩餘目標小於區塊大小，可能不應該跳過這個區塊
            # 而是嘗試精確嵌入所需的位元數
            current_target = min(block_size, remaining_target[0])
            if verbose:
                print(f"  Block at {position} allocated {current_target} bits (of {remaining_target[0]} remaining)")
        
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
        # 如果是接近目標值的最後一個區塊，優先考慮生成恰好數量的數據
        if remaining_target is not None and remaining_target[0] <= block_size:
            # 生成剛好所需數量的數據
            data_size = remaining_target[0]
            if verbose:
                print(f"  Generating exactly {data_size} bits to match target")
        else:
            data_size = block_size
            
        data_to_embed = generate_random_binary_array(data_size, ratio_of_ones)
        data_to_embed = cp.asarray(data_to_embed, dtype=cp.uint8)
        
        # 標記是否區塊有旋轉
        block_was_rotated = False
        original_block = block.copy()  # 儲存原始區塊用於計算指標
        
        # 應用旋轉 (如果 rotation_mode 為 'random')
        rotation = 0
        if 'rotation_mode' in stage_info and stage_info['rotation_mode'] == 'random' and 'block_rotations' in stage_info:
            if size in stage_info['block_rotations']:
                rotation = stage_info['block_rotations'][size]
                if rotation != 0:
                    k = rotation // 90
                    block = cp.rot90(block, k=k)
                    block_was_rotated = True
                    if verbose:
                        print(f"  Applied rotation of {rotation}° to block at {position}")
        
        # 根據預測方法進行不同的處理
        if prediction_method == PredictionMethod.PROPOSED:
            # 檢查是否使用不同權重
            if 'use_different_weights' in stage_info and stage_info['use_different_weights']:
                # PROPOSED 方法需要計算權重 (每個區塊使用不同權重)
                weights, (sub_payload, sub_psnr) = brute_force_weight_search_cuda(
                    block, data_to_embed, local_el, local_target_bpp, local_target_psnr, 
                    embedding, block_size=size
                )
                if verbose:
                    print(f"  Computed unique weights for block at {position}: {weights}")
            else:
                # 使用同樣大小區塊的統一權重
                size_str = str(size)
                if size_str in stage_info['block_size_weights']:
                    # 使用已計算的權重
                    weights = np.array(stage_info['block_size_weights'][size_str], dtype=np.int32)
                    if verbose:
                        print(f"  Using cached weights for {size}x{size} blocks: {weights}")
                else:
                    # 第一次計算此大小的權重
                    weights, (sub_payload, sub_psnr) = brute_force_weight_search_cuda(
                        block, data_to_embed, local_el, local_target_bpp, local_target_psnr, 
                        embedding, block_size=size
                    )
                    # 儲存權重以便後續使用
                    if hasattr(weights, 'tolist'):
                        stage_info['block_size_weights'][size_str] = weights.tolist()
                    else:
                        stage_info['block_size_weights'][size_str] = weights
                    if verbose:
                        print(f"  Computed new weights for {size}x{size} blocks: {weights}")
        else:
            # MED 和 GAP 方法不需要權重
            weights = None
        
        # 計算預測圖像 (使用安全的類型檢查)
        pred_image = predict_image_cuda(block, prediction_method, weights)
        
        # 安全地獲取 NumPy 格式的預測圖像
        if hasattr(pred_image, 'copy_to_host'):
            pred_image_np = pred_image.copy_to_host()
        else:
            pred_image_np = pred_image
            
        # 執行數據嵌入
        embedded_block, payload, pred_block = multi_pass_embedding(
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
        
        # 如果區塊被旋轉過，計算指標時使用未旋轉的原始區塊做比較
        compare_block = original_block
        
        # 將旋轉後的嵌入區塊旋轉回原始方向
        if block_was_rotated:
            # 計算逆旋轉角度
            k = (-rotation // 90) % 4
            embedded_block = cp.rot90(embedded_block, k=k)
            
            # 將預測圖像也旋轉回來
            pred_image_np = np.rot90(pred_image_np, k=k)
            
            if verbose:
                print(f"  Rotated embedded block back by {-rotation}° to original orientation")
        
        # 計算並記錄區塊資訊
        block_info = {
            'position': position,
            'size': size,
            'weights': (weights.tolist() if weights is not None and hasattr(weights, 'tolist') 
                       else "N/A" if prediction_method in [PredictionMethod.MED, PredictionMethod.GAP] 
                       else None),
            'payload': int(payload),
            'psnr': float(calculate_psnr(cp.asnumpy(compare_block), cp.asnumpy(embedded_block))),
            'ssim': float(calculate_ssim(cp.asnumpy(compare_block), cp.asnumpy(embedded_block))),
            'hist_corr': float(histogram_correlation(
                np.histogram(cp.asnumpy(compare_block), bins=256, range=(0, 255))[0],
                np.histogram(cp.asnumpy(embedded_block), bins=256, range=(0, 255))[0]
            )),
            'EL': int(to_numpy(local_el).max()),
            'prediction_method': prediction_method.value,
            'rotation': rotation,
            'original_img': cp.asnumpy(original_block),  # 新增
            'pred_img': pred_image_np,  # 新增
            'embedded_img': cp.asnumpy(embedded_block)  # 新增
        }
        
        # 更新階段資訊
        stage_info['block_info'][str(size)]['blocks'].append(block_info)
        stage_info['payload'] += payload
        
        if verbose:
            print(f"  Block processed at size {size}x{size}")
            print(f"  Prediction method: {prediction_method.value}")
            print(f"  Payload: {payload}")
            print(f"  PSNR: {block_info['psnr']:.2f}")
            if remaining_target is not None:
                print(f"  Remaining target: {remaining_target[0]} bits")
        
        # 返回處理後的區塊，並標記其是否曾經被旋轉過
        return [(embedded_block, position, size, block_was_rotated)]
            
    except Exception as e:
        print(f"Error in block processing: {str(e)}")
        raise

def process_block(block, position, size, stage_info, embedding, variance_threshold, 
                 ratio_of_ones, target_bpp, target_psnr, el_mode, verbose=False,
                 rotation_mode='none', prediction_method=PredictionMethod.PROPOSED,
                 remaining_target=None, max_block_size=1024):
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
    remaining_target : list or None
        剩餘需要嵌入的數據量的可變容器 [target_value]
    max_block_size : int
        最大區塊大小，默認為1024
    """
    try:
        # 精確控制 - 如果剩餘容量很小，優先處理小區塊而不分割大區塊
        if remaining_target is not None:
            # 如果剩餘容量已經不足：
            if remaining_target[0] <= 0:
                # 如果已達到目標容量，直接返回原始區塊，不進行嵌入
                if verbose:
                    print(f"Target reached. Skipping block at {position} (size: {size}x{size})")
                return [(block, position, size, False)]
            
            # 如果剩餘容量很小，小於區塊大小的20%，且區塊比較大，考慮直接處理
            if remaining_target[0] < (size * size * 0.2) and size >= 64:
                if verbose:
                    print(f"Small remaining target ({remaining_target[0]} bits) - processing block directly")
                return process_current_block(
                    block, position, size, stage_info, embedding,
                    ratio_of_ones, target_bpp, target_psnr, el_mode,
                    prediction_method=prediction_method,
                    remaining_target=remaining_target,
                    verbose=verbose
                )
        
        if size < 16:  # 最小區塊大小限制
            return []
        
        # 確保 block 是 CuPy 數組
        if not isinstance(block, cp.ndarray):
            block = cp.asarray(block)
        
        # 計算區塊變異度
        variance = calculate_block_variance_cuda(block)
        
        # 根據區塊大小調整閾值
        adjusted_threshold = variance_threshold
        if size >= 512:
            adjusted_threshold *= 1.3  # 增加對1024塊的處理
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
            if remaining_target is not None:
                print(f"  Remaining target: {remaining_target[0]} bits")
        
        # 根據變異度決定是否分割
        if size > 16 and variance > adjusted_threshold:
            # 繼續分割為四個子區塊
            half_size = size // 2
            sub_blocks = []
            
            # 剩餘容量精確控制 - 為子區塊分配合理的目標容量
            sub_block_targets = None
            if remaining_target is not None and remaining_target[0] > 0:
                # 根據區塊大小分配容量
                sub_block_size = half_size * half_size
                total_size = size * size
                sub_block_targets = [
                    [int(remaining_target[0] * sub_block_size / total_size)],
                    [int(remaining_target[0] * sub_block_size / total_size)],
                    [int(remaining_target[0] * sub_block_size / total_size)],
                    [int(remaining_target[0] * sub_block_size / total_size)]
                ]
                # 確保分配的總和不超過剩餘目標
                total_allocated = sum(target[0] for target in sub_block_targets)
                if total_allocated > remaining_target[0]:
                    # 調整最後一個區塊的分配
                    sub_block_targets[3][0] -= (total_allocated - remaining_target[0])
                
                if verbose:
                    print(f"  Allocated targets for sub-blocks: {[target[0] for target in sub_block_targets]}")
            
            for i in range(2):
                for j in range(2):
                    # 獲取當前子區塊的索引和目標容量
                    sub_idx = i * 2 + j
                    current_target = sub_block_targets[sub_idx] if sub_block_targets else remaining_target
                    
                    # 如果子區塊的目標容量已經為0，跳過處理
                    if current_target is not None and current_target[0] <= 0:
                        # 直接添加未處理的子區塊
                        y_start = position[0] + i * half_size
                        x_start = position[1] + j * half_size
                        sub_block = block[i*half_size:(i+1)*half_size, j*half_size:(j+1)*half_size]
                        sub_blocks.append((sub_block, (y_start, x_start), half_size, False))
                        continue
                        
                    y_start = position[0] + i * half_size
                    x_start = position[1] + j * half_size
                    sub_block = block[i*half_size:(i+1)*half_size, j*half_size:(j+1)*half_size]
                    
                    # 遞迴處理子區塊，傳遞所有必要參數，使用分配的目標容量
                    sub_results = process_block(
                        sub_block, (y_start, x_start), half_size,
                        stage_info, embedding, variance_threshold,
                        ratio_of_ones, target_bpp, target_psnr, el_mode,
                        verbose=verbose,
                        rotation_mode=rotation_mode,
                        prediction_method=prediction_method,
                        remaining_target=current_target,  # 使用分配給這個子區塊的目標
                        max_block_size=max_block_size
                    )
                    
                    # 更新主要的剩餘目標容量
                    if remaining_target is not None and current_target is not None:
                        # 計算實際使用的容量（分配前減去分配後）
                        used_capacity = sub_block_targets[sub_idx][0] - current_target[0]
                        remaining_target[0] -= used_capacity
                        if verbose and used_capacity > 0:
                            print(f"  Sub-block {sub_idx} used {used_capacity} bits, main remaining: {remaining_target[0]}")
                    
                    sub_blocks.extend(sub_results)
            
            return sub_blocks
        else:
            # 處理當前區塊
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
                                 rotation_mode='none',
                                 prediction_method=None,
                                 target_payload_size=-1,
                                 max_block_size=None,
                                 imgName=None,
                                 output_dir=None):
    """
    使用Quad tree的PEE處理函數，支援多種預測方法和payload控制
    
    Parameters:
    -----------
    img : numpy.ndarray
        輸入圖像 (灰階或彩色)
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
    max_block_size : int, optional
        最大區塊大小，預設為圖像大小的一半或512，取較大值
    imgName : str, optional
        圖像名稱，用於儲存視覺化結果
    output_dir : str, optional
        輸出目錄，用於儲存視覺化結果
        
    Returns:
    --------
    tuple
        (final_pee_img, total_payload, pee_stages)
        final_pee_img: 最終處理後的圖像
        total_payload: 總嵌入容量
        pee_stages: 包含每個階段詳細資訊的列表
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
        # 導入必要的模組
        import os
        import cv2
        
        # 定義 ensure_dir 函數
        def ensure_dir(file_path):
            """確保目錄存在，如果不存在則創建"""
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # 參數驗證與預設值設置
        # 檢查並設置圖像名稱
        if imgName is None:
            if rotation_mode == 'random':
                # 使用隨機旋轉模式時需要圖像名稱
                imgName = "unknown_image"  # 使用預設名稱
                print("Warning: No image name provided. Using 'unknown_image' for saving visualizations.")
            else:
                # 非旋轉模式可以不需要圖像名稱
                imgName = "temp"
        
        # 檢查並設置輸出目錄
        if output_dir is None:
            output_dir = "./Prediction_Error_Embedding/outcome"
            print(f"Using default output directory: {output_dir}")
        
        # 預設視覺化路徑
        image_dir = f"{output_dir}/image/{imgName}/quadtree"
        
        # 確保輸出目錄存在
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(f"{image_dir}/rotated_blocks", exist_ok=True)
        
        # 參數合法性檢查
        if min_block_size < 16:
            raise ValueError("min_block_size must be at least 16")
        if min_block_size & (min_block_size - 1) != 0:
            raise ValueError("min_block_size must be a power of 2")
        if not 0 <= ratio_of_ones <= 1:
            raise ValueError("ratio_of_ones must be between 0 and 1")
        if rotation_mode not in ['none', 'random']:
            raise ValueError("rotation_mode must be 'none' or 'random'")
        
        # 確定最大區塊大小
        height, width = img.shape
        if max_block_size is None:
            max_block_size = max(512, min(1024, max(height, width)))
        
        # 檢查圖像大小是否為 max_block_size 的整數倍
        if height % max_block_size != 0 or width % max_block_size != 0:
            # 墊充圖像到合適的大小
            new_height = ((height + max_block_size - 1) // max_block_size) * max_block_size
            new_width = ((width + max_block_size - 1) // max_block_size) * max_block_size
            
            # 建立新圖像並複製原始數據
            padded_img = np.zeros((new_height, new_width), dtype=np.uint8)
            padded_img[:height, :width] = img
            
            # 使用邊緣像素填充剩餘部分
            if height < new_height:
                padded_img[height:, :width] = padded_img[height-1:height, :width]
            if width < new_width:
                padded_img[:, width:] = padded_img[:, width-1:width]
            
            # 更新圖像和尺寸
            img = padded_img
            height, width = img.shape
            print(f"Image resized from {height}x{width} to {new_height}x{new_width} for quadtree processing")
        
        # 預測方法相關設置
        if prediction_method in [PredictionMethod.MED, PredictionMethod.GAP]:
            use_different_weights = False
            print(f"Note: Weight optimization disabled for {prediction_method.value} prediction method")
        
        # 處理旋轉模式設置
        # 警告：當使用精確的目標payload時，不建議使用random旋轉模式
        if rotation_mode == 'random' and target_payload_size > 0:
            print("WARNING: Using random rotation mode with a specific target payload may cause image quality issues.")
            print("For better image quality with target payload, consider using rotation_mode='none'.")
        
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
        
        # 使用可變容器來追蹤剩餘目標payload
        # 這是關鍵修改點：使用list作為可變容器而不是int
        if target_payload_size > 0:
            remaining_target = [target_payload_size]  # 使用list作為可變容器
            print(f"Target payload set: {target_payload_size} bits")
            # 目標容量的填充率（目標佔圖片容量的比例）
            fill_rate = target_payload_size / total_pixels
            print(f"Target fill rate: {fill_rate:.4f} bits per pixel")
        else:
            remaining_target = None
            print("Using maximum embedding capacity")
        
        # 儲存每種大小區塊的權重
        block_size_weights = {}
        
        # 設置精確控制的額外參數
        # 如果設置了目標payload，啟用精確控制模式
        precise_control = target_payload_size > 0
        
        # GPU 記憶體管理
        mem_pool = cp.get_default_memory_pool()

        try:
            # 逐階段處理
            for embedding in range(total_embeddings):
                # 檢查是否達到目標 payload
                if remaining_target is not None and remaining_target[0] <= 0:
                    print(f"Reached target payload ({target_payload_size} bits) at stage {embedding}")
                    break
                    
                # 輸出階段開始資訊
                print(f"\nStarting embedding {embedding}")
                print(f"Memory usage before embedding {embedding}: {mem_pool.used_bytes()/1024/1024:.2f} MB")
                if remaining_target is not None:
                    print(f"Remaining target payload: {remaining_target[0]} bits")

                # 設定目標品質參數
                if embedding == 0:
                    target_psnr = 40.0
                    target_bpp = 0.9
                else:
                    target_psnr = max(28.0, previous_psnr - 1)
                    target_bpp = max(0.5, (previous_payload / total_pixels) * 0.95)

                print(f"Target PSNR for embedding {embedding}: {target_psnr:.2f}")
                print(f"Target BPP for embedding {embedding}: {target_bpp:.4f}")

                # 初始化階段資訊，包括所有可能的區塊大小
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
                    'prediction_method': prediction_method.value,
                    'use_different_weights': use_different_weights,
                    'block_size_weights': {}  # 儲存每種大小區塊的統一權重
                }

                # 旋轉模式設置
                if rotation_mode == 'random':
                    # 為每個區塊大小生成隨機旋轉角度
                    block_rotations = {
                        size: np.random.choice([-270, -180, -90, 0, 90, 180, 270])
                        for size in all_block_sizes
                    }
                    # 如果使用精確容量控制，考慮減少旋轉
                    if precise_control:
                        # 減少旋轉以提高圖像質量
                        for size in [1024, 512, 256]:
                            block_rotations[size] = 0  # 大區塊不旋轉
                            
                    stage_info['block_rotations'] = block_rotations
                    print("\nBlock rotation angles for this stage:")
                    for size, angle in sorted(block_rotations.items(), reverse=True):
                        print(f"  {size}x{size}: {angle}°")

                # 計算要處理的塊數
                num_blocks_horizontal = width // max_block_size
                num_blocks_vertical = height // max_block_size
                print(f"Processing {num_blocks_horizontal}x{num_blocks_vertical} blocks of size {max_block_size}x{max_block_size}")
                
                # 初始化輸出圖像
                stage_img = cp.zeros_like(current_img)
                if rotation_mode == 'random':
                    rotated_stage_img = cp.zeros_like(current_img)
                    
                # 逐塊處理圖像
                processed_blocks = []
                for i in range(num_blocks_vertical):
                    for j in range(num_blocks_horizontal):
                        # 檢查是否已達到目標payload
                        if remaining_target is not None and remaining_target[0] <= 0:
                            # 如果已達到目標，不處理剩餘區塊直接複製
                            y_start = i * max_block_size
                            x_start = j * max_block_size
                            stage_img[y_start:y_start+max_block_size, x_start:x_start+max_block_size] = current_img[y_start:y_start+max_block_size, x_start:x_start+max_block_size]
                            continue
                            
                        # 提取當前塊
                        y_start = i * max_block_size
                        x_start = j * max_block_size
                        current_block = current_img[y_start:y_start+max_block_size, x_start:x_start+max_block_size]
                        
                        # 處理當前塊，傳遞可變容器
                        block_results = process_block(
                            current_block, (y_start, x_start), max_block_size, stage_info, embedding,
                            variance_threshold, ratio_of_ones, target_bpp, target_psnr, el_mode,
                            verbose=False, 
                            rotation_mode=rotation_mode,
                            prediction_method=prediction_method,
                            remaining_target=remaining_target,  # 傳遞可變容器
                            max_block_size=max_block_size
                        )
                        
                        processed_blocks.extend(block_results)
                        
                        # 定期清理記憶體
                        if (i * num_blocks_horizontal + j + 1) % 4 == 0:
                            mem_pool.free_all_blocks()

                # 重建圖像 - 關鍵修改：確保正確處理旋轉
                for block, pos, size, was_rotated in processed_blocks:
                    block = cp.asarray(block)
                    
                    # 將區塊放回最終圖像 - 直接使用已經正確旋轉的區塊
                    stage_img[pos[0]:pos[0]+size, pos[1]:pos[1]+size] = block
                    
                    # 僅在需要旋轉視覺化時保存旋轉狀態
                    if rotation_mode == 'random':
                        rotation = stage_info['block_rotations'].get(size, 0)
                        # 如果需要保存旋轉狀態，使用原始的旋轉版本
                        if rotation != 0 and not was_rotated:
                            # 區塊需要先旋轉，才能放入rotated_stage_img
                            k = rotation // 90
                            rotated_block = cp.rot90(block, k=k)
                            rotated_stage_img[pos[0]:pos[0]+size, pos[1]:pos[1]+size] = rotated_block
                        else:
                            # 使用已旋轉的區塊
                            rotated_stage_img[pos[0]:pos[0]+size, pos[1]:pos[1]+size] = block

                # 保存rotated_stage_img供視覺化使用
                if rotation_mode == 'random':
                    stage_info['rotated_stage_img'] = rotated_stage_img
                    
                    # 建立旋轉後的區塊視覺化
                    rotated_block_visualization = np.zeros_like(cp.asnumpy(original_img))
                    
                    # 定義每種區塊大小的顏色
                    block_colors = {
                        1024: 200,  # 淺灰色
                        512: 180,   # 稍深灰色
                        256: 160,   # 中灰色
                        128: 140,   # 深灰色
                        64: 120,    # 更深灰色
                        32: 100,    # 很深灰色
                        16: 80      # 近乎黑色
                    }
                    
                    # 創建可視化 - 繪製帶有旋轉內容的區塊
                    for size_str in stage_info['block_info']:
                        size = int(size_str)
                        blocks = stage_info['block_info'][size_str]['blocks']
                        
                        # 跳過空區塊大小
                        if not blocks:
                            continue
                            
                        for block_info in blocks:
                            y, x = block_info['position']
                            
                            # 根據區塊大小創建邊框寬度
                            border_width = max(1, size // 64)
                            
                            # 填充區塊內部為原始內容
                            block_area = rotated_stage_img[y:y+size, x:x+size]
                            if isinstance(block_area, cp.ndarray):
                                block_area = cp.asnumpy(block_area)
                                
                            rotated_block_visualization[y:y+size, x:x+size] = block_area
                            
                            # 在區塊周圍繪製邊框
                            rotated_block_visualization[y:y+border_width, x:x+size] = block_colors[size]  # 上邊框
                            rotated_block_visualization[y+size-border_width:y+size, x:x+size] = block_colors[size]  # 下邊框
                            rotated_block_visualization[y:y+size, x:x+border_width] = block_colors[size]  # 左邊框
                            rotated_block_visualization[y:y+size, x+size-border_width:x+size] = block_colors[size]  # 右邊框
                    
                    # 保存旋轉區塊視覺化
                    rotated_viz_path = f"{image_dir}/rotated_blocks/stage_{embedding}_blocks.png"
                    ensure_dir(rotated_viz_path)
                    save_image(rotated_block_visualization, rotated_viz_path)
                    
                    # 添加到階段信息
                    stage_info['rotated_block_visualization'] = rotated_block_visualization
                    
                    print(f"Saved rotated block visualization to {rotated_viz_path}")
                    
                    # 創建旋轉角度視覺化
                    rotation_colors = {
                        0: [200, 200, 200],      # 灰色表示無旋轉
                        90: [200, 100, 100],     # 紅色調表示90°
                        180: [100, 200, 100],    # 綠色調表示180°
                        270: [100, 100, 200],    # 藍色調表示270°
                        -90: [200, 200, 100],    # 黃色調表示-90°
                        -180: [200, 100, 200],   # 紫色調表示-180°
                        -270: [100, 200, 200]    # 青色調表示-270°
                    }
                    
                    # 創建RGB可視化
                    rotated_block_visualization_color = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    # 先填充灰度圖像
                    gray_img = cp.asnumpy(original_img)
                    for i in range(3):
                        rotated_block_visualization_color[:,:,i] = gray_img // 2  # 暗化原圖以便邊框更明顯
                    
                    # 根據旋轉角度繪製彩色邊框
                    for size_str in stage_info['block_info']:
                        size = int(size_str)
                        blocks = stage_info['block_info'][size_str]['blocks']
                        
                        # 跳過空區塊大小
                        if not blocks:
                            continue
                            
                        for block_info in blocks:
                            y, x = block_info['position']
                            
                            # 獲取此區塊大小的旋轉角度
                            rotation = stage_info['block_rotations'][size]
                            color = rotation_colors.get(rotation, [150, 150, 150])  # 未找到旋轉時使用默認灰色
                            
                            # 邊框寬度與區塊大小成比例
                            border_width = max(1, size // 64)
                            
                            # 繪製彩色邊框
                            rotated_block_visualization_color[y:y+border_width, x:x+size, :] = color  # 上方
                            rotated_block_visualization_color[y+size-border_width:y+size, x:x+size, :] = color  # 下方
                            rotated_block_visualization_color[y:y+size, x:x+border_width, :] = color  # 左側
                            rotated_block_visualization_color[y:y+size, x+size-border_width:x+size, :] = color  # 右側
                    
                    # 保存彩色可視化
                    color_viz_path = f"{image_dir}/rotated_blocks/stage_{embedding}_color.png"
                    ensure_dir(color_viz_path)
                    cv2.imwrite(color_viz_path, rotated_block_visualization_color)
                    
                    # 添加圖例
                    legend_img = np.ones((200, 400, 3), dtype=np.uint8) * 255  # 白色背景
                    legend_title = "Rotation Angles Legend"
                    cv2.putText(legend_img, legend_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    
                    # 添加各個旋轉角度的顏色示例
                    y_offset = 60
                    for angle, color in rotation_colors.items():
                        # 繪製顏色方塊
                        cv2.rectangle(legend_img, (10, y_offset), (40, y_offset+20), color, -1)
                        # 添加文字說明
                        cv2.putText(legend_img, f"{angle}°", (50, y_offset+15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                        y_offset += 30
                    
                    # 保存圖例
                    legend_path = f"{image_dir}/rotated_blocks/legend.png"
                    ensure_dir(legend_path)
                    cv2.imwrite(legend_path, legend_img)
                    
                    # 添加到階段信息
                    stage_info['rotated_block_visualization_color'] = rotated_block_visualization_color
                    
                    print(f"Saved colored rotated block visualization to {color_viz_path}")
                
                stage_info['stage_img'] = stage_img

                # 計算品質指標 - 這裡使用未旋轉的stage_img
                stage_img_np = cp.asnumpy(stage_img)
                reference_img_np = cp.asnumpy(original_img)
                psnr = calculate_psnr(reference_img_np, stage_img_np)
                ssim = calculate_ssim(reference_img_np, stage_img_np)
                hist_corr = histogram_correlation(
                    np.histogram(reference_img_np, bins=256, range=(0, 255))[0],
                    np.histogram(stage_img_np, bins=256, range=(0, 255))[0]
                )

                # 更新階段品質指標
                stage_info['psnr'] = float(psnr)
                stage_info['ssim'] = float(ssim)
                stage_info['hist_corr'] = float(hist_corr)
                stage_info['bpp'] = float(stage_info['payload'] / total_pixels)

                # 計算並顯示區塊大小分布
                block_counts = {}
                for size_str in stage_info['block_info']:
                    count = len(stage_info['block_info'][size_str]['blocks'])
                    if count > 0:
                        block_counts[size_str] = count
                
                # 添加到階段信息
                stage_info['block_counts'] = block_counts

                # 品質檢查和警告
                if psnr < 28 or ssim < 0.8:
                    print("Warning: Metrics seem unusually low")
                    # 如果使用的是random旋轉模式，提示可能是旋轉造成的
                    if rotation_mode == 'random':
                        print("This may be caused by the random rotation mode. Consider using rotation_mode='none'")

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

                # 輸出階段摘要
                print(f"\nEmbedding {embedding} summary:")
                print(f"Prediction Method: {prediction_method.value}")
                print(f"Payload: {stage_info['payload']}")
                print(f"BPP: {stage_info['bpp']:.4f}")
                print(f"PSNR: {stage_info['psnr']:.2f}")
                print(f"SSIM: {stage_info['ssim']:.4f}")
                print(f"Hist Corr: {stage_info['hist_corr']:.4f}")
                
                # 輸出區塊大小分布
                print("\nBlock size distribution:")
                for size_str in sorted(block_counts.keys(), key=int, reverse=True):
                    print(f"  {size_str}x{size_str}: {block_counts[size_str]} blocks")
                
                # 輸出目標payload資訊
                if remaining_target is not None:
                    if remaining_target[0] <= 0:
                        print(f"\nTarget payload of {target_payload_size} bits reached")
                        # 如果設置了精確控制目標，檢查實際嵌入量與目標的差距
                        if precise_control:
                            difference = total_payload - target_payload_size
                            if difference != 0:
                                print(f"Actual payload ({total_payload}) differs from target ({target_payload_size}) by {difference} bits")
                                print(f"Accuracy: {total_payload/target_payload_size*100:.2f}%")
                    else:
                        print(f"\nRemaining target payload: {remaining_target[0]} bits")
                
                # 精確控制：嘗試達到確切的目標payload
                if precise_control and target_payload_size > total_payload:
                    # 如果少於目標值且差距不大，嘗試填充差距
                    shortfall = target_payload_size - total_payload
                    if 0 < shortfall <= 1000:  # 小於1000位的差距可以嘗試填充
                        print(f"Attempting to fill missing {shortfall} bits to match exact target")
                        # 可以在這裡實現bit stuffing邏輯
                
                # 準備下一階段
                current_img = stage_img.copy()

                # 清理當前階段的記憶體
                mem_pool.free_all_blocks()
                print(f"Memory usage after embedding {embedding}: {mem_pool.used_bytes()/1024/1024:.2f} MB")

            # 如果圖像之前進行了墊充，現在需要裁剪回原始大小
            final_pee_img = cp.asnumpy(current_img)
            
            # 返回最終結果
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
        # 確保清理所有記憶體
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
    """
    import os
    import cv2
    import numpy as np
    import cupy as cp
    from color import split_color_channels, combine_color_channels
    from common import cleanup_memory
    
    if prediction_method is None:
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
    
    # 🔧 修改：改進目標容量分配邏輯
    if target_payload_size > 0:
        # 估算各通道容量（簡化版本）
        total_pixels = img.shape[0] * img.shape[1]
        estimated_capacity_per_channel = int(total_pixels * 0.4)  # 保守估計
        
        # 按比例分配，但考慮實際容量限制
        base_target = target_payload_size // 3
        blue_target = min(base_target, estimated_capacity_per_channel)
        green_target = min(base_target, estimated_capacity_per_channel)  
        red_target = target_payload_size - blue_target - green_target
        
        channel_targets = [blue_target, green_target, red_target]
        print(f"Target payload allocation - Blue: {blue_target}, Green: {green_target}, Red: {red_target}")
    else:
        channel_targets = [-1, -1, -1]
    
    # Process each channel separately
    print("\nProcessing blue channel...")
    final_b_img, b_payload, b_stages = pee_process_with_quadtree_cuda(
        b_channel, total_embeddings, ratio_of_ones, use_different_weights,
        min_block_size, variance_threshold, el_mode, rotation_mode,
        prediction_method=prediction_method,
        target_payload_size=channel_targets[0],
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
        target_payload_size=channel_targets[1],
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
        target_payload_size=channel_targets[2],
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
    
    # 🔧 修改：創建與灰階一致的階段信息結構
    for i in range(min(len(b_stages), len(g_stages), len(r_stages))):
        # Get stage info from each channel
        b_stage = b_stages[i]
        g_stage = g_stages[i]
        r_stage = r_stages[i]
        
        # 🔧 修改：確保與灰階圖像結構完全一致
        combined_stage = {
            'embedding': b_stage['embedding'],
            'payload': b_stage['payload'] + g_stage['payload'] + r_stage['payload'],
            'bpp': (b_stage['bpp'] + g_stage['bpp'] + r_stage['bpp']) / 3,
            'psnr': (b_stage['psnr'] + g_stage['psnr'] + r_stage['psnr']) / 3,
            'ssim': (b_stage['ssim'] + g_stage['ssim'] + r_stage['ssim']) / 3,
            'hist_corr': (b_stage['hist_corr'] + g_stage['hist_corr'] + r_stage['hist_corr']) / 3,
            
            # 🔧 新增：確保包含與灰階一致的必要欄位
            'rotation_mode': b_stage.get('rotation_mode', rotation_mode),
            'prediction_method': b_stage.get('prediction_method', prediction_method.value),
            
            # 🔧 修改：重構block_info為與灰階一致的結構
            'block_info': {},  # 先初始化為空，下面填充
            
            # 彩色圖像特有的詳細信息（保持向後兼容）
            'channel_payloads': {
                'blue': b_stage['payload'],
                'green': g_stage['payload'],
                'red': r_stage['payload']
            },
            'channel_metrics': {
                'blue': {'psnr': b_stage['psnr'], 'ssim': b_stage['ssim'], 'hist_corr': b_stage['hist_corr']},
                'green': {'psnr': g_stage['psnr'], 'ssim': g_stage['ssim'], 'hist_corr': g_stage['hist_corr']},
                'red': {'psnr': r_stage['psnr'], 'ssim': r_stage['ssim'], 'hist_corr': r_stage['hist_corr']}
            },
            
            # 保留原始通道block_info（用於詳細分析）
            'channel_block_info': {
                'blue': b_stage['block_info'],
                'green': g_stage['block_info'],
                'red': r_stage['block_info']
            }
        }
        
        # 🔧 修改：合併block_info為與灰階一致的扁平結構
        all_sizes = set(b_stage['block_info'].keys()) | set(g_stage['block_info'].keys()) | set(r_stage['block_info'].keys())
        for size_str in all_sizes:
            merged_blocks = []
            
            # 收集各通道的區塊，添加通道標識
            for channel_name, channel_stage in [('blue', b_stage), ('green', g_stage), ('red', r_stage)]:
                if size_str in channel_stage['block_info']:
                    for block in channel_stage['block_info'][size_str]['blocks']:
                        merged_block = block.copy()
                        merged_block['channel'] = channel_name  # 添加通道識別
                        merged_blocks.append(merged_block)
            
            if merged_blocks:
                combined_stage['block_info'][size_str] = {'blocks': merged_blocks}
        
        # 🔧 新增：合併階段圖像
        if 'stage_img' in b_stage and 'stage_img' in g_stage and 'stage_img' in r_stage:
            b_stage_img = cp.asnumpy(b_stage['stage_img']) if isinstance(b_stage['stage_img'], cp.ndarray) else b_stage['stage_img']
            g_stage_img = cp.asnumpy(g_stage['stage_img']) if isinstance(g_stage['stage_img'], cp.ndarray) else g_stage['stage_img']
            r_stage_img = cp.asnumpy(r_stage['stage_img']) if isinstance(r_stage['stage_img'], cp.ndarray) else r_stage['stage_img']
            
            combined_stage['stage_img'] = combine_color_channels(b_stage_img, g_stage_img, r_stage_img)
        
        # 🔧 新增：合併旋轉視覺化圖像（如果存在）
        if 'rotated_stage_img' in b_stage and 'rotated_stage_img' in g_stage and 'rotated_stage_img' in r_stage:
            try:
                b_rotated = cp.asnumpy(b_stage['rotated_stage_img']) if isinstance(b_stage['rotated_stage_img'], cp.ndarray) else b_stage['rotated_stage_img']
                g_rotated = cp.asnumpy(g_stage['rotated_stage_img']) if isinstance(g_stage['rotated_stage_img'], cp.ndarray) else g_stage['rotated_stage_img']
                r_rotated = cp.asnumpy(r_stage['rotated_stage_img']) if isinstance(r_stage['rotated_stage_img'], cp.ndarray) else r_stage['rotated_stage_img']
                
                combined_stage['rotated_stage_img'] = combine_color_channels(b_rotated, g_rotated, r_rotated)
            except Exception as e:
                print(f"Warning: Could not combine rotated stage images: {e}")
        
        # 🔧 新增：合併區塊視覺化（如果存在）
        if all('rotated_block_visualization' in stage for stage in [b_stage, g_stage, r_stage]):
            try:
                # 取藍色通道的區塊視覺化作為基礎
                combined_stage['rotated_block_visualization'] = b_stage['rotated_block_visualization']
                
                # 如果有彩色版本，則使用彩色版本
                if 'rotated_block_visualization_color' in b_stage:
                    combined_stage['rotated_block_visualization_color'] = b_stage['rotated_block_visualization_color']
            except Exception as e:
                print(f"Warning: Could not combine block visualizations: {e}")
        
        # 🔧 新增：添加區塊計數信息（與灰階一致）
        if 'block_counts' in b_stage:
            combined_block_counts = {}
            for stage in [b_stage, g_stage, r_stage]:
                if 'block_counts' in stage:
                    for size_str, count in stage['block_counts'].items():
                        if size_str not in combined_block_counts:
                            combined_block_counts[size_str] = 0
                        combined_block_counts[size_str] += count
            combined_stage['block_counts'] = combined_block_counts
            
        # Add stage to combined stages
        color_pee_stages.append(combined_stage)
        
        # Save stage image if imgName is provided
        if imgName and output_dir and 'stage_img' in combined_stage:
            stage_dir = f"{output_dir}/image/{imgName}/quadtree"
            os.makedirs(stage_dir, exist_ok=True)
            cv2.imwrite(f"{stage_dir}/color_stage_{i}_result.png", combined_stage['stage_img'])
    
    return final_color_img, total_payload, color_pee_stages