import numpy as np
import cupy as cp
from numba import cuda
import itertools
import math
from image_processing import PredictionMethod, predict_image_cuda

# =============================================================================
# EL (Embedding Level) 計算相關函數
# =============================================================================

@cuda.jit
def compute_improved_adaptive_el_kernel(img, local_el, window_size, max_el, block_size):
    """
    計算改進的自適應EL值的CUDA kernel，優化版本
    修改：EL值範圍改為1~15，不再限制為奇數
    """
    x, y = cuda.grid(2)
    if x < img.shape[1] and y < img.shape[0]:
        # 根據區塊大小調整window_size
        actual_window_size = window_size
        if block_size > 0:  # 使用正數來判斷是否有指定block_size
            if block_size >= 512:
                actual_window_size = min(window_size + 3, 9)  # 更大的視窗適合1024大小的圖像
            elif block_size >= 256:
                actual_window_size = min(window_size + 2, 7)
            elif block_size <= 64:
                actual_window_size = max(window_size - 1, 3)
        
        half_window = actual_window_size // 2
        
        # 計算局部統計量
        local_sum = 0
        local_sum_sq = 0
        count = 0
        
        for i in range(max(0, y - half_window), min(img.shape[0], y + half_window + 1)):
            for j in range(max(0, x - half_window), min(img.shape[1], x + half_window + 1)):
                pixel_value = img[i, j]
                local_sum += pixel_value
                local_sum_sq += pixel_value * pixel_value
                count += 1
        
        local_mean = local_sum / count
        local_variance = (local_sum_sq / count) - (local_mean * local_mean)
        
        # 修改：調整variance正規化策略，映射到1~15範圍
        max_variance = 6400  # 預設值
        if block_size > 0:
            if block_size >= 512:
                max_variance = 10000  # 針對更大的圖像調整此值
            elif block_size >= 256:
                max_variance = 8100
            elif block_size <= 64:
                max_variance = 4900
        
        # 正規化variance到0~1範圍
        normalized_variance = min(local_variance / max_variance, 1)
        
        # 修改：映射到1~15範圍，不再限制為奇數
        # 使用反比關係：variance越高，EL越小（更保守）
        el_value = int(15 - normalized_variance * 14)  # 15 - (0~14) = 15~1
        
        # 確保EL值在1~15範圍內（移除奇數限制）
        el_value = max(1, min(el_value, max_el))
        
        local_el[y, x] = el_value

def compute_improved_adaptive_el(img, window_size=5, max_el=15, block_size=None):
    """
    計算改進的自適應EL值
    修改：max_el預設改為15，支援1~15範圍
    """
    if isinstance(img, cp.ndarray):
        img = cp.asnumpy(img)
    
    local_el = cuda.device_array(img.shape, dtype=cp.int32)
    
    threads_per_block = (16, 16)
    blocks_per_grid_x = (img.shape[1] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (img.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # 將None轉換為-1，這樣在kernel中可以正確處理
    block_size_value = -1 if block_size is None else block_size
    
    compute_improved_adaptive_el_kernel[blocks_per_grid, threads_per_block](
        img, local_el, window_size, max_el, block_size_value
    )
    
    return local_el

# =============================================================================
# Variance 計算相關函數
# =============================================================================

@cuda.jit
def calculate_variance_kernel(block, variance_result):
    """
    CUDA kernel for calculating variance of image blocks
    """
    x, y = cuda.grid(2)
    if x < block.shape[1] and y < block.shape[0]:
        # 使用shared memory來優化性能
        tid = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x
        block_size = block.shape[0] * block.shape[1]
        
        # 計算區域平均值
        local_sum = 0
        local_sum_sq = 0
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                pixel_value = block[i, j]
                local_sum += pixel_value
                local_sum_sq += pixel_value * pixel_value
        
        mean = local_sum / block_size
        variance = (local_sum_sq / block_size) - (mean * mean)
        
        # 只需要一個線程寫入結果
        if x == 0 and y == 0:
            variance_result[0] = variance

def calculate_block_variance_cuda(block):
    """
    Calculate variance of a block using CUDA
    """
    threads_per_block = (16, 16)
    blocks_per_grid_x = (block.shape[1] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (block.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    variance_result = cuda.device_array(1, dtype=np.float32)
    
    calculate_variance_kernel[blocks_per_grid, threads_per_block](block, variance_result)
    
    return variance_result[0]

# =============================================================================
# 權重搜索相關函數
# =============================================================================

def brute_force_weight_search_cuda(img, data, local_el, target_bpp, target_psnr, stage, block_size=None):
    """
    使用暴力搜索找到最佳的權重組合
    
    Parameters:
    -----------
    img : numpy.ndarray or cupy.ndarray
        輸入圖像
    data : numpy.ndarray or cupy.ndarray
        要嵌入的數據
    local_el : numpy.ndarray or cupy.ndarray
        局部EL值
    target_bpp : float
        目標BPP
    target_psnr : float
        目標PSNR
    stage : int
        當前嵌入階段
    block_size : int, optional
        區塊大小（用於調整權重搜索範圍）
    
    Returns:
    --------
    tuple
        (最佳權重, (payload, psnr))
    """
    img = cp.asarray(img)
    data = cp.asarray(data)
    
    # 根據區塊大小調整權重範圍
    if block_size is not None:
        if block_size >= 256:
            # 大區塊使用更大的權重範圍
            weight_combinations = cp.array(list(itertools.product(range(1, 20), repeat=4)), dtype=cp.int32)
        elif block_size <= 32:
            # 小區塊使用較小的權重範圍以提高效能
            weight_combinations = cp.array(list(itertools.product(range(1, 8), repeat=4)), dtype=cp.int32)
        elif block_size <= 64:
            # 中小區塊使用中等權重範圍
            weight_combinations = cp.array(list(itertools.product(range(1, 12), repeat=4)), dtype=cp.int32)
        else:
            # 中等區塊使用標準權重範圍
            weight_combinations = cp.array(list(itertools.product(range(1, 16), repeat=4)), dtype=cp.int32)
    else:
        # 默認使用標準權重範圍
        weight_combinations = cp.array(list(itertools.product(range(1, 16), repeat=4)), dtype=cp.int32)
    
    # 初始化結果數組
    results = cp.zeros((len(weight_combinations), 3), dtype=cp.float32)
    
    # 配置CUDA運行參數
    threads_per_block = 256
    blocks_per_grid = (len(weight_combinations) + threads_per_block - 1) // threads_per_block
    
    # 調用評估kernel
    evaluate_weights_kernel[blocks_per_grid, threads_per_block](
        img, data, local_el, weight_combinations, results, 
        target_bpp, target_psnr, stage
    )
    
    # 根據區塊大小調整適應度計算
    if block_size is not None:
        if block_size >= 256:
            # 大區塊更重視PSNR
            results[:, 2] = results[:, 2] * 0.4 + (results[:, 1] / target_psnr) * 0.6
        elif block_size <= 32:
            # 小區塊更重視payload
            results[:, 2] = results[:, 2] * 0.7 + (results[:, 1] / target_psnr) * 0.3
        elif block_size <= 64:
            # 中小區塊平衡PSNR和payload
            results[:, 2] = results[:, 2] * 0.6 + (results[:, 1] / target_psnr) * 0.4
    
    # 找出最佳權重組合
    best_idx = cp.argmax(results[:, 2])
    best_weights = weight_combinations[best_idx]
    best_payload, best_psnr, best_fitness = results[best_idx]
    
    return cp.asnumpy(best_weights), (float(best_payload), float(best_psnr))

@cuda.jit
def evaluate_weights_kernel(img, data, EL, weight_combinations, results, target_bpp, target_psnr, stage):
    """權重評估核心"""
    idx = cuda.grid(1)
    if idx < weight_combinations.shape[0]:
        w1, w2, w3, w4 = weight_combinations[idx]
        
        height, width = img.shape
        payload = 0
        mse = 0.0
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Prediction
                ul = img[y-1, x-1]
                up = img[y-1, x]
                ur = img[y-1, x+1]
                left = img[y, x-1]
                p = (w1*up + w2*ul + w3*ur + w4*left) / (w1 + w2 + w3 + w4)
                pred_val = round(p)
                
                # Embedding
                diff = int(img[y, x]) - int(pred_val)
                if abs(diff) < EL[y, x] and payload < data.shape[0]:  # Use EL[y, x] instead of just EL
                    bit = data[payload]
                    payload += 1
                    if stage == 0:
                        # More aggressive embedding for stage 0
                        embedding_strength = min(3, EL[y, x] - abs(diff))
                    else:
                        embedding_strength = 1
                    
                    if diff >= 0:
                        embedded_val = min(255, int(img[y, x]) + bit * embedding_strength)
                    else:
                        embedded_val = max(0, int(img[y, x]) - (1 - bit) * embedding_strength)
                    mse += (embedded_val - img[y, x]) ** 2
                else:
                    mse += 0  # No change to pixel
        
        if mse > 0:
            psnr = 10 * math.log10((255 * 255) / (mse / (height * width)))
        else:
            psnr = 100.0  # High value for perfect embedding
        
        bpp = payload / (height * width)
        
        # Adaptive fitness criteria
        bpp_fitness = min(1.0, bpp / target_bpp)
        psnr_fitness = max(0, 1 - abs(psnr - target_psnr) / target_psnr)
        
        if stage == 0:
            fitness = bpp_fitness * 0.7 + psnr_fitness * 0.3
        else:
            fitness = bpp_fitness * 0.5 + psnr_fitness * 0.5
        
        results[idx, 0] = payload
        results[idx, 1] = psnr
        results[idx, 2] = fitness

# =============================================================================
# 預測器類別
# =============================================================================

class Predictors:
    @cuda.jit
    def predict_proposed_kernel(img, pred_img, weights, height, width):
        """PROPOSED 方法的 CUDA 預測核心"""
        x, y = cuda.grid(2)
        if 1 <= x < width - 1 and 1 <= y < height - 1:
            w1, w2, w3, w4 = weights[0], weights[1], weights[2], weights[3]
            
            ul = img[y-1, x-1]  # 左上
            up = img[y-1, x]    # 上
            ur = img[y-1, x+1]  # 右上
            left = img[y, x-1]  # 左
            
            # 加權預測
            pred_val = (w1*up + w2*ul + w3*ur + w4*left) / (w1 + w2 + w3 + w4)
            pred_img[y, x] = max(0, min(255, round(pred_val)))

    @cuda.jit  
    def predict_med_kernel(img, pred_img, height, width):
        """MED (Median Edge Detection) 方法的 CUDA 預測核心"""
        x, y = cuda.grid(2)
        if 1 <= x < width - 1 and 1 <= y < height - 1:
            up = img[y-1, x]      # N
            left = img[y, x-1]    # W  
            up_left = img[y-1, x-1]  # NW
            
            # MED 預測邏輯
            if up_left >= max(up, left):
                pred_val = min(up, left)
            elif up_left <= min(up, left):
                pred_val = max(up, left)
            else:
                pred_val = up + left - up_left
                
            pred_img[y, x] = max(0, min(255, int(pred_val)))

    @cuda.jit
    def predict_gap_kernel(img, pred_img, height, width):
        """GAP (Gradient Adjusted Prediction) 方法的 CUDA 預測核心"""
        x, y = cuda.grid(2)
        if 1 <= x < width - 1 and 1 <= y < height - 1:
            up = img[y-1, x]      # N
            left = img[y, x-1]    # W
            up_left = img[y-1, x-1]  # NW
            
            # 計算梯度
            grad_h = abs(left - up_left)
            grad_v = abs(up - up_left)
            
            # GAP 預測邏輯
            if grad_h < grad_v:
                pred_val = left
            elif grad_v < grad_h:
                pred_val = up
            else:
                pred_val = (left + up) / 2
                
            pred_img[y, x] = max(0, min(255, int(pred_val)))

    @cuda.jit
    def predict_rhombus_kernel(img, pred_img, height, width):
        """RHOMBUS 方法的 CUDA 預測核心"""
        x, y = cuda.grid(2)
        if 2 <= x < width - 2 and 2 <= y < height - 2:
            # Rhombus 模式的像素
            n = img[y-1, x]      # 北
            s = img[y+1, x]      # 南  
            e = img[y, x+1]      # 東
            w = img[y, x-1]      # 西
            ne = img[y-1, x+1]   # 東北
            nw = img[y-1, x-1]   # 西北
            se = img[y+1, x+1]   # 東南
            sw = img[y+1, x-1]   # 西南
            
            # Rhombus 預測（簡化版本）
            pred_val = (n + s + e + w + ne + nw + se + sw) / 8
            pred_img[y, x] = max(0, min(255, int(pred_val)))

def predict_image_cuda(img, prediction_method, weights=None):
    """
    使用 CUDA 進行圖像預測（適配 Predictors 類版本）
    
    Parameters:
    -----------
    img : cupy.ndarray or numpy.ndarray
        輸入圖像
    prediction_method : PredictionMethod
        預測方法
    weights : numpy.ndarray, optional
        權重（僅用於 PROPOSED 方法）
        
    Returns:
    --------
    cupy.ndarray : 預測圖像
    """
    # 確保輸入是 CuPy 陣列
    if not isinstance(img, cp.ndarray):
        img = cp.asarray(img)
    
    height, width = img.shape
    pred_img = cp.zeros_like(img)
    
    # 設置 CUDA 執行參數
    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # 🔧 關鍵修改：通過 Predictors 類調用 kernels
    if prediction_method == PredictionMethod.PROPOSED:
        if weights is None:
            # 使用默認權重
            weights = cp.array([1, 1, 1, 1], dtype=cp.int32)
        else:
            weights = cp.asarray(weights, dtype=cp.int32)
            
        # 🔧 修改：通過類調用
        Predictors.predict_proposed_kernel[blocks_per_grid, threads_per_block](
            img, pred_img, weights, height, width
        )
        
    elif prediction_method == PredictionMethod.MED:
        # 🔧 修改：通過類調用
        Predictors.predict_med_kernel[blocks_per_grid, threads_per_block](
            img, pred_img, height, width
        )
        
    elif prediction_method == PredictionMethod.GAP:
        # 🔧 修改：通過類調用
        Predictors.predict_gap_kernel[blocks_per_grid, threads_per_block](
            img, pred_img, height, width
        )
        
    elif prediction_method == PredictionMethod.RHOMBUS:
        # 🔧 修改：通過類調用
        Predictors.predict_rhombus_kernel[blocks_per_grid, threads_per_block](
            img, pred_img, height, width
        )
        
    else:
        raise ValueError(f"Unknown prediction method: {prediction_method}")
    
    return pred_img
    
# =============================================================================
# 嵌入核心邏輯
# =============================================================================

@cuda.jit
def simple_single_embedding_kernel(img, pred_img, data, embedded, payload, height, width, pass_idx):
    """
    簡化版的增強嵌入核心，用於非PROPOSED預測器
    """
    x, y = cuda.grid(2)
    if x < width and y < height:
        diff = int(img[y, x]) - int(pred_img[y, x])
        pixel_val = int(img[y, x])
        
        if payload[0] < data.size:
            # 基於通過次數調整嵌入策略
            if pass_idx == 0:  # 第一次通過 - 較寬鬆標準
                embed_threshold = 3  # 較大閾值，嵌入更多數據
            elif pass_idx == 1:  # 第二次通過 - 中等標準
                embed_threshold = 2
            else:  # 第三次通過 - 最嚴格標準
                embed_threshold = 1
                
            if abs(diff) <= embed_threshold:
                bit = data[payload[0]]
                cuda.atomic.add(payload, 0, 1)
                
                # 簡單嵌入策略
                if diff == 0:
                    if bit == 1:
                        embedded[y, x] = min(255, pixel_val + 1)
                    else:
                        embedded[y, x] = pixel_val
                elif diff > 0:  # 正差值
                    if bit == 1:
                        embedded[y, x] = pixel_val
                    else:
                        embedded[y, x] = max(0, pixel_val - 1)
                else:  # 負差值
                    if bit == 1:
                        embedded[y, x] = min(255, pixel_val + 1)
                    else:
                        embedded[y, x] = pixel_val
            else:
                embedded[y, x] = pixel_val
        else:
            embedded[y, x] = pixel_val
    else:
        embedded[y, x] = img[y, x]  # 邊界像素保持不變

@cuda.jit
def pee_embedding_kernel(img, pred_img, data, embedded, payload, local_el, height, width, pass_idx):
    """
    專為 Stage 0 優化的加強型嵌入核心，提高嵌入容量
    """
    x, y = cuda.grid(2)
    if x < width and y < height:
        diff = int(img[y, x]) - int(pred_img[y, x])
        pixel_val = int(img[y, x])
        el = local_el[y, x]
        
        # 增強型嵌入策略 - 根據通過次數調整策略
        if payload[0] < data.size:
            if pass_idx == 0:  # 第一次通過 - 最寬鬆的嵌入
                effective_el = min(el + 3, 9)  # 大幅擴大嵌入層級
                if abs(diff) <= effective_el:
                    bit = data[payload[0]]
                    cuda.atomic.add(payload, 0, 1)
                    
                    # 對零差值特殊處理
                    if diff == 0:
                        if bit == 1:
                            embedded[y, x] = min(255, pixel_val + 1)
                        else:
                            embedded[y, x] = pixel_val
                    # 對非零差值使用更積極的嵌入策略
                    elif abs(diff) <= 2:
                        if bit == 1:
                            if diff > 0:
                                embedded[y, x] = pixel_val
                            else:
                                embedded[y, x] = min(255, pixel_val + 1)
                        else:
                            if diff > 0:
                                embedded[y, x] = max(0, pixel_val - 1)
                            else:
                                embedded[y, x] = pixel_val
                    else:
                        # 更大的差值使用標準策略
                        if bit == 1:
                            if diff > 0:
                                embedded[y, x] = pixel_val
                            else:
                                embedded[y, x] = min(255, pixel_val + 1)
                        else:
                            if diff > 0:
                                embedded[y, x] = max(0, pixel_val - 1)
                            else:
                                embedded[y, x] = pixel_val
                else:
                    embedded[y, x] = pixel_val
            elif pass_idx == 1:  # 第二次通過 - 中等嵌入強度
                if abs(diff) <= 1:
                    bit = data[payload[0]]
                    cuda.atomic.add(payload, 0, 1)
                    
                    if diff == 0:
                        if bit == 1:
                            embedded[y, x] = min(255, pixel_val + 1)
                        else:
                            embedded[y, x] = pixel_val
                    else:  # diff == 1 or diff == -1
                        if bit == 1:
                            if diff > 0:
                                embedded[y, x] = pixel_val
                            else:
                                embedded[y, x] = min(255, pixel_val + 1)
                        else:
                            if diff > 0:
                                embedded[y, x] = max(0, pixel_val - 1)
                            else:
                                embedded[y, x] = pixel_val
                else:
                    embedded[y, x] = pixel_val
            else:  # 第三次通過 - 僅嵌入最安全的位置
                if diff == 0:
                    bit = data[payload[0]]
                    cuda.atomic.add(payload, 0, 1)
                    
                    if bit == 1:
                        embedded[y, x] = min(255, pixel_val + 1)
                    else:
                        embedded[y, x] = pixel_val
                else:
                    embedded[y, x] = pixel_val
        else:
            embedded[y, x] = pixel_val
    else:
        embedded[y, x] = img[y, x]  # 邊界像素保持不變

@cuda.jit
def rhombus_embedding_kernel(img, pred_img, data, embedded, payload, height, width, stage):
    """
    為 Rhombus 預測器設計的穩定嵌入 kernel
    簡化實現並確保結果穩定性
    """
    x, y = cuda.grid(2)
    if 1 < x < width - 1 and 1 < y < height - 1:
        diff = int(img[y, x]) - int(pred_img[y, x])
        pixel_val = int(img[y, x])
        
        if payload[0] < data.size:
            # 簡單穩定的嵌入邏輯
            # 只在差值為 0 時嵌入，確保最大穩定性
            if diff == 0:
                bit = data[payload[0]]
                cuda.atomic.add(payload, 0, 1)
                
                if bit == 1:
                    embedded[y, x] = min(255, pixel_val + 1)
                else:
                    embedded[y, x] = pixel_val
            else:
                # 維持原始像素值
                embedded[y, x] = pixel_val
        else:
            embedded[y, x] = pixel_val
    else:
        embedded[y, x] = img[y, x]

def multi_pass_embedding(img, data, local_el, weights, stage, 
                        prediction_method=PredictionMethod.PROPOSED,
                        remaining_target=None):
    """
    多種預測方法的嵌入函數，使用組合策略
    
    🔧 組合策略（策略1 + 策略2）：
    - PROPOSED預測器：
      * Stage 0: 使用 3次通過（最大嵌入容量）
      * Stage 1+: 使用 2次通過（平衡容量與品質）
    - MED/GAP/RHOMBUS預測器：
      * 所有Stage: 使用 1次通過（最快處理速度）
    
    這種策略的優勢：
    1. PROPOSED保持最高性能，同時後續Stage更注重品質
    2. 其他預測器獲得最大速度提升
    3. 明確的性能分層：高性能、平衡、高速度
    4. 最佳的資源利用效率
    
    Parameters:
    -----------
    img : numpy.ndarray or cupy.ndarray
        輸入圖像
    data : numpy.ndarray or cupy.ndarray
        要嵌入的數據
    local_el : numpy.ndarray
        局部嵌入層級
    weights : numpy.ndarray or None
        權重向量 (如果適用)
    stage : int
        當前嵌入階段
    prediction_method : PredictionMethod
        預測方法
    remaining_target : list or None
        剩餘需要嵌入的數據量的可變容器 [target_value]
        
    Returns:
    --------
    tuple
        (embedded_img, payload, pred_img)
    """
    # 數據類型轉換
    if isinstance(img, cp.ndarray):
        img = cp.asnumpy(img)
    if isinstance(data, cp.ndarray):
        data = cp.asnumpy(data)
    
    if remaining_target is not None and not isinstance(remaining_target, list):
        remaining_target = [remaining_target]
    
    # 精確容量控制 - 如果剩餘目標容量小於數據量的10%，使用精確嵌入
    precise_embedding = False
    if remaining_target is not None:
        # 如果剩餘容量非常小
        if remaining_target[0] <= len(data) * 0.1 and remaining_target[0] > 0:
            precise_embedding = True
    
    # 限制數據量 - 使用可變容器中的值
    current_target = None
    if remaining_target is not None:
        # 檢查是否還有剩餘容量
        if remaining_target[0] <= 0:
            # 已達到目標，直接返回原圖並且payload為0
            return img, 0, img  # 注意：在這裡預測圖像就是原圖
            
        # 根據精確模式的不同策略設置目標
        if precise_embedding:
            # 精確模式：嘗試嵌入恰好所需的位元
            current_target = remaining_target[0]
        else:
            # 普通模式：取較小值
            current_target = min(len(data), remaining_target[0])
            
        # 限制數據長度
        if current_target < len(data):
            data = data[:current_target]
    
    # 轉換為 CUDA 設備數組
    d_img = cuda.to_device(img)
    d_data = cuda.to_device(data)
    
    # 使用統一的預測函數接口
    pred_img = predict_image_cuda(d_img, prediction_method, weights)
    
    # 保存預測圖像的副本
    if hasattr(pred_img, 'copy_to_host'):
        pred_img_copy = pred_img.copy_to_host()
    else:
        pred_img_copy = pred_img
    
    height, width = d_img.shape
    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    d_embedded = cuda.device_array_like(d_img)
    d_payload = cuda.to_device(np.array([0], dtype=np.int32))
    
    # 精確嵌入模式特殊處理
    if precise_embedding:
        # 針對精確嵌入的優化策略
        # 1. 減少嵌入通道，只使用一次通過，避免過度嵌入
        # 2. 優先使用差值小的像素進行嵌入
        pass_idx = 0  # 只使用單次通過
        
        if prediction_method == PredictionMethod.PROPOSED:
            if hasattr(local_el, 'copy_to_host'):
                local_el_np = local_el.copy_to_host()
            elif isinstance(local_el, cp.ndarray):
                local_el_np = cp.asnumpy(local_el)
            else:
                local_el_np = local_el
                
            d_local_el = cuda.to_device(local_el_np)
            
            # 使用特殊參數的嵌入核心
            pee_embedding_kernel[blocks_per_grid, threads_per_block](
                d_img, pred_img, d_data, d_embedded, d_payload, d_local_el,
                height, width, pass_idx
            )
        else:
            # 對於其他預測器，同樣使用單次通過
            simple_single_embedding_kernel[blocks_per_grid, threads_per_block](
                d_img, pred_img, d_data, d_embedded, d_payload,
                height, width, pass_idx
            )
        
        # 獲取結果
        embedded = d_embedded.copy_to_host()
        payload = d_payload.copy_to_host()[0]
        
        # 更新剩餘目標容量
        if remaining_target is not None:
            actual_payload = min(payload, remaining_target[0])
            remaining_target[0] -= actual_payload
            payload = actual_payload
            
        return embedded, payload, pred_img_copy
    
    # 🔧 組合策略核心：根據預測方法和階段決定通過次數
    if prediction_method == PredictionMethod.PROPOSED:
        # 🎯 策略1：PROPOSED預測器使用動態通過次數
        if stage == 0:
            passes = 3  # Stage 0: 最大容量
            strategy_name = "PROPOSED Stage 0 (Maximum Capacity)"
        else:
            passes = 2  # Stage 1+: 平衡容量與品質
            strategy_name = f"PROPOSED Stage {stage} (Balanced)"
        use_multi_pass = True
    else:
        # 🎯 策略2：其他預測器統一使用1次通過
        passes = 1
        strategy_name = f"{prediction_method.value} Stage {stage} (High Speed)"
        use_multi_pass = False
    
    # 🔧 根據是否使用多次通過來選擇處理邏輯
    if use_multi_pass:
        # 多次通過邏輯（適用於PROPOSED預測器）
        total_payload = 0
        
        for pass_idx in range(passes):
            # 檢查是否已達到目標
            if remaining_target is not None and remaining_target[0] <= 0:
                break
                
            # PROPOSED預測器使用複雜的嵌入核心
            if hasattr(local_el, 'copy_to_host'):
                local_el_np = local_el.copy_to_host()
            elif isinstance(local_el, cp.ndarray):
                local_el_np = cp.asnumpy(local_el)
            else:
                local_el_np = local_el
                
            d_local_el = cuda.to_device(local_el_np)
            
            # 🔧 根據stage和pass_idx調整嵌入強度
            if stage == 0:
                # Stage 0 使用激進策略（原有邏輯）
                pee_embedding_kernel[blocks_per_grid, threads_per_block](
                    d_img, pred_img, d_data[total_payload:], d_embedded, d_payload, d_local_el,
                    height, width, pass_idx
                )
            else:
                # Stage 1+ 使用較保守的策略
                # 可以在這裡調整 pass_idx 或使用不同的參數
                # 例如：將 pass_idx 映射到更保守的值
                conservative_pass_idx = min(pass_idx + 1, 2)  # 讓後續stage更保守
                pee_embedding_kernel[blocks_per_grid, threads_per_block](
                    d_img, pred_img, d_data[total_payload:], d_embedded, d_payload, d_local_el,
                    height, width, conservative_pass_idx
                )
            
            # 更新嵌入結果和總容量
            current_payload = d_payload.copy_to_host()[0]
            if current_payload == 0:
                break  # 如果沒有嵌入任何數據，則停止
                
            total_payload += current_payload
            
            # 更新剩餘目標容量
            if remaining_target is not None:
                # 確保不會超過目標
                actual_payload = min(current_payload, remaining_target[0])
                remaining_target[0] -= actual_payload
                
                # 如果已達到目標，停止處理
                if remaining_target[0] <= 0:
                    break
            
            d_payload = cuda.to_device(np.array([0], dtype=np.int32))
            
            # 保存當前結果供下次嵌入使用
            temp_img = d_embedded.copy_to_host()
            d_img = cuda.to_device(temp_img)
            
            # 更新預測圖像
            pred_img = predict_image_cuda(d_img, prediction_method, weights)
            
            # 更新預測圖像副本
            if hasattr(pred_img, 'copy_to_host'):
                pred_img_copy = pred_img.copy_to_host()
            else:
                pred_img_copy = pred_img
        
        embedded = d_embedded.copy_to_host()
        payload = total_payload
        
        # 確保不會超過目標
        if remaining_target is not None:
            payload = min(payload, current_target)
            
    else:
        # 🔧 單次通過邏輯（適用於其他預測器的所有Stage）
        # 其他預測器使用簡化的單次嵌入方式
        simple_single_embedding_kernel[blocks_per_grid, threads_per_block](
            d_img, pred_img, d_data, d_embedded, d_payload,
            height, width, 0  # pass_idx 固定為 0，因為只有1次通過
        )
        
        # 獲取結果
        embedded = d_embedded.copy_to_host()
        payload = d_payload.copy_to_host()[0]
        
        # 更新剩餘目標容量
        if remaining_target is not None:
            # 確保不會超過目標
            actual_payload = min(payload, remaining_target[0])
            remaining_target[0] -= actual_payload
            payload = actual_payload  # 更新實際嵌入的量
    
    # 確保不會返回超過目標的payload值
    if current_target is not None:
        payload = min(payload, current_target)
    
    return embedded, payload, pred_img_copy

# =============================================================================
# 輔助嵌入功能
# =============================================================================

def pee_embedding_adaptive(img, data, pred_img, EL):
    """自適應 PEE 嵌入（CPU版本）"""
    height, width = img.shape
    embedded = np.zeros_like(img)
    payload = 0
    
    for x in range(height):
        for y in range(width):
            local_std = np.std(img[max(0, x-1):min(height, x+2), max(0, y-1):min(width, y+2)])
            adaptive_EL = min(EL, max(3, int(EL * (1 - local_std / 255))))
            
            diff = int(img[x, y]) - int(pred_img[x, y])
            if abs(diff) < adaptive_EL and payload < len(data):
                bit = int(data[payload])
                payload += 1
                if diff >= 0:
                    embedded[x, y] = min(255, max(0, int(img[x, y]) + bit))
                else:
                    embedded[x, y] = min(255, max(0, int(img[x, y]) - bit))
            else:
                embedded[x, y] = img[x, y]
    
    embedded_data = data[:payload].tolist()
    return embedded, payload, embedded_data

def pee_embedding_adaptive_cuda(img, data, pred_img, local_el, stage=0):
    """自適應 PEE 嵌入（CUDA版本）"""
    height, width = img.shape
    d_embedded = cuda.device_array_like(img)
    d_payload = cuda.to_device(np.array([0], dtype=np.int32))

    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    pee_embedding_kernel[blocks_per_grid, threads_per_block](
        img, pred_img, data, d_embedded, d_payload, local_el,
        height, width, stage
    )

    embedded = d_embedded.copy_to_host()
    payload = d_payload.copy_to_host()[0]
    embedded_data = data.copy_to_host()[:payload].tolist()

    return embedded, payload, embedded_data