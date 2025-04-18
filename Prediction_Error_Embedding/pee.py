import numpy as np
from numba import cuda
import itertools
import math
from image_processing import PredictionMethod, predict_image_cuda
from common import *

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

def pee_embedding_adaptive(img, data, pred_img, EL):
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

def pee_embedding_adaptive_cuda(img, data, pred_img, local_el, stage=0):
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

def multi_pass_embedding(img, data, local_el, weights, stage, 
                        prediction_method=PredictionMethod.PROPOSED,
                        remaining_target=None):
    """
    多種預測方法的嵌入函數，為不同預測器使用不同的嵌入策略
    """
    # 數據類型轉換
    if isinstance(img, cp.ndarray):
        img = cp.asnumpy(img)
    if isinstance(data, cp.ndarray):
        data = cp.asnumpy(data)
    
    # 限制數據量 - 為避免超額嵌入，實際目標設為略小於要求值
    if remaining_target is not None:
        safety_margin = 0.95  # 設置安全邊界為95%
        adjusted_target = int(remaining_target * safety_margin)
        data = data[:min(len(data), adjusted_target)]
        # 記錄調整
        print(f"嵌入控制：目標 {remaining_target} 調整為 {adjusted_target} (安全邊界: {safety_margin})")
    
    # 轉換為 CUDA 設備數組
    d_img = cuda.to_device(img)
    d_data = cuda.to_device(data)
    
    # 使用統一的預測函數接口
    pred_img = predict_image_cuda(d_img, prediction_method, weights)
    
    height, width = d_img.shape
    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    d_embedded = cuda.device_array_like(d_img)
    d_payload = cuda.to_device(np.array([0], dtype=np.int32))
    
    # 對於Stage 0，使用多次嵌入來增加容量
    if stage == 0:
        # 針對 Stage 0 的增強型多次嵌入
        passes = 3  # 使用3次嵌入來增加容量
        total_payload = 0
        
        for pass_idx in range(passes):
            if prediction_method == PredictionMethod.PROPOSED:
                if hasattr(local_el, 'copy_to_host'):
                    local_el_np = local_el.copy_to_host()
                elif isinstance(local_el, cp.ndarray):
                    local_el_np = cp.asnumpy(local_el)
                else:
                    local_el_np = local_el
                    
                d_local_el = cuda.to_device(local_el_np)
                
                # 使用加強版的嵌入核心，專為 Stage 0 優化
                pee_embedding_kernel[blocks_per_grid, threads_per_block](
                    d_img, pred_img, d_data[total_payload:], d_embedded, d_payload, d_local_el,
                    height, width, pass_idx
                )
            else:
                # 對於其他預測器，使用簡化的嵌入核心
                simple_single_embedding_kernel[blocks_per_grid, threads_per_block](
                    d_img, pred_img, d_data[total_payload:], d_embedded, d_payload,
                    height, width, pass_idx
                )
            
            # 更新嵌入結果和總容量
            current_payload = d_payload.copy_to_host()[0]
            if current_payload == 0:
                break  # 如果沒有嵌入任何數據，則停止
                
            total_payload += current_payload
            d_payload = cuda.to_device(np.array([0], dtype=np.int32))
            
            # 保存當前結果供下次嵌入使用 - 修正使用正確的方法
            # 先將結果複製到主機記憶體，然後再轉回設備
            temp_img = d_embedded.copy_to_host()
            d_img = cuda.to_device(temp_img)
            
            # 更新預測圖像
            pred_img = predict_image_cuda(d_img, prediction_method, weights)
        
        embedded = d_embedded.copy_to_host()
        payload = total_payload
    else:
        # 原有的嵌入邏輯
        if prediction_method == PredictionMethod.PROPOSED:
            if hasattr(local_el, 'copy_to_host'):
                local_el_np = local_el.copy_to_host()
            elif isinstance(local_el, cp.ndarray):
                local_el_np = cp.asnumpy(local_el)
            else:
                local_el_np = local_el
                
            d_local_el = cuda.to_device(local_el_np)
            
            # 使用改進的 PEE 嵌入函數
            pee_embedding_kernel[blocks_per_grid, threads_per_block](
                d_img, pred_img, d_data, d_embedded, d_payload, d_local_el,
                height, width, stage
            )
        else:
            # 對於其他預測器，使用簡單的單次嵌入方式
            simple_single_embedding_kernel[blocks_per_grid, threads_per_block](
                d_img, pred_img, d_data, d_embedded, d_payload,
                height, width, stage
            )
        
        # 獲取結果
        embedded = d_embedded.copy_to_host()
        payload = d_payload.copy_to_host()[0]
    
    # 在嵌入後嚴格檢查是否超出
    if remaining_target is not None and payload > remaining_target:
        print(f"注意: 內部調整後仍超額嵌入 ({payload} > {remaining_target})，強制截斷")
        payload = remaining_target
    
    return embedded, payload

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

@cuda.jit
def evaluate_weights_kernel(img, data, EL, weight_combinations, results, target_bpp, target_psnr, stage):
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

