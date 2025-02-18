import numpy as np
from numba import cuda
import itertools
import math
from image_processing import PredictionMethod, predict_image_cuda
from common import *


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
def pee_embedding_kernel(img, pred_img, data, embedded, payload, local_el, height, width, stage):
    """
    改進的 PEE 嵌入 kernel，著重於提高 PSNR
    
    關鍵策略：
    1. 根據局部差值大小動態調整移動距離
    2. 優先使用較小的移動量
    3. 在可能的情況下保持像素在原始位置附近
    """
    x, y = cuda.grid(2)
    if x < width and y < height:
        diff = int(img[y, x]) - int(pred_img[y, x])
        pixel_val = int(img[y, x])
        el = local_el[y, x]
        
        # 計算局部複雜度
        local_complexity = 0
        window_size = 3
        for i in range(max(0, y-window_size), min(height, y+window_size+1)):
            for j in range(max(0, x-window_size), min(width, x+window_size+1)):
                if i != y or j != x:
                    d = abs(int(img[i, j]) - int(pred_img[i, j]))
                    local_complexity = max(local_complexity, d)
        
        # 根據局部複雜度調整嵌入強度
        adaptive_el = min(el, max(3, local_complexity // 2))
        embedding_space = adaptive_el * 2
        
        if payload[0] < data.size:
            if abs(diff) <= adaptive_el:  # 可嵌入區域
                bit = data[payload[0]]
                cuda.atomic.add(payload, 0, 1)
                
                # 優化的移動策略
                if diff >= 0:
                    if bit == 1:
                        # 漸進式移動：從小移動開始嘗試
                        for shift in range(adaptive_el, embedding_space + 1):
                            new_val = pixel_val + (shift - diff)
                            if 0 <= new_val <= 255:
                                embedded[y, x] = new_val
                                break
                        else:
                            # 如果所有嘗試都失敗，保持原值
                            embedded[y, x] = pixel_val
                    else:
                        # 保持原位以維持高 PSNR
                        embedded[y, x] = pixel_val
                else:
                    if bit == 1:
                        # 保持原位以維持高 PSNR
                        embedded[y, x] = pixel_val
                    else:
                        # 漸進式移動：從小移動開始嘗試
                        for shift in range(adaptive_el, embedding_space + 1):
                            new_val = pixel_val - (shift + abs(diff))
                            if 0 <= new_val <= 255:
                                embedded[y, x] = new_val
                                break
                        else:
                            embedded[y, x] = pixel_val
                            
            elif adaptive_el < abs(diff) <= embedding_space:
                # 智能移位策略：根據局部特性決定移動量
                if diff > 0:
                    # 計算最小需要的移動量
                    min_shift = embedding_space - diff + 1
                    for shift in range(min_shift, min_shift + adaptive_el):
                        new_val = pixel_val + shift
                        if 0 <= new_val <= 255:
                            embedded[y, x] = new_val
                            break
                    else:
                        embedded[y, x] = pixel_val
                else:
                    min_shift = embedding_space - abs(diff) + 1
                    for shift in range(min_shift, min_shift + adaptive_el):
                        new_val = pixel_val - shift
                        if 0 <= new_val <= 255:
                            embedded[y, x] = new_val
                            break
                    else:
                        embedded[y, x] = pixel_val
            else:
                # 保持原值
                embedded[y, x] = pixel_val
        else:
            embedded[y, x] = pixel_val

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
    改進的多次嵌入函數，支援多種預測方法
    
    Parameters:
    -----------
    img : numpy.ndarray
        輸入圖像
    data : numpy.ndarray
        要嵌入的數據
    local_el : numpy.ndarray
        局部嵌入層級
    weights : numpy.ndarray or None
        預測權重，只在使用 PROPOSED 方法時需要
    stage : int
        當前嵌入階段
    prediction_method : PredictionMethod
        使用的預測方法
    remaining_target : int, optional
        剩餘需要嵌入的數據量
        
    Returns:
    --------
    tuple
        (embedded_image, actual_payload)
    """
    # 數據類型轉換
    if isinstance(img, cp.ndarray):
        img = cp.asnumpy(img)
    if isinstance(data, cp.ndarray):
        data = cp.asnumpy(data)
    
    # 處理 local_el
    if hasattr(local_el, 'copy_to_host'):
        local_el_np = local_el.copy_to_host()
    elif isinstance(local_el, cp.ndarray):
        local_el_np = cp.asnumpy(local_el)
    else:
        local_el_np = local_el
    
    # 如果是 PROPOSED 方法，處理權重
    if prediction_method == PredictionMethod.PROPOSED:
        if isinstance(weights, cp.ndarray):
            weights = cp.asnumpy(weights)
        if weights is None:
            raise ValueError("Weights must be provided for PROPOSED method")
    
    # 限制數據量
    if remaining_target is not None:
        data = data[:min(len(data), remaining_target)]
    
    # 轉換為 CUDA 設備數組
    d_img = cuda.to_device(img)
    d_data = cuda.to_device(data)
    d_local_el = cuda.to_device(local_el_np)
    
    # 使用統一的預測函數接口
    pred_img = predict_image_cuda(d_img, prediction_method, weights)
    
    # 第一次嵌入
    embedded, payload, _ = pee_embedding_adaptive_cuda(
        d_img, d_data, pred_img, d_local_el, stage
    )
    
    # 嚴格控制 payload
    if remaining_target is not None and payload > remaining_target:
        print(f"Warning: Embedded payload ({payload}) exceeds target ({remaining_target})")
        print("Re-embedding with stricter control...")
        
        reduced_el = np.minimum(local_el_np, np.ones_like(local_el_np) * 3)
        d_reduced_el = cuda.to_device(reduced_el)
        
        remaining_data = data[:remaining_target]
        d_remaining_data = cuda.to_device(remaining_data)
        
        embedded, payload, _ = pee_embedding_adaptive_cuda(
            d_img, d_remaining_data, pred_img, d_reduced_el, stage
        )
    
    return embedded, min(payload, remaining_target) if remaining_target is not None else payload

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