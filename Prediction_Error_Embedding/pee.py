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
def simple_single_embedding_kernel(img, pred_img, data, embedded, payload, height, width, stage):
    """
    簡單的單次嵌入 kernel，適用於 MED 和 GAP 預測器
    保持簡單的實現，只關注基本功能
    """
    x, y = cuda.grid(2)
    if 1 < x < width - 1 and 1 < y < height - 1:
        diff = int(img[y, x]) - int(pred_img[y, x])
        pixel_val = int(img[y, x])
        
        if payload[0] < data.size:
            # Stage 1 嵌入規則
            if stage == 0:
                # 使用較寬鬆的嵌入條件
                if abs(diff) <= 2:  # 允許差值在 -2 到 2 的範圍內嵌入
                    bit = data[payload[0]]
                    cuda.atomic.add(payload, 0, 1)
                    
                    # 簡單的嵌入邏輯
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
            # Stage 2-5 嵌入規則
            else:
                if abs(diff) <= 1:  # 只在差值為 -1, 0, 1 時嵌入
                    bit = data[payload[0]]
                    cuda.atomic.add(payload, 0, 1)
                    
                    # 簡單的嵌入邏輯
                    if diff == 0:
                        if bit == 1:
                            embedded[y, x] = min(255, pixel_val + 1)
                        else:
                            embedded[y, x] = pixel_val
                    elif diff > 0:
                        if bit == 1:
                            embedded[y, x] = pixel_val
                        else:
                            embedded[y, x] = max(0, pixel_val - 1)
                    else:  # diff < 0
                        if bit == 1:
                            embedded[y, x] = min(255, pixel_val + 1)
                        else:
                            embedded[y, x] = pixel_val
                else:
                    embedded[y, x] = pixel_val
        else:
            embedded[y, x] = pixel_val
    else:
        embedded[y, x] = img[y, x]

@cuda.jit
def pee_embedding_kernel(img, pred_img, data, embedded, payload, local_el, height, width, stage):
    """
    改進的 PEE 嵌入 kernel，支援 triple/double embedding 策略
    為 PROPOSED 預測器優化嵌入能力
    """
    x, y = cuda.grid(2)
    if x < width and y < height:
        diff = int(img[y, x]) - int(pred_img[y, x])
        pixel_val = int(img[y, x])
        el = local_el[y, x]
        
        # 嵌入條件根據階段不同而異
        if payload[0] < data.size:
            # Stage 1 使用 triple embedding
            if stage == 0:
                # 擴大嵌入範圍，以增加嵌入量
                effective_el = min(el + 2, 7)  # 擴大嵌入層級，但不超過7
                
                if abs(diff) <= effective_el:
                    bit = data[payload[0]]
                    cuda.atomic.add(payload, 0, 1)
                    
                    # Triple embedding 邏輯
                    if diff == 0:
                        if bit == 1:
                            embedded[y, x] = min(255, pixel_val + 1)
                        else:
                            embedded[y, x] = pixel_val
                    elif diff > 0:  # 正差值
                        if bit == 1:
                            embedded[y, x] = pixel_val  # 保持原值
                        else:
                            embedded[y, x] = max(0, pixel_val - 1)  # 減少1
                    else:  # 負差值
                        if bit == 1:
                            embedded[y, x] = min(255, pixel_val + 1)  # 增加1
                        else:
                            embedded[y, x] = pixel_val  # 保持原值
                else:
                    # 超出嵌入範圍的處理
                    embedded[y, x] = pixel_val
            # Stage 2-5 使用 double embedding
            else:
                if abs(diff) <= el:  # 使用原始 EL
                    bit = data[payload[0]]
                    cuda.atomic.add(payload, 0, 1)
                    
                    # Double embedding 邏輯
                    if diff == 0:
                        if bit == 1:
                            embedded[y, x] = min(255, pixel_val + 1)
                        else:
                            embedded[y, x] = pixel_val
                    elif diff > 0:
                        if bit == 1:
                            embedded[y, x] = pixel_val
                        else:
                            embedded[y, x] = max(0, pixel_val - 1)
                    else:  # diff < 0
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
    
    # 限制數據量
    if remaining_target is not None:
        data = data[:min(len(data), remaining_target)]
    
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
    
    # 根據預測方法選擇不同的嵌入策略
    if prediction_method == PredictionMethod.PROPOSED:
        # 對於 PROPOSED 預測器，使用原有的嵌入方式
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
    elif prediction_method == PredictionMethod.RHOMBUS:
        # 對於 RHOMBUS 預測器，使用專用的嵌入核心
        rhombus_embedding_kernel[blocks_per_grid, threads_per_block](
            d_img, pred_img, d_data, d_embedded, d_payload,
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
    
    # 嚴格控制 payload
    if remaining_target is not None and payload > remaining_target:
        print(f"警告: 嵌入的 payload ({payload}) 超過目標 ({remaining_target})")
        payload = remaining_target
    
    return embedded, min(payload, remaining_target) if remaining_target is not None else payload

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