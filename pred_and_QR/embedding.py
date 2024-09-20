import numpy as np
import math
from cryptography.hazmat.primitives.asymmetric import rsa
from image_processing import (
    generate_perdict_image, 
    image_difference_shift, 
    two_array2D_add_or_subtract,
    generate_histogram,
    improved_predict_image_cuda,
    calculate_psnr,
    calculate_ssim
)
from utils import (
    find_max,
    decode_pee_info
)

try:
    import cupy as cp
    from numba import cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA libraries not found. Falling back to CPU implementation.")

def pee_embedding_adaptive(img, data, weight, block_size=8, threshold=3):
    height, width = img.shape
    pred_img = improved_predict_image_cuda(img, weight, block_size)
    diff = img - pred_img
    
    embedded = np.zeros_like(diff)
    embedded_data = []
    data_index = 0
    
    for i in range(height):
        for j in range(width):
            if data_index >= len(data):
                embedded[i, j] = diff[i, j]
                continue
            
            local_var = np.var(img[max(0, i-1):min(height, i+2), max(0, j-1):min(width, j+2)])
            adaptive_threshold = threshold * (1 + local_var / 1000)
            
            if abs(diff[i, j]) < adaptive_threshold:
                bit = data[data_index]
                if diff[i, j] >= 0:
                    embedded[i, j] = 2 * diff[i, j] + bit
                else:
                    embedded[i, j] = 2 * diff[i, j] - bit
                embedded_data.append(bit)
                data_index += 1
            else:
                embedded[i, j] = diff[i, j]
    
    embedded_img = pred_img + embedded
    return embedded_img, data_index, embedded_data

@cuda.jit
def pee_kernel_adaptive(img, pred_img, data, embedded, payload, base_threshold, EL):
    x, y = cuda.grid(2)
    if x < img.shape[0] and y < img.shape[1]:
        diff = float(img[x, y]) - float(pred_img[x, y])
        
        # 計算局部標準差
        local_std = 0.0
        count = 0
        for i in range(max(0, x-1), min(img.shape[0], x+2)):
            for j in range(max(0, y-1), min(img.shape[1], y+2)):
                local_std += (float(img[i, j]) - float(pred_img[i, j])) ** 2
                count += 1
        local_std = math.sqrt(local_std / count)
        
        # 自適應閾值
        threshold = min(base_threshold + int(local_std), EL // 2)
        
        if abs(diff) <= threshold:
            if payload[0] < len(data):
                bit = data[payload[0]]
                cuda.atomic.add(payload, 0, 1)
                if diff >= 0:
                    embedded[x, y] = pred_img[x, y] + EL * diff + bit
                else:
                    embedded[x, y] = pred_img[x, y] + EL * diff - bit
            else:
                embedded[x, y] = img[x, y]
        else:
            embedded[x, y] = img[x, y]

def pee_embedding_adaptive_cuda(img, data, weight, EL, block_size=8):
    if not CUDA_AVAILABLE:
        return pee_embedding_adaptive(img, data, weight, EL, block_size)

    d_img = cp.asarray(img)
    d_data = cp.asarray(data)
    
    d_pred_img = improved_predict_image_cuda(d_img, weight, block_size)

    d_embedded = cp.zeros_like(d_img)
    d_payload = cp.zeros(1, dtype=int)

    threads_per_block = (16, 16)
    blocks_per_grid_x = int(cp.ceil(img.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(cp.ceil(img.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # 計算全局標準差作為基礎閾值
    global_std = cp.std(d_img - d_pred_img).get()
    base_threshold = min(EL // 2, int(global_std))

    pee_kernel_adaptive[blocks_per_grid, threads_per_block](d_img, d_pred_img, d_data, d_embedded, d_payload, base_threshold, EL)

    embedded_img = d_embedded
    payload = int(d_payload[0])

    print(f"Debug: EL={EL}, base_threshold={base_threshold}, actual_payload={payload}")

    return embedded_img, payload, d_data[:payload]

def histogram_data_hiding(img, flag, encoded_pee_info):
    """Histogram Shifting Embedding"""
    # 確保輸入是 NumPy 陣列
    if isinstance(img, cp.ndarray):
        img = cp.asnumpy(img)

    h_img, w_img = img.shape
    markedImg = img.copy()
    hist, _, _, _ = generate_histogram(img)
    
    print("Debug: Histogram in histogram_data_hiding")
    print(f"Histogram type: {type(hist)}")
    print(f"Histogram length: {len(hist)}")
    print(f"Histogram sample: {hist[:10]}")
    
    peak = find_max(hist)
    print(f"Peak found in histogram_data_hiding: {peak}")
    
    try:
        # 解碼 PEE 資訊
        total_rotations, weights, payloads, ELs = decode_pee_info(encoded_pee_info)
    except (TypeError, ValueError) as e:
        print(f"Error decoding PEE info: {e}")
        print(f"Encoded PEE info type: {type(encoded_pee_info)}")
        print(f"Encoded PEE info length: {len(encoded_pee_info)} bytes")
        raise

    # 計算最大可嵌入載荷
    max_payload = hist[peak]
    
    # 將 PEE 資訊轉換為二進制字符串
    embedding_data = ''
    embedding_data += format(total_rotations, '08b')
    for i in range(total_rotations):
        for w in weights[i]:
            embedding_data += format(w, '08b')
        embedding_data += format(ELs[i], '08b')
        embedding_data += format(payloads[i], '016b')
    
    # 確保我們不超過最大載荷
    embedding_data = embedding_data[:max_payload]
    actual_payload = len(embedding_data)
    
    print(f"Debug: Embedding data length: {actual_payload} bits")
    print(f"Debug: Max payload: {max_payload} bits")
    
    i = 0
    for y in range(h_img):
        for x in range(w_img):
            if i < actual_payload and markedImg[y, x] == peak:
                if flag == 0:
                    markedImg[y, x] -= int(embedding_data[i])
                elif flag == 1:
                    markedImg[y, x] += int(embedding_data[i])
                i += 1
            elif markedImg[y, x] > peak:
                if flag == 1:
                    markedImg[y, x] += 1
            elif markedImg[y, x] < peak:
                if flag == 0:
                    markedImg[y, x] -= 1
    
    print(f"Debug: Actually embedded {i} bits")
    
    return markedImg, peak, actual_payload

