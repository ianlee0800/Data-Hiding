import numpy as np
import math
from cryptography.hazmat.primitives.asymmetric import rsa
from image_processing import (
    generate_histogram,
    improved_predict_image_cuda,
    split_image_into_blocks,
    merge_sub_images
)
from utils import (
    find_max,
    decode_pee_info,
    generate_random_binary_array,
    choose_el,
    find_best_weights_ga
)
from common import (
    calculate_psnr,
    calculate_ssim
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

@cuda.jit(nrt=True)
def pee_kernel_adaptive(img, pred_img, data, embedded, payload, EL):
    x, y = cuda.grid(2)
    if x < img.shape[0] and y < img.shape[1]:
        local_std = 0.0
        count = 0
        for i in range(max(0, x-1), min(img.shape[0], x+2)):
            for j in range(max(0, y-1), min(img.shape[1], y+2)):
                local_std += (float(img[i, j]) - float(pred_img[i, j])) ** 2
                count += 1
        if count > 0:
            local_std = (local_std / count) ** 0.5
        else:
            local_std = 0
        
        adaptive_EL = min(EL, max(3, int(EL * (1 - local_std / 255))))
        
        diff = int(img[x, y]) - int(pred_img[x, y])
        if abs(diff) < adaptive_EL and payload[0] < len(data):
            bit = data[int(payload[0])]
            cuda.atomic.add(payload, 0, 1)
            if diff >= 0:
                embedded[x, y] = min(255, max(0, img[x, y] + bit))
            else:
                embedded[x, y] = min(255, max(0, img[x, y] - bit))
        else:
            embedded[x, y] = img[x, y]

def pee_embedding_adaptive_cuda(img, data, pred_img, EL):
    if not isinstance(img, cp.ndarray):
        img = cp.asarray(img)
    if not isinstance(data, cp.ndarray):
        data = cp.asarray(data)
    if not isinstance(pred_img, cp.ndarray):
        pred_img = cp.asarray(pred_img)
    
    embedded = cp.zeros_like(img)
    d_payload = cp.zeros(1, dtype=cp.int32)
    
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (img.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
        (img.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    )
    
    pee_kernel_adaptive[blocks_per_grid, threads_per_block](img, pred_img, data, embedded, d_payload, EL)
    
    payload = int(d_payload[0])
    embedded_data = data[:payload].get()
    
    return embedded, payload, embedded_data

def pee_embedding_adaptive_cuda_split(img, data, base_EL, block_size=8):
    sub_images = split_image_into_blocks(img)
    embedded_sub_images = []
    total_payload = 0
    embedded_data = []
    block_params = []
    
    print(f"Debug: Number of sub-images: {len(sub_images)}")

    for i, sub_img in enumerate(sub_images):
        print(f"Debug: Processing sub-image {i}")
        sub_img_cp = cp.asarray(sub_img)
        sub_data = data[total_payload:total_payload + sub_img.size]
        
        # 為每個區塊獨立計算參數
        current_EL = choose_el(cp.asnumpy(sub_img_cp), i, total_payload)
        weights, _ = find_best_weights_ga(cp.asnumpy(sub_img_cp), sub_data, current_EL)
        
        print(f"Debug: Sub-image {i} - EL: {current_EL}, Weights: {weights}")

        embedded_img, payload, sub_embedded_data = pee_embedding_adaptive_cuda(
            sub_img_cp, sub_data, weights, current_EL, block_size)
        
        embedded_sub_images.append(cp.asnumpy(embedded_img))
        total_payload += payload
        embedded_data.extend(sub_embedded_data)
        
        # 儲存每個區塊的參數
        block_params.append({
            'weights': weights,
            'EL': current_EL,
            'payload': payload
        })
    
    merged_img = merge_sub_images(*embedded_sub_images)
    
    print(f"Debug: Total payload: {total_payload}")

    return merged_img, total_payload, embedded_data, block_params

def pee_process_with_rotation_cuda(img, rotations, ratio_of_ones, block_size=8):
    try:
        current_img = cp.asarray(img)
        total_payload = 0
        pee_stages = []
        
        for i in range(rotations):
            if i > 0:
                current_img = cp.rot90(current_img)
            
            print(f"Debug: Generating random data for rotation {i}")
            random_data = generate_random_binary_array(current_img.size, ratio_of_ones=ratio_of_ones)
            random_data_gpu = cp.asarray(random_data)
            
            try:
                print(f"Debug: Choosing EL for rotation {i}")
                EL = choose_el(cp.asnumpy(current_img), i, total_payload)
                
                print(f"Debug: Finding best weights for rotation {i}")
                weights, _ = find_best_weights_ga(cp.asnumpy(current_img), cp.asnumpy(random_data_gpu), EL)
                weights = cp.asarray(weights, dtype=cp.float32)
                
                print(f"Debug: Predicting image for rotation {i}")
                pred_img = improved_predict_image_cuda(current_img, weights)
                
                print(f"Debug: Embedding data for rotation {i}")
                embedded_img, payload, embedded_data = pee_embedding_adaptive_cuda(current_img, random_data_gpu, pred_img, EL)
                
                total_payload += payload
                
                print(f"Debug: Calculating PSNR and SSIM for rotation {i}")
                rotated_back = cp.rot90(embedded_img, -i % 4)
                psnr = calculate_psnr(cp.asnumpy(img), cp.asnumpy(rotated_back))
                ssim = calculate_ssim(cp.asnumpy(img), cp.asnumpy(rotated_back))
                
                pee_stages.append({
                    'image': cp.asnumpy(embedded_img),
                    'payload': payload,
                    'weights': weights.tolist(),
                    'EL': EL,
                    'psnr': psnr,
                    'ssim': ssim,
                    'block_params': [{'weights': weights.tolist(), 'EL': EL, 'payload': payload}]
                })
                
                current_img = embedded_img
            except Exception as e:
                print(f"Error in rotation {i}: {e}")
                current_img = cp.asarray(img)
        
        return cp.asnumpy(current_img), total_payload, pee_stages
    except Exception as e:
        print(f"Error in pee_process_with_rotation_cuda: {e}")
        return img, 0, []

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
        print(f"Encoded PEE info content: {encoded_pee_info}")
        raise

    # 計算最大可嵌入載荷
    max_payload = hist[peak]
    
    # 將 PEE 資訊轉換為二進制字符串
    embedding_data = ''
    embedding_data += format(total_rotations, '08b')
    for i in range(total_rotations):
        for w in weights[i]:
            # 將浮點數轉換為整數（乘以100並四捨五入），然後轉換為16位二進制
            embedding_data += format(int(round(w * 100)), '016b')
        embedding_data += format(ELs[i], '08b')
        embedding_data += format(payloads[i], '032b')
    
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

