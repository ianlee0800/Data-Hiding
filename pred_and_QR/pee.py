import numpy as np
from image_processing import (
    improved_predict_image_cuda,
)

from common import *
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
        
        adaptive_EL = min(EL, max(3, int(EL * (1 - local_std / 255))))
        
        diff = int(img[x, y]) - int(pred_img[x, y])
        if abs(diff) < adaptive_EL and payload[0] < len(data):
            bit = int(data[int(payload[0])])
            cuda.atomic.add(payload, 0, 1)
            if diff >= 0:
                embedded[x, y] = min(255, max(0, int(img[x, y]) + bit))
            else:
                embedded[x, y] = min(255, max(0, int(img[x, y]) - bit))
        else:
            embedded[x, y] = img[x, y]

def pee_embedding_adaptive_cuda(img, data, pred_img, EL):
    try:
        threadsperblock = (16, 16)
        blockspergrid_x = (img.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (img.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        d_img = cuda.to_device(img.astype(np.uint8))
        d_data = cuda.to_device(data.astype(np.uint8))
        d_pred_img = cuda.to_device(pred_img.astype(np.uint8))
        d_embedded = cuda.device_array_like(img)
        d_payload = cuda.to_device(np.array([0], dtype=np.int32))
        
        pee_kernel_adaptive[blockspergrid, threadsperblock](d_img, d_pred_img, d_data, d_embedded, d_payload, EL)
        
        embedded = d_embedded.copy_to_host()
        payload = int(d_payload.copy_to_host()[0])
        embedded_data = d_data[:payload].copy_to_host()

        return embedded, payload, embedded_data
    except Exception as e:
        print(f"CUDA error: {e}. Falling back to CPU implementation.")
        return pee_embedding_adaptive_cpu(img, data, pred_img, EL)

def pee_embedding_adaptive_cpu(img, data, pred_img, EL):
    embedded = np.zeros_like(img)
    payload = 0
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # 计算局部标准差
            local_std = np.std(img[max(0, x-1):min(img.shape[0], x+2), max(0, y-1):min(img.shape[1], y+2)])
            
            # 自适应 EL
            adaptive_EL = min(EL, max(3, int(EL * (1 - local_std / 255))))
            
            diff = int(img[x, y]) - int(pred_img[x, y])
            if abs(diff) < adaptive_EL and payload < len(data):
                bit = data[payload]
                payload += 1
                if diff >= 0:
                    # 使用 np.clip 来避免溢出
                    embedded[x, y] = np.clip(int(img[x, y]) + bit, 0, 255)
                else:
                    # 使用 np.clip 来避免溢出
                    embedded[x, y] = np.clip(int(img[x, y]) - bit, 0, 255)
            else:
                embedded[x, y] = img[x, y]
    
    embedded_data = data[:payload]
    return embedded, payload, embedded_data

def pee_embedding_adaptive_cpu(img, data, pred_img, EL):
    embedded = np.zeros_like(img)
    payload = 0
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # 計算局部標準差
            local_std = np.std(img[max(0, x-1):min(img.shape[0], x+2), max(0, y-1):min(img.shape[1], y+2)])
            
            # 自適應 EL
            adaptive_EL = min(EL, max(3, int(EL * (1 - local_std / 255))))
            
            diff = int(img[x, y]) - int(pred_img[x, y])
            if abs(diff) < adaptive_EL and payload < len(data):
                bit = data[payload]
                payload += 1
                if diff >= 0:
                    embedded[x, y] = min(255, max(0, img[x, y] + bit))
                else:
                    embedded[x, y] = min(255, max(0, img[x, y] - bit))
            else:
                embedded[x, y] = img[x, y]
    
    embedded_data = data[:payload]
    return embedded, payload, embedded_data


