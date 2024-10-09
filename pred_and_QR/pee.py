import numpy as np
from numba import cuda
import math
from image_processing import improved_predict_image_cuda
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
    x, y = cuda.grid(2)
    if x < width and y < height:
        diff = int(img[y, x]) - int(pred_img[y, x])
        
        if abs(diff) < local_el[y, x] and payload[0] < data.size:
            bit = data[payload[0]]
            cuda.atomic.add(payload, 0, 1)
            
            if stage == 0:
                embedding_strength = min(3, local_el[y, x] - abs(diff))
            else:
                embedding_strength = 1
            
            if diff >= 0:
                embedded[y, x] = min(255, int(img[y, x]) + bit * embedding_strength)
            else:
                embedded[y, x] = max(0, int(img[y, x]) - (1 - bit) * embedding_strength)
        else:
            embedded[y, x] = img[y, x]

def pee_embedding_adaptive_cuda(img, data, pred_img, local_el, stage=0):
    height, width = img.shape
    d_embedded = cuda.device_array_like(img)
    d_payload = cuda.to_device(np.array([0], dtype=np.int32))

    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    pee_embedding_kernel[blocks_per_grid, threads_per_block](
        img, pred_img, data, d_embedded, d_payload, local_el, height, width, stage
    )

    embedded = d_embedded.copy_to_host()
    payload = d_payload.copy_to_host()[0]
    embedded_data = data.copy_to_host()[:payload].tolist()

    return embedded, payload, embedded_data

def multi_pass_embedding(img, data, local_el, weights, stage):
    
    # Convert CuPy arrays to NumPy if necessary
    if isinstance(img, cp.ndarray):
        img = cp.asnumpy(img)
    if isinstance(data, cp.ndarray):
        data = cp.asnumpy(data)
    
    # Handle local_el specifically since it's a DeviceNDArray
    if isinstance(local_el, cuda.cudadrv.devicearray.DeviceNDArray):
        local_el_np = local_el.copy_to_host()
    elif isinstance(local_el, cp.ndarray):
        local_el_np = cp.asnumpy(local_el)
    else:
        local_el_np = local_el
    
    if isinstance(weights, cp.ndarray):
        weights = cp.asnumpy(weights)
    
    # Convert NumPy arrays to Numba device arrays
    d_img = cuda.to_device(img)
    d_data = cuda.to_device(data)
    d_local_el = cuda.to_device(local_el_np)
    d_weights = cuda.to_device(weights)
    
    pred_img = improved_predict_image_cuda(d_img, d_weights)
    embedded, payload, _ = pee_embedding_adaptive_cuda(d_img, d_data, pred_img, d_local_el, stage)
    
    if stage == 0 and payload < data.size:
        # Second pass with reduced EL
        second_el_np = np.minimum(np.maximum(local_el_np - 1, 1), 3)
        d_second_el = cuda.to_device(second_el_np)
        second_pred_img = improved_predict_image_cuda(embedded, d_weights)
        embedded, additional_payload, _ = pee_embedding_adaptive_cuda(embedded, d_data[payload:], second_pred_img, d_second_el, stage)
        payload += additional_payload
    
    return embedded, payload