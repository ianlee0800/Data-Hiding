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
def pee_embedding_kernel(img, pred_img, data, embedded, payload, EL, height, width):
    x, y = cuda.grid(2)
    if x < width and y < height:
        local_sum = 0.0
        count = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    diff = float(img[ny, nx]) - float(pred_img[ny, nx])
                    local_sum += diff * diff
                    count += 1
        
        local_std = math.sqrt(local_sum / count) if count > 0 else 0

        adaptive_EL = min(EL, max(3, int(EL * (1 - local_std / 255))))

        diff = int(img[y, x]) - int(pred_img[y, x])

        if abs(diff) < adaptive_EL and payload[0] < len(data):
            bit = data[payload[0]]
            cuda.atomic.add(payload, 0, 1)
            if diff >= 0:
                embedded[y, x] = min(255, max(0, int(img[y, x]) + bit))
            else:
                embedded[y, x] = min(255, max(0, int(img[y, x]) - bit))
        else:
            embedded[y, x] = img[y, x]

def pee_embedding_adaptive_cuda(img, data, pred_img, EL):
    
    # 确保输入是 cupy array
    if not isinstance(img, cp.ndarray):
        img = cp.asarray(img)
    if not isinstance(data, cp.ndarray):
        data = cp.asarray(data)
    if not isinstance(pred_img, cp.ndarray):
        pred_img = cp.asarray(pred_img)
    
    height, width = img.shape
    d_img = to_gpu(img)
    d_pred_img = to_gpu(pred_img)
    d_data = to_gpu(data)
    d_embedded = cuda.device_array_like(d_img)
    d_payload = cuda.to_device(np.array([0], dtype=np.int32))

    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    pee_embedding_kernel[blocks_per_grid, threads_per_block](
        d_img, d_pred_img, d_data, d_embedded, d_payload, EL, height, width
    )

    embedded = to_cpu(d_embedded)
    payload = int(d_payload.copy_to_host()[0])
    embedded_data = to_cpu(d_data[:payload]).tolist()

    return embedded, payload, embedded_data  # 确保返回 cupy array

def choose_pee_implementation(use_cuda=True):
    if use_cuda and cuda.is_available():
        return improved_predict_image_cuda, pee_embedding_adaptive_cuda
    else:
        return improved_predict_image_cpu, pee_embedding_adaptive

# 測試函數
def test_pee_functions():
    # 創建測試數據
    img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    data = np.random.randint(0, 2, 10000, dtype=np.uint8)
    EL = 5

    # 選擇實現（GPU 如果可用，否則 CPU）
    predict_func, embed_func = choose_pee_implementation()

    # 運行預測
    pred_img = predict_func(img, weights)

    # 運行嵌入
    embedded, payload, embedded_data = embed_func(img, data, pred_img, EL)

    # 使用 common.py 中的 calculate_psnr
    psnr = calculate_psnr(img, embedded)

    print(f"使用的實現: {'CUDA' if predict_func.__name__.endswith('cuda') else 'CPU'}")
    print(f"嵌入的數據量: {payload}")
    print(f"嵌入後圖像的形狀: {embedded.shape}")
    print(f"PSNR: {psnr}")
    print(f"原始圖像和嵌入後圖像的差異: {np.sum(img != embedded)}")

if __name__ == "__main__":
    test_pee_functions()