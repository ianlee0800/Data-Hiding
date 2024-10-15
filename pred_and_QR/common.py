import numpy as np
import cupy as cp
import cv2
from numba import cuda

class DataType:
    INT8 = np.int8
    UINT8 = np.uint8
    INT32 = np.int32
    FLOAT32 = np.float32

def to_numpy(data):
    if isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    return np.asarray(data)

def to_cupy(data):
    if isinstance(data, np.ndarray):
        return cp.asarray(data)
    return cp.asarray(to_numpy(data))

def ensure_type(data, dtype):
    if isinstance(data, cp.ndarray):
        return data.astype(dtype)
    return np.asarray(data, dtype=dtype)

def to_gpu(data):
    return cp.asarray(data)

def to_cpu(data):
    return cp.asnumpy(data)

def find_max(hist):
    """找到直方圖中的峰值"""
    if isinstance(hist, tuple):
        print("Warning: hist is a tuple. Using the first element.")
        hist = hist[0]
    
    if not isinstance(hist, np.ndarray):
        print(f"Warning: hist is not a numpy array. Type: {type(hist)}")
        hist = np.array(hist)
    
    if hist.ndim > 1:
        print(f"Warning: hist has {hist.ndim} dimensions. Flattening.")
        hist = hist.flatten()
    
    # 避免選擇 255 作為峰值
    peak = np.argmax(hist[:-1])
    return peak

def histogram_correlation(hist1, hist2):
    hist1 = hist1.astype(np.float64)
    hist2 = hist2.astype(np.float64)
    
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    mean1, mean2 = np.mean(hist1), np.mean(hist2)
    
    numerator = np.sum((hist1 - mean1) * (hist2 - mean2))
    denominator = np.sqrt(np.sum((hist1 - mean1)**2) * np.sum((hist2 - mean2)**2))
    
    if denominator == 0:
        return 1.0
    else:
        return numerator / denominator

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # 如果 MSE 为 0，返回无穷大
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    img1 = to_numpy(img1)
    img2 = to_numpy(img2)

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return round(ssim_map.mean(), 4)

def calculate_correlation(img1, img2):
    """計算兩個圖像的相關係數"""
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    correlation = np.corrcoef(img1_flat, img2_flat)[0, 1]
    return round(correlation, 4)

def improved_predict_image_cpu(img, weight):
    height, width = img.shape
    pred_img = np.zeros_like(img)
    
    for y in range(1, height-1):
        for x in range(1, width-1):
            ul = int(img[y-1,x-1])
            up = int(img[y-1,x])
            ur = int(img[y-1,x+1])
            left = int(img[y,x-1])
            p = (weight[0]*up+weight[1]*ul+weight[2]*ur+weight[3]*left)/np.sum(weight)
            pred_img[y,x] = round(p)
    
    return pred_img

@cuda.jit
def compute_improved_adaptive_el_kernel(img, local_el, window_size, max_el):
    x, y = cuda.grid(2)
    if x < img.shape[1] and y < img.shape[0]:
        local_sum = 0
        local_sum_sq = 0
        count = 0
        for i in range(max(0, y - window_size // 2), min(img.shape[0], y + window_size // 2 + 1)):
            for j in range(max(0, x - window_size // 2), min(img.shape[1], x + window_size // 2 + 1)):
                pixel_value = img[i, j]
                local_sum += pixel_value
                local_sum_sq += pixel_value * pixel_value
                count += 1
        
        local_mean = local_sum / count
        local_variance = (local_sum_sq / count) - (local_mean * local_mean)
        
        # 使用方差來調整EL值
        normalized_variance = min(local_variance / 6400, 1)  # 假設最大方差為6400 (80^2)
        el_value = int(max_el * (1 - normalized_variance))
        
        # 確保EL值為奇數
        el_value = max(1, min(el_value, max_el))
        if el_value % 2 == 0:
            el_value -= 1
        
        local_el[y, x] = el_value

def compute_improved_adaptive_el(img, window_size=5, max_el=7):
    local_el = cp.empty(img.shape, dtype=cp.int32)
    threads_per_block = (16, 16)
    blocks_per_grid_x = (img.shape[1] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (img.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    compute_improved_adaptive_el_kernel[blocks_per_grid, threads_per_block](img, local_el, window_size, max_el)
    
    return local_el