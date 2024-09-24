import numpy as np
import cupy as cp
import cv2

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
    # 確保 hist 是一維數組
    hist = np.asarray(hist).flatten()
    
    # 避免選擇 255 作為峰值
    peak = np.argmax(hist[:-1])
    return peak

def histogram_correlation(hist1, hist2):
    hist1 = to_numpy(hist1)
    hist2 = to_numpy(hist2)
    
    max_length = max(len(hist1), len(hist2))
    hist1_padded = np.pad(hist1, (0, max_length - len(hist1)), 'constant')
    hist2_padded = np.pad(hist2, (0, max_length - len(hist2)), 'constant')
    
    correlation = np.corrcoef(hist1_padded, hist2_padded)[0, 1]
    return round(correlation, 4)

def calculate_psnr(img1, img2):
    img1 = to_numpy(img1)
    img2 = to_numpy(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
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

def choose_el_for_rotation(psnr, current_payload, total_pixels, rotation, total_rotations, target_payload=480000):
    """
    基于当前的 PSNR、已嵌入载荷、目标 payload 和旋转次数动态选择 EL 值
    
    :param psnr: 当前的 PSNR 值
    :param current_payload: 当前已嵌入的比特数
    :param total_pixels: 图像总像素数
    :param rotation: 当前旋转次数
    :param total_rotations: 总旋转次数
    :param target_payload: 目标总 payload
    :return: 选择的 EL 值 (1, 3, 5, 或 7)
    """
    remaining_payload = max(0, target_payload - current_payload)
    progress_factor = (total_rotations - rotation) / total_rotations
    payload_progress = current_payload / target_payload

    # 根据剩余 payload 和进度因子调整基础 EL 值
    if remaining_payload > target_payload * 0.5 and progress_factor > 0.4:
        base_el = 7
    elif remaining_payload > target_payload * 0.3 and progress_factor > 0.2:
        base_el = 5
    elif remaining_payload > target_payload * 0.1 or progress_factor > 0.1:
        base_el = 3
    else:
        base_el = 1

    # 根据 PSNR 和 payload 进度微调 EL 值
    if psnr > 50 and payload_progress < 0.8:
        el_adjustment = 2
    elif psnr > 45 and payload_progress < 0.9:
        el_adjustment = 1
    elif psnr < 40 or payload_progress > 0.95:
        el_adjustment = -1
    else:
        el_adjustment = 0

    adjusted_el = base_el + el_adjustment

    # 确保 EL 值在有效范围内 (1, 3, 5, 7)
    valid_els = [1, 3, 5, 7]
    return min(valid_els, key=lambda x: abs(x - adjusted_el))