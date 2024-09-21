import numpy as np
import cupy as cp
import cv2
import math

def find_max(array1D):
    """找出一維陣列中最大值"""
    if not array1D:
        return None
    if isinstance(array1D[0], (list, np.ndarray)):
        return max(range(len(array1D)), key=lambda i: max(array1D[i]) if len(array1D[i]) > 0 else float('-inf'))
    else:
        return max(range(len(array1D)), key=lambda i: array1D[i])

def histogram_correlation(hist1, hist2):
    """計算兩個直方圖的相關係數"""
    max_length = max(len(hist1), len(hist2))
    hist1_padded = hist1 + [0] * (max_length - len(hist1))
    hist2_padded = hist2 + [0] * (max_length - len(hist2))
    correlation = np.corrcoef(hist1_padded, hist2_padded)[0, 1]
    return round(correlation, 4)

def calculate_psnr(img1, img2):
    """計算峰值訊噪比PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return round(psnr, 2)

def calculate_ssim(img1, img2):
    """計算結構相似性SSIM"""
    # 確保輸入是 NumPy 陣列
    if isinstance(img1, cp.ndarray):
        img1 = cp.asnumpy(img1)
    if isinstance(img2, cp.ndarray):
        img2 = cp.asnumpy(img2)

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

def improved_predict_image_cpu(img, weight, block_size=8):
    height, width = img.shape
    pred_img = np.zeros_like(img)
    
    for x in range(1, height):
        for y in range(1, width):
            ul = float(img[x-1, y-1])
            up = float(img[x-1, y])
            ur = float(img[x-1, y+1]) if y < width - 1 else up
            left = float(img[x, y-1])
            
            w0, w1, w2, w3 = weight[0], weight[1], weight[2], weight[3]
            w_sum = w0 + w1 + w2 + w3
            p = (w0*up + w1*ul + w2*ur + w3*left) / w_sum
            
            pred_img[x, y] = min(max(int(round(p)), 0), 255)
    
    return pred_img

def pee_embedding_adaptive_cpu(img, data, pred_img, EL):
    embedded = np.zeros_like(img)
    payload = 0
    height, width = img.shape
    
    for x in range(height):
        for y in range(width):
            local_window = img[max(0, x-1):min(height, x+2), max(0, y-1):min(width, y+2)]
            local_std = np.std(local_window)
            
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