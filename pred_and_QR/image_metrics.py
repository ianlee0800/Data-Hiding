import numpy as np
import math

def calculate_psnr(img1, img2):
    """
    計算兩個圖像之間的峰值信噪比 (PSNR)
    :param img1: 第一個圖像 (numpy array)
    :param img2: 第二個圖像 (numpy array)
    :return: PSNR 值
    """
    if img1.shape != img2.shape:
        raise ValueError("兩個圖像的尺寸必須相同")
    
    height, width = img1.shape
    size_img = height * width
    max_pixel = 255.0
    
    mse = np.sum((img1.astype(float) - img2.astype(float)) ** 2) / size_img
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return round(psnr, 2)

def calculate_ssim(img1, img2):
    """
    計算兩個圖像之間的結構相似性 (SSIM)
    :param img1: 第一個圖像 (numpy array)
    :param img2: 第二個圖像 (numpy array)
    :return: SSIM 值
    """
    if img1.shape != img2.shape:
        raise ValueError("兩個圖像的尺寸必須相同")
    
    height, width = img1.shape
    size_img = height * width
    
    # 常數，避免除以零
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    
    # 計算平均值
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # 計算方差和協方差
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]
    
    # 計算 SSIM
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim = numerator / denominator
    
    return round(ssim, 4)

# 可以添加其他圖像質量度量函數
def calculate_mse(img1, img2):
    """
    計算兩個圖像之間的均方誤差 (MSE)
    :param img1: 第一個圖像 (numpy array)
    :param img2: 第二個圖像 (numpy array)
    :return: MSE 值
    """
    if img1.shape != img2.shape:
        raise ValueError("兩個圖像的尺寸必須相同")
    
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    return round(mse, 2)

# 如果需要直接運行此文件進行測試
if __name__ == "__main__":
    # 這裡可以添加一些測試代碼
    # 例如:
    test_img1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    test_img2 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    print(f"PSNR: {calculate_psnr(test_img1, test_img2)}")
    print(f"SSIM: {calculate_ssim(test_img1, test_img2)}")
    print(f"MSE: {calculate_mse(test_img1, test_img2)}")