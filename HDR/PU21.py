import numpy as np

def calculate_mse(img1, img2):
    """
    计算两个图像之间的均方误差(MSE)。
    """
    mse = np.mean((img1 - img2) ** 2, axis=(0, 1))
    return mse

def calculate_psnr(img1, img2, max_pixel=255):
    """
    计算PSNR值，现在会对每个颜色通道单独计算，然后取平均。
    """
    mse = calculate_mse(img1, img2)
    psnr_per_channel = 10 * np.log10((max_pixel ** 2) / mse)
    psnr = np.mean(psnr_per_channel)
    return psnr

def _ssim_channel(img1_c, img2_c):
    """
    计算两个图像通道之间的SSIM。
    """
    C1 = (255 * 0.01) ** 2
    C2 = (255 * 0.03) ** 2
    C3 = C2 / 2

    mean1 = np.mean(img1_c)
    mean2 = np.mean(img2_c)

    var1 = np.var(img1_c)
    var2 = np.var(img2_c)

    covar = np.cov(img1_c.flatten(), img2_c.flatten())[0][1]

    luminance = (2 * mean1 * mean2 + C1) / (mean1 ** 2 + mean2 ** 2 + C1)
    contrast = (2 * np.sqrt(var1) * np.sqrt(var2) + C2) / (var1 + var2 + C2)
    structure = (covar + C3) / (np.sqrt(var1) * np.sqrt(var2) + C3)

    ssim_channel = luminance * contrast * structure
    return ssim_channel

def calculate_ssim(img1, img2):
    """
    计算两个图像之间的平均SSIM值，对每个颜色通道单独计算，然后取平均。
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")
    
    ssim_total = 0
    channels = img1.shape[2] if img1.ndim == 3 else 1

    if channels > 1:
        for channel in range(channels):
            img1_c = img1[:, :, channel]
            img2_c = img2[:, :, channel]
            ssim_total += _ssim_channel(img1_c, img2_c)
    else:
        ssim_total = _ssim_channel(img1, img2)
    
    return ssim_total / channels

def encode_linear_to_pu(Y, params):
    """
    简化的线性亮度到PU空间的转换。
    """
    Y = np.clip(Y, 0.005, 10000)  # 假设的有效亮度范围
    V = params['scale'] * Y ** params['gamma']  # 简化的编码逻辑
    return V

def pu21_quality_assessment(image_ref, image_test, metric='PSNR', params={'scale': 1, 'gamma': 2.2}):
    """
    根据指定的质量评估指标（PSNR或SSIM），计算两幅图像之间的PU21质量分数。
    """
    if image_ref.shape != image_test.shape:
        raise ValueError("Input images must have the same dimensions.")

    # 对图像进行PU编码，假设encode_linear_to_pu可以处理彩色图像
    encoded_ref = encode_linear_to_pu(image_ref, params)
    encoded_test = encode_linear_to_pu(image_test, params)

    # 计算质量分数
    if metric.upper() == 'PSNR':
        quality_score = calculate_psnr(encoded_ref, encoded_test)
    elif metric.upper() == 'SSIM':
        quality_score = calculate_ssim(encoded_ref, encoded_test)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return quality_score