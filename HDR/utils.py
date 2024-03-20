import numpy as np

def float_to_rgbe(hdr_image):
    rgbe_image = np.zeros((*hdr_image.shape[:2], 4), dtype=np.uint8)
    max_rgb = np.max(hdr_image[:, :, :3], axis=2)
    valid_mask = max_rgb >= 1e-32  # 避免除以0

    exp = np.floor(np.log2(max_rgb[valid_mask])) + 128
    exp = np.clip(exp, 128 - 128, 128 + 127).astype(np.uint8)

    scale_factor = (255.0 / max_rgb[valid_mask]) * (2.0 ** (exp - 128))
    scaled_rgb = hdr_image[:, :, :3][valid_mask] * np.expand_dims(scale_factor, axis=-1)
    scaled_rgb = np.clip(scaled_rgb, 0, 255).astype(np.uint8)

    rgbe_image[:, :, :3][valid_mask] = scaled_rgb
    rgbe_image[:, :, 3][valid_mask] = exp

    return rgbe_image

def rgbe_to_float(rgbe_image):
    """
    将RGBE格式的图像转换回浮点数格式的HDR图像。
    """
    # 分离RGB和E通道
    rgb = rgbe_image[:, :, :3].astype(np.float32)
    e = rgbe_image[:, :, 3].astype(np.float32)

    # 将指数E转换回浮点数的比例因子
    scale = 2.0 ** (e - 128.0)
    scale = np.expand_dims(scale, axis=-1)  # 使其形状与rgb数组匹配

    # 逆向计算原始的浮点数RGB值
    hdr_image = rgb * scale / 255.0

    return hdr_image