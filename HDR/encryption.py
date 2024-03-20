import numpy as np
import os
from utils import rgbe_to_float
import cv2

def encrypt_image(rgbe_image, encryption_key):
    # 分離RGB和E通道
    rgb_channels = rgbe_image[:, :, :3]
    e_channel = rgbe_image[:, :, 3]
    
    # 確保至少有一個通道的值大於或等於128
    max_channel = np.max(rgb_channels, axis=2)
    e_channel[max_channel < 128] += 1
    
    # 生成隨機二進位數字RB
    rb = np.random.randint(0, 2, size=e_channel.shape, dtype=np.uint8)
    
    # 根據encryption_key生成二進位數字串
    key_bits = np.unpackbits(np.array([encryption_key], dtype=np.uint8))
    key_bits = np.tile(key_bits, (e_channel.shape[0], e_channel.shape[1], 1))
    key_bits = key_bits[:, :, :8]
    
    # 對RGB通道進行置換
    permuted_rgb = np.zeros_like(rgb_channels)
    for k in range(1, 8):
        p_i_j_k = np.bitwise_xor(rb, key_bits[:, :, k-1])
        mask = (e_channel == k)
        permuted_rgb[mask] = np.where(p_i_j_k[mask, np.newaxis], rgb_channels[mask][:, [1, 2, 0]], rgb_channels[mask])
    
    mask = (e_channel >= 8)
    permuted_rgb[mask] = rgb_channels[mask]
    
    # 更新E通道值
    e_channel -= rb
    
    # 組合置換後的RGB通道和更新後的E通道
    encrypted_image = np.concatenate((permuted_rgb, e_channel[:, :, np.newaxis]), axis=2)
    
    return encrypted_image

def save_encrypted_image(encrypted_image, image_name, directory="./HDR/encrypted"):
    # 從影像名稱中移除副檔名
    image_name_without_extension = os.path.splitext(image_name)[0]
    
    # 構建加密後的影像檔案名稱
    encrypted_filename = f"{image_name_without_extension}_encrypted.hdr"
    
    # 創建目錄(如果不存在)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 構建完整的檔案路徑
    file_path = os.path.join(directory, encrypted_filename)
    
    # 將加密後的影像從RGBE格式轉換回HDR格式
    hdr_image = rgbe_to_float(encrypted_image)
    
    # 將HDR影像轉換為OpenCV可寫入的格式(float32)
    hdr_image = hdr_image.astype(np.float32)
    
    # 儲存加密後的HDR影像
    cv2.imwrite(file_path, hdr_image)
    print(f"加密後的影像已儲存至 {file_path}")