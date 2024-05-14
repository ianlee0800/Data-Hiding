import os
from PIL import Image
import numpy as np
from skimage.transform import resize
from utils import *

def main():
    # 設定輸入輸出資料夾路徑
    input_folder = "./JPEG/data"
    output_folder = "./JPEG/data"

    # 建立輸出資料夾(如果不存在)
    os.makedirs(output_folder, exist_ok=True)

    # 循環處理輸入資料夾中的每個圖像檔案
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            # 讀取圖像
            image_path = os.path.join(input_folder, filename)
            cover_image = Image.open(image_path)

            # 將圖像轉換為YCbCr色彩空間
            cover_image_ycbcr = cover_image.convert("YCbCr")

            # 提取Y, Cb, Cr通道
            cover_y = np.array(cover_image_ycbcr)[:, :, 0]
            cover_cb = np.array(cover_image_ycbcr)[:, :, 1]
            cover_cr = np.array(cover_image_ycbcr)[:, :, 2]

            # 提取JPEG量化表
            C_STRUCT = extract_quantization_tables(image_path)

            # 設定嵌入率(bpnzAC)
            Payload = 0.4

            # 檢查是否有未壓縮的DCT係數可用於側信息隱寫術
            si_available = False
            uncompressed_dct = None
            # ... (code to check for uncompressed DCT coefficients)

            # 對Y通道進行J-MiPOD嵌入
            stego_y, pChange_y, ChangeRate, Deflection = JMiPODv0(cover_y, C_STRUCT, Payload, si_available, uncompressed_dct)

            # 生成要嵌入的訊息
            payload_size = int(np.round(Payload * cover_y.size * np.log(2)))
            message = generate_message(payload_size)

            # 使用syndrome coding進行實際嵌入
            stego_y_syn = stc_embed(cover_y, pChange_y, C_STRUCT['quant_tables'][0], message)

            # 將stego_y_syn的值限制在0到255之間
            stego_y_syn = np.clip(stego_y_syn, 0, 255).astype(np.uint8)

            # 將嵌入後的Y通道與原始的Cb、Cr通道組合
            stego_image_ycbcr = np.stack((stego_y_syn, cover_cb, cover_cr), axis=2)

            # 將stego_image_ycbcr轉換為uint8類型
            stego_image_ycbcr = stego_image_ycbcr.astype(np.uint8)

            # 將YCbCr色彩空間轉換回RGB
            stego_image = Image.fromarray(stego_image_ycbcr, mode="YCbCr").convert("RGB")

            # 計算 PSNR
            cover_image_rgb = cover_image.convert("RGB")
            cover_image_np = np.array(cover_image_rgb)
            stego_image_np = np.array(stego_image)

            # 將圖像尺寸調整為與論文中使用的大小相同
            fixed_size = (512, 512)
            cover_image_np = resize(cover_image_np, fixed_size, anti_aliasing=True, preserve_range=True)
            stego_image_np = resize(stego_image_np, fixed_size, anti_aliasing=True, preserve_range=True)

            psnr = calculate_psnr(cover_image_np, stego_image_np)

            print(f"File: {filename}, PSNR: {psnr:.2f} dB, Change Rate: {ChangeRate:.4f}, Deflection: {Deflection:.4f}")

            # 儲存嵌入後的圖像
            output_path = os.path.join(output_folder, f"stego_{filename}")
            stego_image.save(output_path, quality=95)

def generate_message(payload_size):
    """
    Generates a random message of the specified payload size.
    """
    # Generate a random message of the specified payload size
    message = np.random.randint(0, 2, size=payload_size)
    
    return message

if __name__ == "__main__":
    main()