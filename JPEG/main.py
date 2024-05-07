import os
from PIL import Image
import numpy as np
from utils import extract_quantization_tables, VarianceEstimationDCT2D, JMiPODv0, dct2, idct2, stc_embed
import cv2

def main():
    # 設定輸入輸出資料夾路徑
    input_folder = "./JPEG/data"
    output_folder = "./JPEG/data"

    # 建立輸出資料夾(如果不存在)
    os.makedirs(output_folder, exist_ok=True)

    # 循環處理輸入資料夾中的每個圖像檔案
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
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

            # 設定嵌入率(bpp)
            Payload = 0.4

            # 對Y通道進行J-MiPOD嵌入
            stego_y, pChange_y = JMiPODv0(cover_y, C_STRUCT, Payload)

            # 使用syndrome coding模擬實際嵌入過程
            stego_y_syn = stc_embed(cover_y, pChange_y, C_STRUCT['quant_tables'][0])

            # 將嵌入後的Y通道與原始的Cb、Cr通道組合
            stego_image_ycbcr = np.stack((stego_y_syn, cover_cb, cover_cr), axis=2).astype(np.uint8)

            # 將YCbCr色彩空間轉換回RGB
            stego_image = Image.fromarray(stego_image_ycbcr, mode="YCbCr").convert("RGB")

            # 計算 PSNR 和 SSIM
            cover_image_np = np.array(cover_image)
            stego_image_np = np.array(stego_image)
            psnr = cv2.PSNR(cover_image_np, stego_image_np)
            ssim = cv2.SSIM(cover_image_np, stego_image_np)

            print(f"File: {filename}, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

            # 儲存嵌入後的圖像
            output_path = os.path.join(output_folder, f"stego_{filename}")
            stego_image.save(output_path, quality=95)

if __name__ == "__main__":
    main()