import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from HS_encoder import *  # 確保 HS_encoder 模塊中有所需的函數

# 配置路徑
ORIGINAL_IMAGES_PATH = "./HS/images/"
MARKED_IMAGES_PATH = "./HS/marked/"
DECODED_IMAGES_PATH = "./HS/decoded/"

def read_image(file_path):
    """讀取影像"""
    return cv2.imread(file_path)

def extract_info(img, img_type, peak, height, width, shift):
    extracted_array = []
    if img_type == "Grayscale":
        for y in range(height):
            for x in range(width):
                if (shift == 1 and img[y, x, 0] == peak+1) or \
                   (shift == -1 and img[y, x, 0] == peak):
                    extracted_array.append(1)
    else:  # 彩色
        for c in range(3):
            for y in range(height):
                for x in range(width):
                    if (shift == 1 and img[y, x, c] == peak[c]+1) or \
                       (shift == -1 and img[y, x, c] == peak[c]):
                        extracted_array.append(1)
    return extracted_array

def restore_image(img, img_type, peak, height, width):
    out_img = img.copy()
    if img_type == "Grayscale":
        for y in range(height):
            for x in range(width):
                if out_img[y, x, 0] > peak:
                    out_img[y, x, 0] -= 1
    else:  # 彩色
        for c in range(3):
            for y in range(height):
                for x in range(width):
                    if out_img[y, x, c] > peak[c]:
                        out_img[y, x, c] -= 1
    return out_img

def calculate_metrics(orig_img, out_img, img_type):
    """計算 PSNR 和 SSIM"""
    if img_type == "Grayscale":
        psnr = calculate_psnr(orig_img, out_img)  
        ssim = calculate_ssim(orig_img, out_img)
    else:
        psnr = calculate_psnr(orig_img, out_img)
        ssim = calculate_ssim(orig_img, out_img)
    return psnr, ssim

def main():
    img_name = input("image name: ")
    file_type = "png"
    marked_img = read_image(f"{MARKED_IMAGES_PATH}{img_name}_markedImg.{file_type}")
    height, width = marked_img.shape[:2]

    # 读取嵌入信息与峰值信息
    hide_array = np.load(f"{HS_HIDE_DATA_PATH}{img_name}_HS_hide_data.npy")
    peak = np.load(f"{PEAK_PATH}{img_name}_peak.npy")

    # 读取label判断影像类型
    with open(f"{MARKED_IMAGES_PATH}{img_name}_info.txt") as f:
        img_type = f.read().strip()

    # 根据图像类型设置 shift
    if img_type == "Grayscale":
        shift = -1 if peak == 255 else 1
    else:  # 彩色
        # 检查 peak 是否是数组
        if peak.ndim > 0: 
            shift = -1 if any(p == 255 for p in peak) else 1
        else:
            shift = -1 if peak == 255 else 1

    # 影像还原与解码
    out_img = restore_image(marked_img, img_type, peak, height, width)
    extracted_array = extract_info(marked_img, img_type, peak, height, width, shift)

    # 驗證解碼結果
    if np.all(hide_array == extracted_array):
        print("decode_successful")
    else:
        print("decode_fail")

    # 計算指標
    orig_img = read_image(f"{ORIGINAL_IMAGES_PATH}{img_name}.{file_type}")
    psnr, ssim = calculate_metrics(orig_img, out_img, img_type)

    # 輸出結果
    print("Extracted message:", extracted_array)
    print("peak=", peak)
    print(f"{img_name} decode size= {height} x {width}")
    print("message_length=", len(extracted_array))
    print("PSNR:", psnr)
    print("SSIM:", ssim)

    # 顯示圖像
    cv2.imshow("Original Image", orig_img)
    cv2.imshow("Decoded Image", out_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
