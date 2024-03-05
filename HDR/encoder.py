import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt

def float_to_rgbe(hdr_image):
    rgbe_image = np.zeros((*hdr_image.shape[:2], 4), dtype=np.uint8)
    for i in range(hdr_image.shape[0]):
        for j in range(hdr_image.shape[1]):
            rgb = hdr_image[i, j, :3]
            max_value = np.max(rgb)
            if max_value < 1e-32:
                rgbe_image[i, j, :] = [0, 0, 0, 0]
                continue
            exp = int(math.floor(math.log(max_value, 2))) + 128
            exp = max(min(exp, 128 + 127), 128 - 128)
            
            if max_value > 1e-32:
                scale_factor = 2**exp / 255.0
                scaled_rgb = rgb / max_value * scale_factor
                scaled_rgb = np.clip(scaled_rgb, 0, 255)
            else:
                scaled_rgb = np.zeros(3)
                
            rgbe_image[i, j, :3] = scaled_rgb.astype(np.uint8)
            rgbe_image[i, j, 3] = exp
            
    return rgbe_image

def read_and_convert_hdr_to_rgbe(image_path):
    if not os.path.exists(image_path):
        print("檔案不存在，請確認檔案名稱和路徑是否正確。")
        return None
    hdr_image = cv2.imread(image_path, flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    rgbe_image = float_to_rgbe(hdr_image)
    return rgbe_image

def plot_histogram_of_e_values(rgbe_image):
    E_values = rgbe_image[:, :, 3].flatten()
    plt.hist(E_values, bins=256, range=(0, 255), color='blue', alpha=0.7)
    plt.title('Histogram of E values in the Image')
    plt.xlabel('E value')
    plt.ylabel('Frequency')
    plt.show()

# 讓用戶輸入圖像名稱
image_name = input("請輸入HDR圖像的名稱（包含檔案擴展名，例如：image.hdr）: ")
image_path = os.path.join('./HDR/HDR images', image_name)

# 讀取並轉換HDR影像
rgbe_image = read_and_convert_hdr_to_rgbe(image_path)

if rgbe_image is not None:
    print("HDR圖像讀取和轉換為RGBE格式完成。")
    plot_histogram_of_e_values(rgbe_image)
else:
    print("未能進行轉換。")