import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from common import find_max, calculate_psnr, calculate_ssim, histogram_correlation

def read_image(filepath, grayscale=True):
    """讀取圖像"""
    if grayscale:
        return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(filepath)

def save_image(image, filepath):
    """保存圖像"""
    cv2.imwrite(filepath, image)

def save_histogram(img, filename, title):
    plt.figure(figsize=(10, 6))
    plt.hist(img.flatten(), bins=256, range=[0,255], density=True, alpha=0.7)
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.savefig(filename)
    plt.close()

def save_difference_histogram(diff, filename, title):
    plt.figure(figsize=(10, 6))
    plt.hist(diff.flatten(), bins=100, range=[-50, 50], density=True, alpha=0.7)
    plt.title(title)
    plt.xlabel("Difference Value")
    plt.ylabel("Frequency")
    plt.savefig(filename)
    plt.close()

def calculate_correlation(img1, img2):
    """計算兩個圖像的相關係數"""
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    correlation = np.corrcoef(img1_flat, img2_flat)[0, 1]
    return round(correlation, 4)

def generate_histogram(array2D):
    """生成直方圖"""
    height, width = array2D.shape
    values = array2D.flatten()
    min_val = int(np.min(values))
    max_val = int(np.max(values))
    range_size = max_val - min_val + 1
    num = [0] * range_size
    for value in values:
        num[int(value) - min_val] += 1
    return num, min_val, max_val, range_size

def image_rerotation(image, times):
    """影像轉回原方向"""
    return np.rot90(image, -times % 4)

def check_quality_after_stage(stage_name, original_img, embedded_img):
    """檢查每個階段後的圖像質量"""
    psnr = calculate_psnr(original_img, embedded_img)
    ssim = calculate_ssim(original_img, embedded_img)
    hist_orig, _, _, _ = generate_histogram(original_img)
    hist_emb, _, _, _ = generate_histogram(embedded_img)
    corr = histogram_correlation(hist_orig, hist_emb)
    print(f"{stage_name}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, Histogram Correlation={corr:.4f}")

def array2D_transfer_to_array1D(array2D):
    """二維陣列轉換為一維陣列"""
    array1D = []
    row, column = array2D.shape
    for y in range(row):
        for x in range(column):
            value = array2D[y,x]
            array1D.append(1 if value >= 128 else 0)
    return array1D

def array1D_transfer_to_array2D(array1D):
    """一維陣列轉換為二維陣列"""
    length = len(array1D)
    side = int(length**0.5)
    array2D = np.zeros((side, side), dtype=np.uint8)
    i = 0
    for y in range(side):
        for x in range(side):
            array2D[y,x] = 255 if array1D[i] == 1 else 0
            i += 1
    return array2D

def two_array2D_add_or_subtract(array2D_1, array2D_2, sign):
    """兩個二維陣列的數值相加或相減"""
    return array2D_1 + sign * array2D_2

def find_w(image):
    """找出模塊的大小"""
    height, width = image.shape
    RLArray = [0] * height
    for y in range(height):
        RunLength = 0
        for x in range(width - 1):
            if image[y, x] == image[y, x + 1]:
                RunLength += 1
            else:
                RLArray[RunLength + 1] += 1
                RunLength = 0
    return find_max(RLArray)

def simplified_qrcode(qrcode):
    """簡化QRcode"""
    height, width = qrcode.shape
    same = 0
    simQrcode = None
    bits = 0
    for y in range(height - 1):
        if np.array_equal(qrcode[y], qrcode[y+1]):
            same += 1
        else:
            same = 0
        if same == 1:
            bits += 1
            if simQrcode is None:
                simQrcode = [qrcode[y]]
            else:
                simQrcode.append(qrcode[y])
    simQrcode = np.array(simQrcode)
    outQrcode = np.zeros((bits, bits), np.uint8)
    i = 0
    same = 0
    for x in range(width - 1):
        if np.array_equal(qrcode[:, x], qrcode[:, x+1]):
            same += 1
        else:
            same = 0
        if same == 1:
            outQrcode[:, i] = simQrcode[:, x]
            i += 1
    return bits, outQrcode

def simplified_stego(qrcode, bits, w):
    """簡化隱寫QRcode"""
    simQrcode = np.zeros((bits, bits), np.uint8)
    for j in range(bits):
        for i in range(bits):
            simQrcode[j, i] = qrcode[j*w, i*w]
    return simQrcode

def generate_perdict_image(img, weight):
    """生成預測影像"""
    height, width = img.shape
    temp = img.copy()
    for y in range(1, height-1):
        for x in range(1, width-1):
            ul = int(img[y-1,x-1])
            up = int(img[y-1,x])
            ur = int(img[y-1,x+1])
            left = int(img[y,x-1])
            p = (weight[0]*up+weight[1]*ul+weight[2]*ur+weight[3]*left)/(weight[0]+weight[1]+weight[2]+weight[3])
            temp[y,x] = round(p)
    return temp

def improved_predict_image(img, weight=None, block_size=8):
    """改进的预测图像生成"""
    height, width = img.shape
    pred_img = np.zeros_like(img)
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = img[i:i+block_size, j:j+block_size]
            if i == 0 and j == 0:
                pred_img[i:i+block_size, j:j+block_size] = np.mean(block)
            elif i == 0:
                pred_img[i:i+block_size, j:j+block_size] = block[:, 0][:, np.newaxis]
            elif j == 0:
                pred_img[i:i+block_size, j:j+block_size] = block[0, :][np.newaxis, :]
            else:
                if weight is None:
                    left = img[i:i+block_size, j-1][:, np.newaxis]
                    top = img[i-1, j:j+block_size][np.newaxis, :]
                    pred_img[i:i+block_size, j:j+block_size] = (left + top) / 2
                else:
                    for y in range(i, min(i+block_size, height)):
                        for x in range(j, min(j+block_size, width)):
                            ul = int(img[y-1, x-1]) if y > 0 and x > 0 else 0
                            up = int(img[y-1, x]) if y > 0 else 0
                            ur = int(img[y-1, x+1]) if y > 0 and x < width-1 else 0
                            left = int(img[y, x-1]) if x > 0 else 0
                            p = (weight[0]*up + weight[1]*ul + weight[2]*ur + weight[3]*left) / sum(weight)
                            pred_img[y, x] = round(p)

    return pred_img

def image_difference_shift(array2D, a):
    """影像差值偏移"""
    row, column = array2D.shape
    array2D_s = array2D.copy()
    func = a % 2
    r = np.floor(a/2)
    for j in range(1, row-1):
        for i in range(1, column-1):
            value = array2D[j,i]
            shift = value
            if func == 1 or a == 1:
                if value > r:
                    shift = value+r
                elif value < -r:
                    shift = value-r-1
            elif func == 0:
                if value > r:
                    shift = value+r
                elif value < (-r+1):
                    shift = value-r
            array2D_s[j,i] = shift
    return array2D_s

def find_locaion(qrcode, bits):
    """找出所有方格的左上角座標"""
    pixOfBits = qrcode.shape[0] // bits
    return np.array([np.arange(bits) * pixOfBits, np.arange(bits) * pixOfBits])