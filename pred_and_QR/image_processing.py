import numpy as np
import cv2
import matplotlib.pyplot as plt
from common import find_max, calculate_psnr, calculate_ssim, histogram_correlation

try:
    import cupy as cp
    from numba import cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA libraries not found. Falling back to CPU implementation.")

def read_image(filepath, grayscale=True):
    """讀取圖像"""
    if grayscale:
        return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(filepath)

def save_image(image, filepath):
    """保存圖像"""
    cv2.imwrite(filepath, image)

def image_rerotation(image, times):
    """將影像旋轉回原方向"""
    if isinstance(image, np.ndarray):
        image = cp.asarray(image)
    return cp.rot90(image, -times % 4)

def save_histogram(img, filename, title):
    if isinstance(img, cp.ndarray):
        img = cp.asnumpy(img)
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

@cuda.jit
def improved_predict_image_kernel(img, weight, pred_img, block_size):
    x, y = cuda.grid(2)
    if 1 <= x < img.shape[0] and 1 <= y < img.shape[1]:
        ul = float(img[x-1, y-1])
        up = float(img[x-1, y])
        ur = float(img[x-1, y+1]) if y < img.shape[1] - 1 else up
        left = float(img[x, y-1])
        w_sum = weight[0] + weight[1] + weight[2] + weight[3]
        p = (weight[0]*up + weight[1]*ul + weight[2]*ur + weight[3]*left) / w_sum
        pred_img[x, y] = int(round(p))  # 確保結果是整數

def improved_predict_image_cuda(img, weight, block_size=8):
    if isinstance(img, np.ndarray):
        img = cp.asarray(img)
    
    d_img = img
    d_weight = cp.asarray(weight)
    d_pred_img = cp.zeros_like(d_img)

    threads_per_block = (16, 16)
    blocks_per_grid_x = int(cp.ceil(img.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(cp.ceil(img.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    improved_predict_image_kernel[blocks_per_grid, threads_per_block](d_img, d_weight, d_pred_img, block_size)

    return d_pred_img

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
