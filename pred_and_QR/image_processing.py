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
def improved_predict_image_kernel(img, weight, pred_img):
    x, y = cuda.grid(2)
    if x < img.shape[0] and y < img.shape[1]:
        if x > 0 and y > 0:
            ul = float(img[x-1, y-1])
            up = float(img[x-1, y])
            ur = float(img[x-1, y+1]) if y < img.shape[1] - 1 else up
            left = float(img[x, y-1])
            
            w0, w1, w2, w3 = weight[0], weight[1], weight[2], weight[3]
            w_sum = w0 + w1 + w2 + w3
            
            if w_sum > 0:
                p = (w0*up + w1*ul + w2*ur + w3*left) / w_sum
                pred_img[x, y] = min(max(int(p + 0.5), 0), 255)
            else:
                pred_img[x, y] = img[x, y]
        else:
            pred_img[x, y] = img[x, y]

def improved_predict_image_cuda(img, weight):
    threads_per_block = (32, 32)
    blocks_per_grid = ((img.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
                       (img.shape[1] + threads_per_block[1] - 1) // threads_per_block[1])
    
    d_img = cuda.to_device(img)
    d_weight = cuda.to_device(weight)
    d_pred_img = cuda.device_array_like(img)
    
    improved_predict_image_kernel[blocks_per_grid, threads_per_block](d_img, d_weight, d_pred_img)
    
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

def split_image_into_blocks(image):
    """
    將圖像分割成2x2的塊，並提取每個塊的四個角落像素。
    
    Args:
    image (numpy.ndarray): 輸入圖像
    
    Returns:
    tuple: 包含四個numpy.ndarray的元組，每個代表一個子圖像（左上、右上、左下、右下）
    """
    height, width = image.shape
    sub_height, sub_width = height // 2, width // 2
    
    top_left = image[0::2, 0::2]
    top_right = image[0::2, 1::2]
    bottom_left = image[1::2, 0::2]
    bottom_right = image[1::2, 1::2]
    
    return top_left, top_right, bottom_left, bottom_right

def merge_sub_images(top_left, top_right, bottom_left, bottom_right):
    """
    將四個子圖像合併成一個完整的圖像。
    
    Args:
    top_left, top_right, bottom_left, bottom_right (numpy.ndarray): 四個子圖像
    
    Returns:
    numpy.ndarray: 合併後的完整圖像
    """
    sub_height, sub_width = top_left.shape
    height, width = sub_height * 2, sub_width * 2
    
    merged = np.zeros((height, width), dtype=top_left.dtype)
    merged[0::2, 0::2] = top_left
    merged[0::2, 1::2] = top_right
    merged[1::2, 0::2] = bottom_left
    merged[1::2, 1::2] = bottom_right
    
    return merged

if __name__ == "__main__":
    import numpy as np
    from numba import cuda

    # 測試 improved_predict_image_cuda
    test_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    test_weight = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    
    result = improved_predict_image_cuda(test_img, test_weight)
    print("improved_predict_image_cuda test result shape:", result.shape)
    print("improved_predict_image_cuda test result sample:", result[0:5, 0:5])