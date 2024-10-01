import numpy as np
import cv2
import matplotlib.pyplot as plt
from common import find_max, calculate_psnr, calculate_ssim, histogram_correlation
import cupy as cp
from numba import cuda

def read_image(filepath, grayscale=True):
    """讀取圖像"""
    if grayscale:
        return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(filepath)

def save_image(image, filepath):
    """保存圖像"""
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array")
    
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
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
    hist, _ = np.histogram(array2D, bins=256, range=(0, 255))
    return hist

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

def split_image(img):
    if isinstance(img, cp.ndarray):
        xp = cp
    else:
        xp = np
    height, width = img.shape
    sub_height, sub_width = height // 2, width // 2
    sub_images = [
        img[0::2, 0::2],  # 左上
        img[0::2, 1::2],  # 右上
        img[1::2, 0::2],  # 左下
        img[1::2, 1::2]   # 右下
    ]
    return sub_images

def split_image_into_quarters(img):
    h, w = img.shape
    mid_h, mid_w = h // 2, w // 2
    return [
        img[:mid_h, :mid_w],
        img[:mid_h, mid_w:],
        img[mid_h:, :mid_w],
        img[mid_h:, mid_w:]
    ]

def merge_image(sub_images):
    if not sub_images:
        raise ValueError("No sub-images to merge")
    
    print(f"Number of sub-images to merge: {len(sub_images)}")
    print(f"Sub-image shapes: {[img.shape for img in sub_images]}")
    
    if isinstance(sub_images[0], cp.ndarray):
        xp = cp
    else:
        xp = np
    
    sub_height, sub_width = sub_images[0].shape
    height, width = sub_height * 2, sub_width * 2
    
    merged = xp.zeros((height, width), dtype=sub_images[0].dtype)
    merged[0::2, 0::2] = sub_images[0]
    merged[0::2, 1::2] = sub_images[1]
    merged[1::2, 0::2] = sub_images[2]
    merged[1::2, 1::2] = sub_images[3]
    
    print(f"Merged image shape: {merged.shape}")
    
    return merged

@cuda.jit
def improved_predict_kernel(img, weight, pred_img, height, width):
    x, y = cuda.grid(2)
    if x < width and y < height:
        if 0 < x < width-1 and 0 < y < height-1:
            ul = img[y-1, x-1]
            up = img[y-1, x]
            ur = img[y-1, x+1]
            left = img[y, x-1]
            p = (weight[0]*up + weight[1]*ul + weight[2]*ur + weight[3]*left) / \
                (weight[0] + weight[1] + weight[2] + weight[3])
            pred_img[y, x] = round(p)
        else:
            pred_img[y, x] = img[y, x]

def improved_predict_image_cuda(img, weights):
    # 确保输入是 CuPy 数组
    img = cp.asarray(img)
    weights = cp.asarray(weights)
    """
    CUDA 版本的改進預測圖像函數
    
    :param img: numpy array，輸入圖像
    :param weights: numpy array，預測權重
    :return: numpy array，預測圖像
    """
    height, width = img.shape
    d_img = cuda.to_device(img)
    d_weights = cuda.to_device(weights)
    d_pred_img = cuda.device_array_like(img)

    # 設置網格和塊大小
    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # 執行 kernel
    improved_predict_kernel[blocks_per_grid, threads_per_block](d_img, d_weights, d_pred_img, height, width)

    # 將結果複製回主機
    pred_img = d_pred_img.copy_to_host()
    
    return cp.asarray(pred_img)  # 确保返回 CuPy 数组

def create_collage(images):
    """Create a 2x2 collage from four images."""
    assert len(images) == 4, "Must provide exactly 4 images for the collage"
    
    # Ensure all images are numpy arrays
    images = [np.array(img) if not isinstance(img, np.ndarray) else img for img in images]
    
    # Find the maximum dimensions
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    
    # Create the collage
    collage = np.zeros((max_height * 2, max_width * 2), dtype=np.uint8)
    
    positions = [(0, 0), (0, max_width), (max_height, 0), (max_height, max_width)]
    
    for img, (y, x) in zip(images, positions):
        h, w = img.shape
        collage[y:y+h, x:x+w] = img
    
    return collage
