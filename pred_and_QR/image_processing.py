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

def split_image_flexible(img, split_size, block_base=False):
    """
    將圖像切割成 split_size x split_size 個區塊
    
    Parameters:
    -----------
    img : numpy.ndarray or cupy.ndarray
        輸入圖像 (512x512)
    split_size : int
        每個維度要切割的數量（例如：4 表示切成 4x4=16 塊）
    block_base : bool
        True: 使用 block-based 分割
        False: 使用 quarter-based 分割
    
    Returns:
    --------
    list
        包含所有切割後區塊的列表
    """
    if isinstance(img, cp.ndarray):
        xp = cp
    else:
        xp = np

    height, width = img.shape
    sub_height = height // split_size
    sub_width = width // split_size
    
    sub_images = []
    if block_base:
        # Block-based splitting
        for i in range(split_size):
            for j in range(split_size):
                sub_img = img[i*sub_height:(i+1)*sub_height, 
                             j*sub_width:(j+1)*sub_width]
                sub_images.append(xp.asarray(sub_img))
    else:
        # Quarter-based splitting (交錯式分割)
        for i in range(split_size):
            for j in range(split_size):
                sub_img = img[i::split_size, j::split_size]
                sub_images.append(xp.asarray(sub_img))
    
    return sub_images

def merge_image_flexible(sub_images, split_size, block_base=False):
    """
    將切割後的區塊合併回完整圖像
    
    Parameters:
    -----------
    sub_images : list
        包含所有切割後區塊的列表
    split_size : int
        原始切割時的尺寸
    block_base : bool
        True: 使用 block-based 合併
        False: 使用 quarter-based 合併
    
    Returns:
    --------
    numpy.ndarray or cupy.ndarray
        合併後的完整圖像
    """
    if not sub_images:
        raise ValueError("No sub-images to merge")
    
    # 確保所有子圖像都是 CuPy 數組
    sub_images = [cp.asarray(img) for img in sub_images]
    
    sub_height = 512 // split_size
    sub_width = 512 // split_size
    
    merged = cp.zeros((512, 512), dtype=sub_images[0].dtype)
    
    if block_base:
        # Block-based merging
        for idx, sub_img in enumerate(sub_images):
            i = idx // split_size
            j = idx % split_size
            merged[i*sub_height:(i+1)*sub_height, 
                   j*sub_width:(j+1)*sub_width] = sub_img
    else:
        # Quarter-based merging (交錯式合併)
        for idx, sub_img in enumerate(sub_images):
            i = idx // split_size
            j = idx % split_size
            merged[i::split_size, j::split_size] = sub_img
    
    return merged

def verify_image_dimensions(img, split_size):
    """
    檢查圖像尺寸是否適合進行切割，並返回建議的新尺寸
    
    Parameters:
    -----------
    img : numpy.ndarray or cupy.ndarray
        輸入圖像
    split_size : int
        每個維度要切割的數量
    
    Returns:
    --------
    tuple
        建議的新圖像尺寸 (height, width)
    """
    height, width = img.shape
    
    # 檢查是否需要調整尺寸
    new_height = ((height + split_size - 1) // split_size) * split_size
    new_width = ((width + split_size - 1) // split_size) * split_size
    
    if new_height != height or new_width != width:
        return (new_height, new_width)
    return (height, width)

def create_collage(images):
    """Create a collage from multiple images."""
    # 確保輸入的圖像數量是完全平方數
    n = len(images)
    grid_size = int(np.ceil(np.sqrt(n)))
    
    # 確保所有圖像都是numpy數組
    images = [np.array(img) if not isinstance(img, np.ndarray) else img for img in images]
    
    # 找出最大維度
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    
    # 創建拼貼圖
    collage = np.zeros((max_height * grid_size, max_width * grid_size), dtype=np.uint8)
    
    # 填充圖像
    for idx, img in enumerate(images):
        i = idx // grid_size
        j = idx % grid_size
        y = i * max_height
        x = j * max_width
        h, w = img.shape
        collage[y:y+h, x:x+w] = img
    
    return collage

@cuda.jit
def improved_predict_kernel(img, weights, pred_img, height, width):
    x, y = cuda.grid(2)
    if 1 < x < width - 1 and 1 < y < height - 1:
        up = int(img[y-1, x])
        left = int(img[y, x-1])
        ul = int(img[y-1, x-1])
        ur = int(img[y-1, x+1])
        
        # Context-based prediction
        if abs(up - ul) < abs(left - ul):
            base_pred = left
        else:
            base_pred = up
        
        # Weighted prediction
        weighted_pred = (weights[0]*up + weights[1]*left + weights[2]*ul + weights[3]*ur)
        total_weight = weights[0] + weights[1] + weights[2] + weights[3]
        
        # Combine predictions
        final_pred = (base_pred + int(weighted_pred / total_weight)) // 2
        
        pred_img[y, x] = min(255, max(0, final_pred))
    else:
        pred_img[y, x] = img[y, x]

def improved_predict_image_cuda(img, weights):
    height, width = img.shape
    d_img = cuda.to_device(img)
    d_weights = cuda.to_device(weights)
    d_pred_img = cuda.device_array_like(img)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    improved_predict_kernel[blocks_per_grid, threads_per_block](d_img, d_weights, d_pred_img, height, width)

    return d_pred_img