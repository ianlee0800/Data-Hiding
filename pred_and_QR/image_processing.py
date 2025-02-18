import numpy as np
import cv2
import matplotlib.pyplot as plt
from common import calculate_psnr, calculate_ssim, histogram_correlation
import cupy as cp
from numba import cuda
from enum import Enum

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

def split_image_flexible(img, split_size, block_base=False, quad_tree=False, positions=None):
    """
    將圖像切割成區塊
    
    Parameters:
    -----------
    img : numpy.ndarray or cupy.ndarray
        輸入圖像
    split_size : int
        每個維度要切割的數量（例如：4 表示切成 4x4=16 塊）
        對於quad_tree模式，這個參數代表當前要切割的區塊大小
    block_base : bool
        True: 使用 block-based 分割
        False: 使用 quarter-based 分割
    quad_tree : bool
        True: 使用quad tree分割模式
        False: 使用原有的分割模式
    positions : list of tuple, optional
        只在quad_tree=True時使用
        記錄每個區塊在原圖中的位置 [(y, x), ...]
    
    Returns:
    --------
    list
        包含所有切割後區塊的列表
        如果是quad_tree模式，還會返回positions列表
    """
    if isinstance(img, cp.ndarray):
        xp = cp
    else:
        xp = np

    height, width = img.shape
    
    # quad tree模式
    if quad_tree:
        if positions is None:
            positions = []
            
        sub_images = []
        current_positions = []
        block_size = split_size  # 在quad tree模式下，split_size代表區塊大小
        
        # 檢查輸入的區塊大小是否符合要求
        if height < block_size or width < block_size:
            raise ValueError(f"Image size ({height}x{width}) is smaller than block size ({block_size})")
        
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # 確保不會超出圖像邊界
                actual_block_size = min(block_size, height - y, width - x)
                if actual_block_size == block_size:  # 只處理完整的區塊
                    sub_img = img[y:y+block_size, x:x+block_size]
                    sub_images.append(xp.asarray(sub_img))
                    current_positions.append((y, x))
        
        if positions is not None:
            positions.extend(current_positions)
        
        return sub_images, current_positions
    
    # 原有的分割模式
    else:
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
    將切割後的區塊合併回完整圖像，支援不同的輸入尺寸
    
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
    cupy.ndarray
        合併後的完整圖像
    """
    if not sub_images:
        raise ValueError("No sub-images to merge")
    
    # 確保所有子圖像都是 CuPy 數組
    sub_images = [cp.asarray(img) for img in sub_images]
    
    # 從第一個子圖像獲取尺寸資訊
    sub_height, sub_width = sub_images[0].shape
    total_size = sub_height * split_size  # 計算完整圖像的尺寸
    
    # 創建對應尺寸的輸出圖像
    merged = cp.zeros((total_size, total_size), dtype=sub_images[0].dtype)
    
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

class PredictionMethod(Enum):
    PROPOSED = "proposed"  # 您現有的weighted prediction方法
    MED = "med"           # Median Edge Detection
    GAP = "gap"           # Gradient Adjusted Prediction
    
@cuda.jit
def med_predict_kernel(img, pred_img, height, width):
    """
    MED預測的CUDA kernel實現
    """
    x, y = cuda.grid(2)
    if 1 < x < width - 1 and 1 < y < height - 1:
        a = int(img[y, x-1])    # left
        b = int(img[y-1, x])    # up
        c = int(img[y-1, x-1])  # up-left
        
        # MED 預測邏輯
        if c >= max(a, b):
            pred = min(a, b)
        elif c <= min(a, b):
            pred = max(a, b)
        else:
            pred = a + b - c
            
        pred_img[y, x] = min(255, max(0, pred))
    else:
        pred_img[y, x] = img[y, x]

@cuda.jit
def gap_predict_kernel(img, pred_img, height, width):
    """
    GAP預測的CUDA kernel實現
    """
    x, y = cuda.grid(2)
    if 1 < x < width - 1 and 1 < y < height - 1:
        a = int(img[y, x-1])     # left
        b = int(img[y-1, x])     # up
        c = int(img[y-1, x-1])   # up-left
        d = int(img[y-1, x+1])   # up-right
        
        # 計算水平和垂直方向的差異
        dh = abs(a - c) + abs(b - d)
        dv = abs(a - c) + abs(b - c)
        
        # GAP 預測邏輯
        if dv - dh > 80:
            pred = a  # 水平邊緣
        elif dh - dv > 80:
            pred = b  # 垂直邊緣
        else:
            pred = (a + b) // 2  # 平滑區域
            
        pred_img[y, x] = min(255, max(0, pred))
    else:
        pred_img[y, x] = img[y, x]

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

def predict_image_cuda(img, prediction_method, weights=None):
    """
    統一的預測函數接口
    
    Parameters:
    -----------
    img : numpy.ndarray or cupy.ndarray
        輸入圖像
    prediction_method : PredictionMethod
        預測方法
    weights : numpy.ndarray, optional
        只在使用PROPOSED方法時需要的權重參數
        
    Returns:
    --------
    numpy.ndarray
        預測後的圖像
    """
    height, width = img.shape
    d_img = cuda.to_device(img)
    d_pred_img = cuda.device_array_like(img)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    if prediction_method == PredictionMethod.PROPOSED:
        if weights is None:
            raise ValueError("Weights must be provided for PROPOSED method")
        d_weights = cuda.to_device(weights)
        improved_predict_kernel[blocks_per_grid, threads_per_block](
            d_img, d_weights, d_pred_img, height, width
        )
    elif prediction_method == PredictionMethod.MED:
        med_predict_kernel[blocks_per_grid, threads_per_block](
            d_img, d_pred_img, height, width
        )
    elif prediction_method == PredictionMethod.GAP:
        gap_predict_kernel[blocks_per_grid, threads_per_block](
            d_img, d_pred_img, height, width
        )

    return d_pred_img.copy_to_host()

def add_grid_lines(img, block_info):
    """
    為 quadtree 分割結果添加格線
    
    Parameters:
    -----------
    img : numpy.ndarray
        原始圖像
    block_info : dict
        包含各區塊資訊的字典，格式如：
        {
            '256': {'blocks': [{'position': (y,x), 'size': 256}, ...]},
            '128': {'blocks': [...]},
            '64': {'blocks': [...]},
            '32': {'blocks': [...]},
            '16': {'blocks': [...]}
        }
    
    Returns:
    --------
    numpy.ndarray
        添加格線後的圖像
    """
    grid_img = img.copy()
    grid_color = 128  # 使用中灰色作為格線顏色
    
    # 對不同大小的區塊使用不同的格線寬度，注意使用整數作為鍵值
    line_widths = {
        512: 3,
        256: 3,
        128: 2,
        64: 2,
        32: 1,
        16: 1
    }
    
    # 從大到小處理各個區塊
    for size_str in sorted(block_info.keys(), key=lambda x: int(x), reverse=True):
        size = int(size_str)  # 將字符串轉換為整數
        line_width = line_widths[size]
        blocks = block_info[size_str]['blocks']
        
        for block in blocks:
            y, x = block['position']
            block_size = block['size']
            
            # 繪製水平線
            for i in range(line_width):
                grid_img[y+i:y+i+1, x:x+block_size] = grid_color  # 上邊界
                grid_img[y+block_size-i-1:y+block_size-i, x:x+block_size] = grid_color  # 下邊界
            
            # 繪製垂直線
            for i in range(line_width):
                grid_img[y:y+block_size, x+i:x+i+1] = grid_color  # 左邊界
                grid_img[y:y+block_size, x+block_size-i-1:x+block_size-i] = grid_color  # 右邊界
    
    return grid_img