import numpy as np
import cupy as cp
import cv2
from numba import cuda

class DataType:
    INT8 = np.int8
    UINT8 = np.uint8
    INT32 = np.int32
    FLOAT32 = np.float32

def to_numpy(data):
    if isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    return np.asarray(data)

def to_cupy(data):
    if isinstance(data, np.ndarray):
        return cp.asarray(data)
    return cp.asarray(to_numpy(data))

def ensure_type(data, dtype):
    if isinstance(data, cp.ndarray):
        return data.astype(dtype)
    return np.asarray(data, dtype=dtype)

def to_gpu(data):
    return cp.asarray(data)

def to_cpu(data):
    return cp.asnumpy(data)

def find_max(hist):
    """找到直方圖中的峰值"""
    if isinstance(hist, tuple):
        print("Warning: hist is a tuple. Using the first element.")
        hist = hist[0]
    
    if not isinstance(hist, np.ndarray):
        print(f"Warning: hist is not a numpy array. Type: {type(hist)}")
        hist = np.array(hist)
    
    if hist.ndim > 1:
        print(f"Warning: hist has {hist.ndim} dimensions. Flattening.")
        hist = hist.flatten()
    
    # 避免選擇 255 作為峰值
    peak = np.argmax(hist[:-1])
    return peak

def histogram_correlation(hist1, hist2):
    hist1 = hist1.astype(np.float64)
    hist2 = hist2.astype(np.float64)
    
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    mean1, mean2 = np.mean(hist1), np.mean(hist2)
    
    numerator = np.sum((hist1 - mean1) * (hist2 - mean2))
    denominator = np.sqrt(np.sum((hist1 - mean1)**2) * np.sum((hist2 - mean2)**2))
    
    if denominator == 0:
        return 1.0
    else:
        return numerator / denominator

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # 如果 MSE 为 0，返回无穷大
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    img1 = to_numpy(img1)
    img2 = to_numpy(img2)

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return round(ssim_map.mean(), 4)

def calculate_metrics_with_rotation(current_img, stage_img, original_img, embedding):
    """
    計算考慮旋轉的 metrics，優化版本支援大型圖像
    """
    # 計算旋轉角度
    current_rotation = (embedding * 90) % 360
    
    # 如果當前階段有旋轉，先將圖像旋轉回原始方向
    if current_rotation != 0:
        # 計算需要的逆旋轉次數
        k = (4 - (current_rotation // 90)) % 4
        if isinstance(stage_img, cp.ndarray):
            # 對於大型圖像，分塊處理旋轉以節省記憶體
            stage_img_aligned = cp.zeros_like(stage_img)
            block_size = 512  # 可以根據可用記憶體調整此值
            
            for i in range(0, stage_img.shape[0], block_size):
                for j in range(0, stage_img.shape[1], block_size):
                    # 確保不超出邊界
                    end_i = min(i + block_size, stage_img.shape[0])
                    end_j = min(j + block_size, stage_img.shape[1])
                    
                    # 提取區塊並旋轉
                    block = stage_img[i:end_i, j:end_j]
                    rotated_block = cp.rot90(block, k=k)
                    
                    # 計算旋轉後的放置位置
                    if k % 2 == 0:  # 0 或 180 度
                        stage_img_aligned[i:end_i, j:end_j] = rotated_block
                    else:  # 90 或 270 度
                        # 對於 90/270 度旋轉，長寬交換，需要特殊處理
                        # 這個簡化版本僅適用於正方形區塊
                        if block.shape[0] == block.shape[1]:
                            stage_img_aligned[i:end_i, j:end_j] = rotated_block
                        else:
                            # 對於非正方形區塊，使用整體旋轉
                            cleanup_memory()  # 先清理記憶體
                            stage_img_aligned = cp.rot90(stage_img, k=k)
                            break
        else:
            stage_img_aligned = np.rot90(stage_img, k=k)
    else:
        stage_img_aligned = stage_img
    
    # 確保數據類型一致
    if isinstance(stage_img_aligned, cp.ndarray):
        stage_img_aligned = cp.asnumpy(stage_img_aligned)
    if isinstance(original_img, cp.ndarray):
        original_img = cp.asnumpy(original_img)
        
    # 計算指標
    psnr = calculate_psnr(original_img, stage_img_aligned)
    ssim = calculate_ssim(original_img, stage_img_aligned)
    hist_corr = histogram_correlation(
        np.histogram(original_img, bins=256, range=(0, 255))[0],
        np.histogram(stage_img_aligned, bins=256, range=(0, 255))[0]
    )
    
    return psnr, ssim, hist_corr

def calculate_correlation(img1, img2):
    """計算兩個圖像的相關係數"""
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    correlation = np.corrcoef(img1_flat, img2_flat)[0, 1]
    return round(correlation, 4)

def improved_predict_image_cpu(img, weight):
    height, width = img.shape
    pred_img = np.zeros_like(img)
    
    for y in range(1, height-1):
        for x in range(1, width-1):
            ul = int(img[y-1,x-1])
            up = int(img[y-1,x])
            ur = int(img[y-1,x+1])
            left = int(img[y,x-1])
            p = (weight[0]*up+weight[1]*ul+weight[2]*ur+weight[3]*left)/np.sum(weight)
            pred_img[y,x] = round(p)
    
    return pred_img

@cuda.jit
def compute_improved_adaptive_el_kernel(img, local_el, window_size, max_el, block_size):
    """
    計算改進的自適應EL值的CUDA kernel，優化版本
    修改：EL值範圍改為1~5，不再限制為奇數
    """
    x, y = cuda.grid(2)
    if x < img.shape[1] and y < img.shape[0]:
        # 根據區塊大小調整window_size
        actual_window_size = window_size
        if block_size > 0:  # 使用正數來判斷是否有指定block_size
            if block_size >= 512:
                actual_window_size = min(window_size + 3, 9)  # 更大的視窗適合1024大小的圖像
            elif block_size >= 256:
                actual_window_size = min(window_size + 2, 7)
            elif block_size <= 64:
                actual_window_size = max(window_size - 1, 3)
        
        half_window = actual_window_size // 2
        
        # 計算局部統計量
        local_sum = 0
        local_sum_sq = 0
        count = 0
        
        for i in range(max(0, y - half_window), min(img.shape[0], y + half_window + 1)):
            for j in range(max(0, x - half_window), min(img.shape[1], x + half_window + 1)):
                pixel_value = img[i, j]
                local_sum += pixel_value
                local_sum_sq += pixel_value * pixel_value
                count += 1
        
        local_mean = local_sum / count
        local_variance = (local_sum_sq / count) - (local_mean * local_mean)
        
        # 修改：調整variance正規化策略，映射到1~5範圍
        max_variance = 6400  # 預設值
        if block_size > 0:
            if block_size >= 512:
                max_variance = 10000  # 針對更大的圖像調整此值
            elif block_size >= 256:
                max_variance = 8100
            elif block_size <= 64:
                max_variance = 4900
        
        # 正規化variance到0~1範圍
        normalized_variance = min(local_variance / max_variance, 1)
        
        # 修改：映射到1~5範圍，不再限制為奇數
        # 使用反比關係：variance越高，EL越小（更保守）
        el_value = int(5 - normalized_variance * 4)  # 5 - (0~4) = 5~1
        
        # 確保EL值在1~5範圍內（移除奇數限制）
        el_value = max(1, min(el_value, 5))
        
        local_el[y, x] = el_value


def compute_improved_adaptive_el(img, window_size=5, max_el=5, block_size=None):
    """
    計算改進的自適應EL值
    修改：max_el預設改為5，支援1~5範圍
    """
    if isinstance(img, cp.ndarray):
        img = cp.asnumpy(img)
    
    local_el = cuda.device_array(img.shape, dtype=cp.int32)
    
    threads_per_block = (16, 16)
    blocks_per_grid_x = (img.shape[1] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (img.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # 將None轉換為-1，這樣在kernel中可以正確處理
    block_size_value = -1 if block_size is None else block_size
    
    compute_improved_adaptive_el_kernel[blocks_per_grid, threads_per_block](
        img, local_el, window_size, max_el, block_size_value
    )
    
    return local_el

@cuda.jit
def calculate_variance_kernel(block, variance_result):
    """
    CUDA kernel for calculating variance of image blocks
    """
    x, y = cuda.grid(2)
    if x < block.shape[1] and y < block.shape[0]:
        # 使用shared memory來優化性能
        tid = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x
        block_size = block.shape[0] * block.shape[1]
        
        # 計算區域平均值
        local_sum = 0
        local_sum_sq = 0
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                pixel_value = block[i, j]
                local_sum += pixel_value
                local_sum_sq += pixel_value * pixel_value
        
        mean = local_sum / block_size
        variance = (local_sum_sq / block_size) - (mean * mean)
        
        # 只需要一個線程寫入結果
        if x == 0 and y == 0:
            variance_result[0] = variance

def calculate_block_variance_cuda(block):
    """
    Calculate variance of a block using CUDA
    """
    threads_per_block = (16, 16)
    blocks_per_grid_x = (block.shape[1] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (block.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    variance_result = cuda.device_array(1, dtype=np.float32)
    
    calculate_variance_kernel[blocks_per_grid, threads_per_block](block, variance_result)
    
    return variance_result[0]

def cleanup_memory():
    """
    清理 GPU 記憶體資源，避免記憶體洩漏
    此函數應該在處理大量資料後和主要處理階段之間呼叫
    """
    try:
        import cupy as cp
        import gc
        
        # 獲取 CuPy 記憶體池
        mem_pool = cp.get_default_memory_pool()
        pinned_pool = cp.get_default_pinned_memory_pool()
        
        # 顯示清理前的記憶體使用情況
        used_bytes = mem_pool.used_bytes()
        total_bytes = mem_pool.total_bytes()
        
        if used_bytes > 0:
            print(f"GPU 記憶體使用前: {used_bytes/1024/1024:.2f}MB / {total_bytes/1024/1024:.2f}MB")
        
        # 釋放所有記憶體區塊
        mem_pool.free_all_blocks()
        pinned_pool.free_all_blocks()
        
        # 強制執行垃圾回收
        gc.collect()
        
        # 確保CUDA內核完成執行
        cp.cuda.Stream.null.synchronize()
        
        # 顯示清理後的記憶體使用情況
        used_bytes_after = mem_pool.used_bytes()
        if used_bytes > 0:
            print(f"GPU 記憶體使用後: {used_bytes_after/1024/1024:.2f}MB / {total_bytes/1024/1024:.2f}MB")
            print(f"已釋放 {(used_bytes - used_bytes_after)/1024/1024:.2f}MB 記憶體")
    except Exception as e:
        print(f"清理 GPU 記憶體時出錯: {str(e)}")
        print("繼續執行程式...")

def check_memory_status():
    """
    檢查系統記憶體和 GPU 記憶體使用情況
    Returns:
        dict: 包含記憶體使用情況的字典
    """
    # 系統記憶體
    try:
        import psutil
        mem = psutil.virtual_memory()
        system_info = {
            'total': mem.total / (1024 ** 3),  # GB
            'available': mem.available / (1024 ** 3),  # GB
            'percent': mem.percent,
            'used': mem.used / (1024 ** 3),  # GB
        }
    except:
        system_info = {'error': 'Cannot get system memory info'}
    
    # GPU 記憶體
    try:
        import cupy as cp
        mem_pool = cp.get_default_memory_pool()
        gpu_info = {
            'used': mem_pool.used_bytes() / (1024 ** 3),  # GB
            'total': mem_pool.total_bytes() / (1024 ** 3)  # GB
        }
    except:
        gpu_info = {'error': 'Cannot get GPU memory info'}
    
    return {
        'system': system_info,
        'gpu': gpu_info
    }
    
def print_memory_status(label=""):
    """
    打印當前記憶體使用情況
    Args:
        label (str): 打印標籤
    """
    mem_status = check_memory_status()
    
    print(f"===== Memory Status {label} =====")
    if 'error' not in mem_status['system']:
        print(f"System Memory: {mem_status['system']['used']:.2f}GB / {mem_status['system']['total']:.2f}GB ({mem_status['system']['percent']}%)")
    else:
        print(f"System Memory: {mem_status['system']['error']}")
    
    if 'error' not in mem_status['gpu']:
        print(f"GPU Memory: {mem_status['gpu']['used']:.2f}GB / {mem_status['gpu']['total']:.2f}GB")
    else:
        print(f"GPU Memory: {mem_status['gpu']['error']}")
    print("===============================")
    
    
