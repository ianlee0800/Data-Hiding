import numpy as np
import cupy as cp
import cv2
import os
from scipy.signal import savgol_filter
from prettytable import PrettyTable

# =============================================================================
# 常數和配置類
# =============================================================================

class MeasurementConstants:
    """測量相關的常數定義"""
    # 默認測量參數
    DEFAULT_SEGMENTS = 20
    DEFAULT_STEP_SIZE = 100000
    DEFAULT_TOTAL_EMBEDDINGS = 5
    DEFAULT_EL_MODE = 0
    
    # 數據處理參數
    DEFAULT_CORRECTION_STRENGTH = 0.5
    COLOR_CORRECTION_STRENGTH = 0.3
    MIN_SMOOTHING_POINTS = 7
    
    # Savitzky-Golay 平滑參數
    SAVGOL_WINDOW_LENGTH = 7
    SAVGOL_POLY_ORDER = 2
    SAVGOL_ORIGINAL_WEIGHT = 0.7  # 原始數據權重
    SAVGOL_SMOOTHED_WEIGHT = 0.3  # 平滑數據權重

class PathConstants:
    """路徑相關的常數定義"""
    # 基礎路徑
    BASE_OUTPUT_DIR = "./Prediction_Error_Embedding/outcome"
    BASE_IMAGE_DIR = "./Prediction_Error_Embedding/image"
    ALTERNATIVE_IMAGE_DIR = "./pred_and_QR/image"

class QuadtreeConstants:
    """Quadtree 方法專用常數"""
    DEFAULT_MIN_BLOCK_SIZE = 16
    DEFAULT_VARIANCE_THRESHOLD = 300
    DEFAULT_ADAPTIVE_THRESHOLD = False
    DEFAULT_SEARCH_MODE = 'balanced'
    DEFAULT_TARGET_BPP = 0.8
    DEFAULT_TARGET_PSNR = 35.0

class DataType:
    """數據類型常數（從 common.py 移入）"""
    INT8 = np.int8
    UINT8 = np.uint8
    INT32 = np.int32
    FLOAT32 = np.float32

# =============================================================================
# 數據轉換工具類（整合自 common.py）
# =============================================================================

class DataConverter:
    """統一的數據轉換工具類"""
    
    @staticmethod
    def to_numpy(data):
        """轉換為 NumPy 陣列"""
        if isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return np.asarray(data)
    
    @staticmethod
    def to_cupy(data):
        """轉換為 CuPy 陣列"""
        if isinstance(data, np.ndarray):
            return cp.asarray(data)
        return cp.asarray(DataConverter.to_numpy(data))
    
    @staticmethod
    def ensure_type(data, dtype):
        """確保數據為指定類型"""
        if isinstance(data, cp.ndarray):
            return data.astype(dtype)
        return np.asarray(data, dtype=dtype)
    
    @staticmethod
    def to_gpu(data):
        """將數據移至 GPU"""
        return cp.asarray(data)
    
    @staticmethod
    def to_cpu(data):
        """將數據移至 CPU"""
        if isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return np.asarray(data)
    
    @staticmethod
    def prepare_image(img):
        """預處理圖像數據"""
        if isinstance(img, cp.ndarray):
            img = cp.asnumpy(img)
        return np.array(img, dtype=np.uint8)

# =============================================================================
# 測量點生成器（新增）
# =============================================================================

class MeasurementPointGenerator:
    """測量點生成器類"""
    
    @staticmethod
    def generate_points(max_payload, step_size=None, segments=None):
        """
        生成測量點序列
        
        Parameters:
        -----------
        max_payload : int
            最大載荷容量
        step_size : int, optional
            步長模式的步長（位元）
        segments : int, optional
            分段模式的分段數
            
        Returns:
        --------
        tuple : (payload_points, measurement_mode, mode_description)
        """
        if step_size is not None and step_size > 0:
            # 步長模式
            payload_points = list(range(step_size, max_payload, step_size))
            measurement_mode = "step_size"
            mode_description = f"Step size: {step_size} bits"
        else:
            # 分段模式
            if segments is None:
                segments = MeasurementConstants.DEFAULT_SEGMENTS
            
            segments = max(2, min(segments, max_payload // 1000))  # 確保合理的分段數
            
            interval = max_payload / segments
            payload_points = [int(i * interval) for i in range(1, segments)]
            measurement_mode = "segments"
            mode_description = f"Segments: {segments}"
        
        return payload_points, measurement_mode, mode_description

# =============================================================================
# 品質指標計算（新增，整合自 common.py）
# =============================================================================

def calculate_psnr(img1, img2):
    """計算 PSNR"""
    img1 = DataConverter.to_numpy(img1)
    img2 = DataConverter.to_numpy(img2)
    
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    """計算 SSIM"""
    img1 = DataConverter.to_numpy(img1)
    img2 = DataConverter.to_numpy(img2)

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

def histogram_correlation(hist1, hist2):
    """計算直方圖相關性"""
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

def calculate_quality_metrics_unified(original_img, embedded_img, img_info):
    """
    統一的品質指標計算函數
    
    Parameters:
    -----------
    original_img : numpy.ndarray
        原始圖像
    embedded_img : numpy.ndarray
        嵌入後圖像
    img_info : dict
        圖像資訊字典
        
    Returns:
    --------
    tuple : (psnr, ssim, hist_corr)
    """
    # 確保數據格式
    original_img = DataConverter.to_numpy(original_img)
    embedded_img = DataConverter.to_numpy(embedded_img)
    
    if img_info.get('is_color', False):
        # 彩色圖像處理
        return _calculate_color_metrics(original_img, embedded_img)
    else:
        # 灰階圖像處理
        psnr = calculate_psnr(original_img, embedded_img)
        ssim = calculate_ssim(original_img, embedded_img)
        
        # 計算直方圖相關性
        hist1 = np.histogram(original_img, bins=256, range=(0, 255))[0]
        hist2 = np.histogram(embedded_img, bins=256, range=(0, 255))[0]
        hist_corr = histogram_correlation(hist1, hist2)
        
        return psnr, ssim, hist_corr

def _calculate_color_metrics(original_img, embedded_img):
    """計算彩色圖像的品質指標"""
    # 整體PSNR
    psnr = calculate_psnr(original_img, embedded_img)
    
    # 分通道計算SSIM和直方圖相關性
    b1, g1, r1 = cv2.split(original_img)
    b2, g2, r2 = cv2.split(embedded_img)
    
    ssim_values = [calculate_ssim(c1, c2) for c1, c2 in [(b1, b2), (g1, g2), (r1, r2)]]
    ssim = np.mean(ssim_values)
    
    hist_corr_values = []
    for c1, c2 in [(b1, b2), (g1, g2), (r1, r2)]:
        h1 = np.histogram(c1, bins=256, range=(0, 255))[0]
        h2 = np.histogram(c2, bins=256, range=(0, 255))[0]
        hist_corr_values.append(histogram_correlation(h1, h2))
    hist_corr = np.mean(hist_corr_values)
    
    return psnr, ssim, hist_corr

# =============================================================================
# 圖像資訊獲取（供 measurement.py 使用，但建議後續移至 image_processing.py）
# =============================================================================

def get_image_info(img):
    """
    獲取圖像的基本資訊
    
    注意：此函數建議後續移至 image_processing.py
    
    Parameters:
    -----------
    img : numpy.ndarray
        輸入圖像
        
    Returns:
    --------
    dict : 圖像資訊字典
    """
    img = DataConverter.to_numpy(img)
    
    is_color = len(img.shape) == 3 and img.shape[2] == 3
    
    if is_color:
        height, width, channels = img.shape
        pixel_count = height * width  # 像素位置數，不包含通道維度
        type_name = "color"
        description = f"Color image: {height}x{width}x{channels}"
    else:
        height, width = img.shape
        pixel_count = height * width
        type_name = "grayscale"
        description = f"Grayscale image: {height}x{width}"
    
    return {
        'height': height,
        'width': width,
        'pixel_count': pixel_count,
        'is_color': is_color,
        'type_name': type_name,
        'description': description
    }

# =============================================================================
# 路徑管理工具函數
# =============================================================================

def get_output_directories(img_name, method, base_dir=None):
    """統一獲取所有輸出目錄路徑"""
    if base_dir is None:
        base_dir = PathConstants.BASE_OUTPUT_DIR
    
    # 基本目錄
    directories = {
        'base': base_dir,
        'image': f"{base_dir}/image/{img_name}/{method}",
        'histogram': f"{base_dir}/histogram/{img_name}/{method}",
        'plots': f"{base_dir}/plots/{img_name}/{method}",
        'data': f"{base_dir}/data/{img_name}"
    }
    
    # 為特定方法添加專門的子目錄
    if method == "rotation":
        directories.update({
            'rotated': f"{directories['image']}/rotated",
            'subimages': f"{directories['image']}/subimages",
            'diff_histograms': f"{directories['histogram']}/difference_histograms"
        })
    elif method == "split":
        directories.update({
            'split_viz': f"{directories['image']}/split_visualization",
            'diff_histograms': f"{directories['histogram']}/difference_histograms"
        })
    elif method == "quadtree":
        directories.update({
            'quadtree_viz': f"{directories['image']}/quadtree_visualization",
            'with_grid': f"{directories['image']}/with_grid",
            'block_viz': f"{directories['image']}/block_visualizations",
            'heatmaps': f"{directories['image']}/heatmaps",
            'block_distribution': f"{directories['plots']}/block_distribution",
            'diff_histograms': f"{directories['histogram']}/difference_histograms",
            'block_histograms': f"{directories['histogram']}/block_histograms",
            'channels': f"{directories['image']}/channels"
        })
    
    return directories

def create_all_directories(directories):
    """創建所有需要的目錄"""
    for dir_path in directories.values():
        ensure_dir(f"{dir_path}/dummy.txt")

def get_precise_measurement_directories(img_name, method_name):
    """專門為精確測量獲取目錄結構"""
    base_dir = PathConstants.BASE_OUTPUT_DIR
    return {
        'result': f"{base_dir}/plots/{img_name}/precise_{method_name.lower()}",
        'comparison': f"{base_dir}/plots/{img_name}/precise_comparison",
        'data': f"{base_dir}/plots/{img_name}/data_{method_name.lower()}"
    }

def find_image_path(img_name, filetype="png"):
    """尋找圖像文件路徑"""
    # 嘗試主要路徑
    primary_path = f"{PathConstants.BASE_IMAGE_DIR}/{img_name}.{filetype}"
    if os.path.exists(primary_path):
        return primary_path
    
    # 嘗試備用路徑
    alternative_path = f"{PathConstants.ALTERNATIVE_IMAGE_DIR}/{img_name}.{filetype}"
    if os.path.exists(alternative_path):
        return alternative_path
    
    # 都找不到則拋出錯誤
    raise FileNotFoundError(f"Cannot find image: {img_name}.{filetype}")

# =============================================================================
# 基本工具函數
# =============================================================================

def generate_random_binary_array(size, ratio_of_ones=0.5):
    """生成指定大小的随机二进制数组，可调整1的比例"""
    return np.random.choice([0, 1], size=size, p=[1-ratio_of_ones, ratio_of_ones])

def ensure_dir(file_path):
    """確保目錄存在，如果不存在則創建"""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# =============================================================================
# PEE 資訊表格相關功能
# =============================================================================

def create_pee_info_table(pee_stages, use_different_weights, total_pixels, 
                         split_size, quad_tree=False):
    """創建 PEE 資訊表格的完整函數 - 支援彩色圖像"""    
    table = PrettyTable()
    
    # 檢測是否為彩色圖像
    is_color_image = False
    if pee_stages and 'channel_payloads' in pee_stages[0]:
        is_color_image = True
    elif pee_stages and 'block_info' in pee_stages[0]:
        if isinstance(pee_stages[0]['block_info'], dict):
            for size_str, size_info in pee_stages[0]['block_info'].items():
                if isinstance(size_info, dict) and 'blocks' in size_info:
                    for block in size_info['blocks']:
                        if 'channel' in block:
                            is_color_image = True
                            break
                    if is_color_image:
                        break
    
    if is_color_image:
        # 彩色圖像的表格欄位
        table.field_names = [
            "Embedding", "Total Payload", "BPP", "PSNR", "SSIM", "Hist Corr",
            "Blue Payload", "Green Payload", "Red Payload", "Block Counts", "Note"
        ]
        
        for stage in pee_stages:
            table.add_row([
                stage['embedding'],
                stage['payload'],
                f"{stage['bpp']:.4f}",
                f"{stage['psnr']:.2f}",
                f"{stage['ssim']:.4f}",
                f"{stage['hist_corr']:.4f}",
                stage['channel_payloads']['blue'],
                stage['channel_payloads']['green'], 
                stage['channel_payloads']['red'],
                sum(stage.get('block_counts', {}).values()) if 'block_counts' in stage else '-',
                "Color Image"
            ])
            table.add_row(["-" * 5] * len(table.field_names))
    
    elif quad_tree:
        # Quad tree 模式的表格欄位
        table.field_names = [
            "Embedding", "Block Size", "Block Position", "Payload", "BPP",
            "PSNR", "SSIM", "Hist Corr", "Weights", "EL", "Note"
        ]
        
        for stage in pee_stages:
            # 添加整體 stage 資訊
            table.add_row([
                f"{stage['embedding']} (Overall)",
                "-", "-", stage['payload'],
                f"{stage['bpp']:.4f}",
                f"{stage['psnr']:.2f}",
                f"{stage['ssim']:.4f}",
                f"{stage['hist_corr']:.4f}",
                "-", "-", "Stage Summary"
            ])
            table.add_row(["-" * 5] * len(table.field_names))
            
            # 處理區塊資訊
            for size_str in sorted(stage['block_info'].keys(), key=int, reverse=True):
                blocks = stage['block_info'][size_str]['blocks']
                for block in blocks:
                    block_pixels = block['size'] * block['size']
                    
                    weights_display = (
                        "N/A" if block['weights'] == "N/A"
                        else ", ".join([f"{w:.2f}" for w in block['weights']]) if block['weights']
                        else "-"
                    )
                    
                    table.add_row([
                        stage['embedding'],
                        f"{block['size']}x{block['size']}",
                        f"({block['position'][0]}, {block['position'][1]})",
                        block['payload'],
                        f"{block['payload'] / block_pixels:.4f}",
                        f"{block['psnr']:.2f}",
                        f"{block['ssim']:.4f}",
                        f"{block['hist_corr']:.4f}",
                        weights_display,
                        block.get('EL', '-'),
                        "Different weights" if use_different_weights else ""
                    ])
            table.add_row(["-" * 5] * len(table.field_names))
            
    else:
        # 標準模式的表格欄位
        table.field_names = [
            "Embedding", "Sub-image", "Payload", "BPP", "PSNR", "SSIM",
            "Hist Corr", "Weights", "EL", "Rotation", "Note"
        ]
        
        for stage in pee_stages:
            # 添加整體 stage 資訊
            table.add_row([
                f"{stage['embedding']} (Overall)",
                "-", stage['payload'],
                f"{stage['bpp']:.4f}",
                f"{stage['psnr']:.2f}",
                f"{stage['ssim']:.4f}",
                f"{stage['hist_corr']:.4f}",
                "-", "-", "-", "Stage Summary"
            ])
            table.add_row(["-" * 5] * len(table.field_names))
            
            # 處理標準模式的區塊資訊
            total_blocks = split_size * split_size
            sub_image_pixels = total_pixels // total_blocks
            
            for i, block in enumerate(stage['block_params']):
                weights_display = "-"
                if 'weights' in block:
                    if block['weights'] == "N/A":
                        weights_display = "N/A"
                    elif block['weights']:
                        try:
                            weights_display = ", ".join([f"{w:.2f}" for w in block['weights']])
                        except:
                            weights_display = str(block['weights'])
                
                payload = block.get('payload', 0)
                psnr = block.get('psnr', 0)
                ssim = block.get('ssim', 0)
                hist_corr = block.get('hist_corr', 0)
                el = block.get('EL', '-')
                rotation = block.get('rotation', 0)
                
                table.add_row([
                    stage['embedding'] if i == 0 else "",
                    i, payload,
                    f"{payload / sub_image_pixels:.4f}",
                    f"{psnr:.2f}",
                    f"{ssim:.4f}",
                    f"{hist_corr:.4f}",
                    weights_display,
                    el,
                    f"{rotation}°",
                    "Different weights" if use_different_weights else ""
                ])
        
        table.add_row(["-" * 5] * len(table.field_names))
    
    return table

# =============================================================================
# 嵌入數據生成相關函數
# =============================================================================

def generate_embedding_data(total_embeddings, sub_images_per_stage, max_capacity_per_subimage, 
                           ratio_of_ones=0.5, target_payload_size=-1):
    """生成嵌入數據"""
    # 如果沒有指定目標payload，使用最大容量模式
    if target_payload_size <= 0:
        max_stage_payload = sub_images_per_stage * max_capacity_per_subimage
        stage_data = []
        for _ in range(total_embeddings):
            sub_data_list = []
            for _ in range(sub_images_per_stage):
                sub_data = generate_random_binary_array(max_capacity_per_subimage, ratio_of_ones)
                sub_data_list.append(sub_data)
            stage_data.append({
                'sub_data': sub_data_list,
                'remaining_target': 0
            })
        return {
            'stage_data': stage_data,
            'total_target': max_stage_payload * total_embeddings
        }
    
    # 使用指定的payload size
    total_remaining = target_payload_size
    stage_data = []
    
    # 為每個stage分配潛在的最大容量
    potential_capacity_per_stage = max_capacity_per_subimage * sub_images_per_stage
    
    for stage in range(total_embeddings):
        sub_data_list = []
        
        for sub_img in range(sub_images_per_stage):
            # 計算這個子圖像可能需要的最大數據量
            max_possible = min(max_capacity_per_subimage, total_remaining)
            
            # 如果是最後一個stage的最後一個子圖像，確保生成足夠的數據
            if stage == total_embeddings - 1 and sub_img == sub_images_per_stage - 1:
                sub_data = generate_random_binary_array(total_remaining, ratio_of_ones)
            else:
                sub_data = generate_random_binary_array(max_possible, ratio_of_ones)
            
            sub_data_list.append(sub_data)
        
        stage_data.append({
            'sub_data': sub_data_list,
            'remaining_target': total_remaining
        })
    
    return {
        'stage_data': stage_data,
        'total_target': target_payload_size
    }

# =============================================================================
# 記憶體管理工具（從 common.py 移入）
# =============================================================================

def cleanup_memory():
    """清理 GPU 記憶體資源，避免記憶體洩漏"""
    try:
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
    """檢查系統記憶體和 GPU 記憶體使用情況"""
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
    """打印當前記憶體使用情況"""
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

# =============================================================================
# 向後兼容接口（暫時保留）
# =============================================================================

# 為了確保現有代碼不中斷，提供一些別名
to_numpy = DataConverter.to_numpy
to_cupy = DataConverter.to_cupy
ensure_type = DataConverter.ensure_type
to_gpu = DataConverter.to_gpu
to_cpu = DataConverter.to_cpu