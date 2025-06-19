"""
Integrated PEE Visualization System - 整合的 PEE 視覺化系統

重構目標：
1. 統一三個模組 (visualization.py, color.py, image_processing.py) 的功能
2. 消除重複代碼和邏輯
3. 提供統一的配置管理和錯誤處理
4. 創建清晰的模組間協調機制
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import cupy as cp
import pandas as pd
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Union, Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ================== 配置和枚舉類 ==================

class PredictionMethod(Enum):
    """預測方法枚舉"""
    PROPOSED = "proposed"
    MED = "med"
    GAP = "gap"
    RHOMBUS = "rhombus"

class ImageType(Enum):
    """圖像類型枚舉"""
    GRAYSCALE = "grayscale"
    COLOR = "color"

class VisualizationType(Enum):
    """視覺化類型枚舉"""
    SPLIT = "split"
    QUADTREE = "quadtree"
    ROTATION = "rotation"
    HEATMAP = "heatmap"
    COMPARISON = "comparison"
    METRICS = "metrics"

@dataclass
class SystemConfig:
    """系統配置類"""
    output_dir: str = "./output"
    dpi: int = 300
    figure_size: Tuple[int, int] = (12, 8)
    color_threshold: float = 5.0
    grid_alpha: float = 0.7
    line_width: int = 2
    marker_size: int = 8
    
    # 顏色配置
    block_colors: Dict[int, Tuple[int, int, int]] = None
    method_colors: Dict[str, str] = None
    channel_colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.block_colors is None:
            self.block_colors = {
                1024: (220, 220, 220), 512: (200, 200, 200), 256: (100, 100, 200),
                128: (100, 200, 100), 64: (200, 100, 100), 32: (200, 200, 100), 16: (200, 100, 200)
            }
        
        if self.method_colors is None:
            self.method_colors = {
                'rotation': 'blue', 'split': 'green', 'quadtree': 'red',
                'proposed': 'blue', 'med': 'red', 'gap': 'green', 'rhombus': 'purple'
            }
        
        if self.channel_colors is None:
            self.channel_colors = {'blue': 'blue', 'green': 'green', 'red': 'red'}

# ================== 核心處理引擎 ==================

class ImageProcessor:
    """統一的圖像處理引擎"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
    
    def read_image_auto(self, filepath: str) -> Tuple[np.ndarray, ImageType]:
        """自動讀取圖像並判斷類型"""
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {filepath}")
        
        img_type = self._detect_image_type(img)
        
        if img_type == ImageType.GRAYSCALE:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(f"Detected grayscale image: {filepath}")
        else:
            print(f"Detected color image: {filepath}")
        
        return img, img_type
    
    def save_image(self, img, path):
        """
        保存圖像文件
        
        Parameters:
        -----------
        img : numpy.ndarray
            要保存的圖像
        path : str
            保存路徑
            
        Returns:
        --------
        bool : 保存是否成功
        """
        try:
            # 確保目錄存在
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 準備圖像數據
            img_to_save = self.prepare_image(img)
            
            # 保存圖像
            success = cv2.imwrite(path, img_to_save)
            
            if success:
                print(f"✓ 圖像已保存: {path}")
            else:
                print(f"✗ 圖像保存失敗: {path}")
                
            return success
            
        except Exception as e:
            print(f"✗ 保存圖像時發生錯誤: {e}")
            return False
    
    def _detect_image_type(self, img: np.ndarray) -> ImageType:
        """檢測圖像類型"""
        if len(img.shape) == 2:
            return ImageType.GRAYSCALE
        
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(img_ycrcb)
        
        cb_std, cr_std = np.std(cb), np.std(cr)
        print(f"Chroma std - Cb: {cb_std:.2f}, Cr: {cr_std:.2f}")
        
        if cb_std < self.config.color_threshold and cr_std < self.config.color_threshold:
            return ImageType.GRAYSCALE
        return ImageType.COLOR
    
    def split_color_channels(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """分離彩色通道"""
        return cv2.split(img)
    
    def combine_color_channels(self, b: np.ndarray, g: np.ndarray, r: np.ndarray) -> np.ndarray:
        """合併彩色通道"""
        return cv2.merge([b, g, r])
    
    def convert_channel_to_color(self, channel: np.ndarray, channel_name: str) -> np.ndarray:
        """將單通道轉換為對應顏色顯示"""
        if len(channel.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape: {channel.shape}")
        
        colored = np.zeros((*channel.shape, 3), dtype=np.uint8)
        
        channel_map = {
            'blue': (0, 0, 0), 'green': (0, 1, 0), 'red': (0, 0, 1)
        }
        
        if channel_name in channel_map:
            b_idx, g_idx, r_idx = channel_map[channel_name]
            colored[:, :, 0] = channel * (1 - b_idx)
            colored[:, :, 1] = channel * g_idx  
            colored[:, :, 2] = channel * r_idx
        else:
            raise ValueError(f"Unknown channel: {channel_name}")
        
        return colored

    def convert_single_channel_to_color(self, channel: np.ndarray, channel_name: str) -> np.ndarray:
        """
        將單通道圖像轉換為對應顏色的彩色圖像（quadtree.py 需要的函數）
        
        這是 convert_channel_to_color 的別名，為了與 quadtree.py 的調用保持一致
        """
        return self.convert_channel_to_color(channel, channel_name)

    def prepare_image(self, img: Union[np.ndarray, cp.ndarray]) -> np.ndarray:
        """預處理圖像數據"""
        if isinstance(img, cp.ndarray):
            img = cp.asnumpy(img)
        return np.array(img, dtype=np.uint8)

    def get_image_info(self, img: np.ndarray) -> Dict[str, Union[int, bool, str]]:
        """
        獲取圖像的基本資訊
        
        Parameters:
        -----------
        img : numpy.ndarray
            輸入圖像
            
        Returns:
        --------
        dict : 圖像資訊字典
        """
        img = self.prepare_image(img)
        
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

    def split_image_flexible(self, img: np.ndarray, split_size: int, 
                           block_base: bool = False, quad_tree: bool = False,
                           positions: Optional[List] = None) -> Union[List, Tuple]:
        """靈活的圖像分割"""
        if isinstance(img, cp.ndarray):
            xp = cp
        else:
            xp = np
        
        height, width = img.shape
        
        if quad_tree:
            return self._split_quadtree(img, split_size, positions, xp)
        else:
            return self._split_regular(img, split_size, block_base, xp)
    
    def _split_quadtree(self, img, block_size, positions, xp):
        """Quadtree分割實現"""
        height, width = img.shape
        sub_images, current_positions = [], []
        
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                actual_size = min(block_size, height - y, width - x)
                if actual_size == block_size:
                    sub_img = img[y:y+block_size, x:x+block_size]
                    sub_images.append(xp.asarray(sub_img))
                    current_positions.append((y, x))
        
        if positions is not None:
            positions.extend(current_positions)
        
        return sub_images, current_positions
    
    def _split_regular(self, img, split_size, block_base, xp):
        """常規分割實現"""
        height, width = img.shape
        sub_height, sub_width = height // split_size, width // split_size
        sub_images = []
        
        for i in range(split_size):
            for j in range(split_size):
                if block_base:
                    sub_img = img[i*sub_height:(i+1)*sub_height, 
                                j*sub_width:(j+1)*sub_width]
                else:
                    sub_img = img[i::split_size, j::split_size]
                sub_images.append(xp.asarray(sub_img))
        
        return sub_images
    
    def merge_image_flexible(self, sub_images: List, split_size: int, 
                           block_base: bool = False) -> cp.ndarray:
        """靈活的圖像合併"""
        if not sub_images:
            raise ValueError("No sub-images to merge")
        
        sub_images = [cp.asarray(img) for img in sub_images]
        sub_height, sub_width = sub_images[0].shape
        total_size = sub_height * split_size
        merged = cp.zeros((total_size, total_size), dtype=sub_images[0].dtype)
        
        for idx, sub_img in enumerate(sub_images):
            i, j = idx // split_size, idx % split_size
            if block_base:
                merged[i*sub_height:(i+1)*sub_height, 
                      j*sub_width:(j+1)*sub_width] = sub_img
            else:
                merged[i::split_size, j::split_size] = sub_img
        
        return merged

class MetricsCalculator:
    """統一的指標計算器"""
    
    @staticmethod
    def calculate_psnr(original: np.ndarray, embedded: np.ndarray) -> float:
        """計算PSNR"""
        mse = np.mean((original.astype(np.float64) - embedded.astype(np.float64)) ** 2)
        return float('inf') if mse == 0 else 10 * np.log10((255.0 ** 2) / mse)
    
    @staticmethod
    def calculate_ssim(original: np.ndarray, embedded: np.ndarray) -> float:
        """計算SSIM"""
        from skimage.metrics import structural_similarity
        return structural_similarity(original, embedded, data_range=255)
    
    @staticmethod
    def histogram_correlation(hist1: np.ndarray, hist2: np.ndarray) -> float:
        """計算直方圖相關性"""
        return np.corrcoef(hist1, hist2)[0, 1]
    
    def calculate_color_metrics(self, original: np.ndarray, embedded: np.ndarray) -> Dict[str, float]:
        """計算彩色圖像指標"""
        original = self._prepare_for_metrics(original)
        embedded = self._prepare_for_metrics(embedded)
        
        # 整體PSNR
        psnr = self.calculate_psnr(original, embedded)
        
        # 分通道計算SSIM和直方圖相關性
        b1, g1, r1 = cv2.split(original)
        b2, g2, r2 = cv2.split(embedded)
        
        ssim_values = [self.calculate_ssim(c1, c2) for c1, c2 in [(b1, b2), (g1, g2), (r1, r2)]]
        ssim = np.mean(ssim_values)
        
        hist_corr_values = []
        for c1, c2 in [(b1, b2), (g1, g2), (r1, r2)]:
            h1 = np.histogram(c1, bins=256, range=(0, 255))[0]
            h2 = np.histogram(c2, bins=256, range=(0, 255))[0]
            hist_corr_values.append(self.histogram_correlation(h1, h2))
        hist_corr = np.mean(hist_corr_values)
        
        return {
            'psnr': psnr,
            'ssim': ssim,
            'hist_corr': hist_corr,
            'channel_ssim': dict(zip(['blue', 'green', 'red'], ssim_values)),
            'channel_hist_corr': dict(zip(['blue', 'green', 'red'], hist_corr_values))
        }
    
    def _prepare_for_metrics(self, img):
        """為指標計算準備圖像"""
        if isinstance(img, cp.ndarray):
            img = cp.asnumpy(img)
        return img

# ================== 視覺化引擎 ==================

class BaseVisualizer(ABC):
    """視覺化基礎類"""
    
    def __init__(self, config: SystemConfig, image_processor: ImageProcessor):
        self.config = config
        self.processor = image_processor
    
    def ensure_dir(self, path: str):
        """確保目錄存在"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    def save_image(self, img: np.ndarray, path: str) -> bool:
        """保存圖像"""
        self.ensure_dir(path)
        img = self.processor.prepare_image(img)
        success = cv2.imwrite(path, img)
        print(f"{'✓' if success else '✗'} 圖像保存: {path}")
        return success
    
    def setup_plot(self, figsize: Optional[Tuple] = None):
        """設置標準圖表"""
        figsize = figsize or self.config.figure_size
        plt.figure(figsize=figsize)
    
    def finalize_plot(self, title: str, xlabel: str, ylabel: str, 
                     save_path: str, legend: bool = True):
        """完成圖表設置並保存"""
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=self.config.grid_alpha)
        
        if legend:
            plt.legend(fontsize=10)
        
        plt.tight_layout()
        self.ensure_dir(save_path)
        plt.savefig(save_path, dpi=self.config.dpi)
        plt.close()
        print(f"✓ 圖表保存: {save_path}")

class ImageVisualizer(BaseVisualizer):
    """圖像視覺化器"""
    
    def create_split_visualization(self, img: np.ndarray, split_size: int, 
                                 block_base: bool = False) -> np.ndarray:
        """創建分割示意圖"""
        height, width = img.shape
        visualization = img.copy()
        
        if block_base:
            sub_height, sub_width = height // split_size, width // split_size
            for i in range(1, split_size):
                y, x = i * sub_height, i * sub_width
                cv2.line(visualization, (0, y), (width, y), 255, 2)
                cv2.line(visualization, (x, 0), (x, height), 255, 2)
        else:
            for i in range(0, height, split_size):
                for j in range(0, width, split_size):
                    cv2.circle(visualization, (j, i), 3, 255, -1)
        
        return visualization
    
    def create_quadtree_visualization(self, block_info: Dict, img_shape: Tuple) -> np.ndarray:
        """創建quadtree視覺化"""
        height, width = img_shape
        visualization = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        for size_str in sorted(block_info.keys(), key=int, reverse=True):
            size = int(size_str)
            color = self.config.block_colors.get(size, (150, 150, 150))
            blocks = block_info[size_str]['blocks']
            
            for block in blocks:
                y, x = block['position']
                cv2.rectangle(visualization, (x, y), (x + size, y + size), color, -1)
                cv2.rectangle(visualization, (x, y), (x + size, y + size), (0, 0, 0), 1)
        
        return visualization
    
    def create_heatmap(self, original: np.ndarray, embedded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """創建差異熱圖"""
        original = self.processor.prepare_image(original)
        embedded = self.processor.prepare_image(embedded)
        
        if len(original.shape) == 3:
            diff = np.abs(embedded.astype(np.float32) - original.astype(np.float32))
            combined_diff = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            diff = np.abs(embedded.astype(np.float32) - original.astype(np.float32))
            combined_diff = diff
        
        heatmap = cv2.applyColorMap(combined_diff.astype(np.uint8), cv2.COLORMAP_JET)
        
        # 創建混合圖像
        if len(embedded.shape) == 2:
            embedded = cv2.cvtColor(embedded, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(embedded, 0.3, heatmap, 0.7, 0)
        
        return heatmap, blended
    
    def create_comparison_image(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """創建對比圖像"""
        img1, img2 = self.processor.prepare_image(img1), self.processor.prepare_image(img2)
        
        # 統一高度
        h1, h2 = img1.shape[0], img2.shape[0]
        max_h = max(h1, h2)
        
        def pad_to_height(img, target_h):
            if img.shape[0] < target_h:
                padding = ((0, target_h - img.shape[0]), (0, 0))
                if len(img.shape) == 3:
                    padding += ((0, 0),)
                img = np.pad(img, padding, mode='constant')
            return img
        
        img1, img2 = pad_to_height(img1, max_h), pad_to_height(img2, max_h)
        
        # 添加分隔線
        separator_shape = (max_h, 5) + img1.shape[2:]
        separator = np.ones(separator_shape, dtype=np.uint8) * 128
        
        return np.hstack((img1, separator, img2))

class MetricsVisualizer(BaseVisualizer):
    """指標視覺化器"""
    
    def create_distribution_chart(self, data_dict: Dict, title: str, 
                                xlabel: str, ylabel: str, save_path: str):
        """創建分布圖表"""
        self.setup_plot()
        
        items = sorted(data_dict.items(), 
                      key=lambda x: int(x[0]) if x[0].isdigit() else x[0], 
                      reverse=True)
        
        if items:
            labels, values = zip(*items)
            if any(label.isdigit() for label in labels):
                labels = [f"{label}x{label}" if label.isdigit() else label for label in labels]
                plt.xticks(rotation=45)
            
            plt.bar(labels, values, color='skyblue')
        
        self.finalize_plot(title, xlabel, ylabel, save_path, legend=False)
    
    def create_metrics_comparison(self, stages: List, metrics_dict: Dict, 
                                title: str, save_path: str):
        """創建指標比較圖表"""
        self.setup_plot()
        
        style_map = {
            'psnr': ('b.-', 'PSNR (dB)'),
            'ssim': ('r.-', 'SSIM'), 
            'hist_corr': ('g.-', 'Histogram Correlation'),
            'bpp': ('k.-', 'BPP')
        }
        
        for metric_name, values in metrics_dict.items():
            style, label = style_map.get(metric_name, ('.-', metric_name.upper()))
            plt.plot(stages, values, style, 
                    linewidth=self.config.line_width, 
                    markersize=self.config.marker_size, 
                    label=label)
        
        self.finalize_plot(title, 'Stage', 'Metrics Value', save_path)
    
    def create_histogram_comparison(self, original: np.ndarray, embedded: np.ndarray, save_path: str):
        """創建直方圖比較"""
        if len(original.shape) == 3:
            self.setup_plot((15, 5))
            channels = cv2.split(original)
            embedded_channels = cv2.split(embedded)
            colors = ['blue', 'green', 'red']
            titles = ['Blue Channel', 'Green Channel', 'Red Channel']
            
            for i, (ch, emb_ch, color, title) in enumerate(zip(channels, embedded_channels, colors, titles)):
                plt.subplot(1, 3, i+1)
                plt.hist(ch.flatten(), bins=256, range=[0,255], color=color, alpha=0.7, label='Original')
                plt.hist(emb_ch.flatten(), bins=256, range=[0,255], color=color, alpha=0.5, label='Embedded')
                plt.title(title)
                plt.xlabel("Pixel Value")
                plt.ylabel("Frequency")
                plt.legend()
        else:
            self.setup_plot()
            plt.hist(original.flatten(), bins=256, range=[0,255], alpha=0.7, label='Original')
            plt.hist(embedded.flatten(), bins=256, range=[0,255], alpha=0.5, label='Embedded')
            plt.title("Histogram Comparison")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.legend()
        
        plt.tight_layout()
        self.ensure_dir(save_path)
        plt.savefig(save_path, dpi=self.config.dpi)
        plt.close()

class ComparisonVisualizer(BaseVisualizer):
    """比較分析視覺化器"""
    
    def create_measurement_plots(self, df: pd.DataFrame, img_name: str, 
                               method: str, predictor: str, save_dir: str, 
                               is_color: bool = False):
        """創建精確測量圖表組合"""
        plots = [
            ('BPP', 'PSNR', 'BPP-PSNR', 'blue'),
            ('BPP', 'SSIM', 'BPP-SSIM', 'red'),
            ('BPP', 'Hist_Corr', 'BPP-Histogram Correlation', 'green')
        ]
        
        for x_col, y_col, title_suffix, color in plots:
            if y_col in df.columns:
                self._create_single_plot(df, x_col, y_col, title_suffix, color, 
                                       img_name, method, predictor, save_dir, is_color)
    
    def _create_single_plot(self, df, x_col, y_col, title_suffix, color, 
                          img_name, method, predictor, save_dir, is_color):
        """創建單個測量圖表"""
        self.setup_plot()
        
        image_type = "Color" if is_color else "Grayscale"
        title = f'{title_suffix} for {img_name} ({image_type})\nMethod: {method}, Predictor: {predictor}'
        
        plt.plot(df[x_col], df[y_col], color=color, 
                linewidth=self.config.line_width,
                marker='o', markersize=self.config.marker_size,
                label=f'{method}, {predictor}')
        
        # 添加數據標籤
        steps = max(1, len(df) // 10)
        for i in range(0, len(df), steps):
            row = df.iloc[i]
            plt.annotate(f'({row[x_col]:.4f}, {row[y_col]:.2f})',
                        (row[x_col], row[y_col]), 
                        textcoords="offset points", xytext=(0,10), ha='center',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                        fontsize=8)
        
        filename = f"{title_suffix.lower().replace('-', '_').replace(' ', '_')}.png"
        save_path = os.path.join(save_dir, filename)
        self.finalize_plot(title, x_col, y_col, save_path)
    
    def create_method_comparison(self, results_dict: Dict, img_name: str, 
                               predictor: str, save_dir: str):
        """創建方法比較圖表"""
        comparisons = [
            ('BPP', 'PSNR', 'Method Comparison (BPP vs PSNR)'),
            ('BPP', 'SSIM', 'Method Comparison (BPP vs SSIM)')
        ]
        
        for x_col, y_col, title_prefix in comparisons:
            self.setup_plot()
            
            for method, df in results_dict.items():
                if x_col in df.columns and y_col in df.columns:
                    color = self.config.method_colors.get(method, 'black')
                    plt.plot(df[x_col], df[y_col], color=color, 
                            linewidth=self.config.line_width, marker='o', 
                            markersize=self.config.marker_size, label=f'{method}')
            
            title = f'{title_prefix}\nPredictor: {predictor}, Image: {img_name}'
            filename = f"method_comparison_{x_col.lower()}_{y_col.lower()}.png"
            save_path = os.path.join(save_dir, filename)
            self.finalize_plot(title, x_col, y_col, save_path)

# ================== 統一的系統管理器 ==================

class PEEVisualizationSystem:
    """PEE視覺化系統的統一管理器"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.processor = ImageProcessor(self.config)
        self.metrics = MetricsCalculator()
        
        # 創建視覺化器
        self.image_viz = ImageVisualizer(self.config, self.processor)
        self.metrics_viz = MetricsVisualizer(self.config, self.processor)
        self.comparison_viz = ComparisonVisualizer(self.config, self.processor)
    
    # ================== 圖像處理API ==================
    
    def read_image_auto(self, filepath: str) -> Tuple[np.ndarray, ImageType]:
        """自動讀取並識別圖像類型"""
        return self.processor.read_image_auto(filepath)
    
    def get_image_info(self, img: np.ndarray) -> Dict[str, Union[int, bool, str]]:
        """獲取圖像基本資訊"""
        return self.processor.get_image_info(img)
    
    def split_image(self, img: np.ndarray, split_size: int, **kwargs) -> Union[List, Tuple]:
        """分割圖像"""
        return self.processor.split_image_flexible(img, split_size, **kwargs)
    
    def merge_image(self, sub_images: List, split_size: int, **kwargs) -> cp.ndarray:
        """合併圖像"""
        return self.processor.merge_image_flexible(sub_images, split_size, **kwargs)
    
    def process_color_channels(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """處理彩色通道"""
        b, g, r = self.processor.split_color_channels(img)
        return {'blue': b, 'green': g, 'red': r}
    
    # ================== 指標計算API ==================
    
    def calculate_metrics(self, original: np.ndarray, embedded: np.ndarray, 
                         is_color: bool = False) -> Dict[str, float]:
        """計算圖像品質指標"""
        if is_color:
            return self.metrics.calculate_color_metrics(original, embedded)
        else:
            return {
                'psnr': self.metrics.calculate_psnr(original, embedded),
                'ssim': self.metrics.calculate_ssim(original, embedded)
            }
    
    # ================== 視覺化API ==================
    
    def visualize_split(self, img: np.ndarray, split_size: int, 
                       block_base: bool = False, save_path: Optional[str] = None) -> np.ndarray:
        """分割視覺化"""
        result = self.image_viz.create_split_visualization(img, split_size, block_base)
        if save_path:
            self.image_viz.save_image(result, save_path)
        return result
    
    def visualize_quadtree(self, block_info: Dict, img_shape: Tuple, 
                          save_path: Optional[str] = None) -> np.ndarray:
        """Quadtree視覺化"""
        result = self.image_viz.create_quadtree_visualization(block_info, img_shape)
        if save_path:
            self.image_viz.save_image(result, save_path)
        return result
    
    def enhance_with_grid_visualization(self, combined_stage, b_stage_img, g_stage_img, r_stage_img, 
                                    image_dir, stage_num):
        """增強網格視覺化 - 為彩色圖像的各通道創建帶網格的視覺化"""
        try:
            import os
            grid_dir = f"{image_dir}/with_grid"
            os.makedirs(grid_dir, exist_ok=True)
            
            channels = {'blue': b_stage_img, 'green': g_stage_img, 'red': r_stage_img}
            
            if 'channel_block_info' in combined_stage:
                for ch_name, ch_img in channels.items():
                    if ch_name in combined_stage['channel_block_info']:
                        ch_block_info = combined_stage['channel_block_info'][ch_name]
                        grid_visualization = self.visualize_quadtree(ch_block_info, ch_img.shape)
                        grid_path = f"{grid_dir}/stage_{stage_num}_{ch_name}_channel_grid.png"
                        self.image_viz.save_image(grid_visualization, grid_path)
            
            print(f"Enhanced grid visualization completed for stage {stage_num}")
        except Exception as e:
            print(f"Warning: Enhanced grid visualization failed: {e}")

    def enhance_block_visualizations(self, combined_stage, img, image_dir, stage_num):
        """增強區塊視覺化 - 為各種區塊大小創建詳細的視覺化"""
        try:
            import os
            block_viz_dir = f"{image_dir}/block_visualizations"
            os.makedirs(block_viz_dir, exist_ok=True)
            
            if 'channel_block_info' in combined_stage:
                for ch_name in ['blue', 'green', 'red']:
                    if ch_name in combined_stage['channel_block_info']:
                        ch_block_info = combined_stage['channel_block_info'][ch_name]
                        ch_viz_dir = f"{block_viz_dir}/{ch_name}"
                        os.makedirs(ch_viz_dir, exist_ok=True)
                        ch_img = cv2.split(img)[['blue', 'green', 'red'].index(ch_name)]
                        self.create_all_visualizations(ch_block_info, ch_img, ch_viz_dir, stage_num)
            elif 'block_info' in combined_stage:
                self.create_all_visualizations(combined_stage['block_info'], img, block_viz_dir, stage_num)
                
            print(f"Enhanced block visualizations completed for stage {stage_num}")
        except Exception as e:
            print(f"Warning: Enhanced block visualizations failed: {e}")

    def enhance_final_visualizations(self, color_pee_stages, final_b_img, final_g_img, final_r_img, 
                                    output_dir):
        """增強最終視覺化 - 為最終結果創建綜合視覺化"""
        try:
            import os
            final_viz_dir = f"{output_dir}/final_visualizations"
            os.makedirs(final_viz_dir, exist_ok=True)
            
            final_color_img = self.processor.combine_color_channels(final_b_img, final_g_img, final_r_img)
            channels = {'blue': final_b_img, 'green': final_g_img, 'red': final_r_img}
            
            for ch_name, ch_img in channels.items():
                colored_ch = self.processor.convert_channel_to_color(ch_img, ch_name)
                ch_path = f"{final_viz_dir}/final_{ch_name}_colored.png"
                self.image_viz.save_image(colored_ch, ch_path)
            
            final_path = f"{final_viz_dir}/final_combined.png"
            cv2.imwrite(final_path, final_color_img)
            
            if color_pee_stages:
                final_stage = color_pee_stages[-1]
                if 'channel_block_info' in final_stage:
                    for ch_name in ['blue', 'green', 'red']:
                        if ch_name in final_stage['channel_block_info']:
                            ch_block_info = final_stage['channel_block_info'][ch_name]
                            ch_img = channels[ch_name]
                            final_quadtree = self.visualize_quadtree(ch_block_info, ch_img.shape)
                            quadtree_path = f"{final_viz_dir}/final_{ch_name}_quadtree.png"
                            self.image_viz.save_image(final_quadtree, quadtree_path)
            
            print(f"Enhanced final visualizations completed")
        except Exception as e:
            print(f"Warning: Enhanced final visualizations failed: {e}")
    
    def create_heatmap(self, original: np.ndarray, embedded: np.ndarray, 
                      save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """創建差異熱圖"""
        heatmap, blended = self.image_viz.create_heatmap(original, embedded)
        if save_path:
            self.image_viz.save_image(heatmap, save_path)
            blend_path = save_path.replace('.png', '_blend.png')
            self.image_viz.save_image(blended, blend_path)
        return heatmap, blended
    
    def save_comparison_image(self, img1: np.ndarray, img2: np.ndarray, save_path: str) -> np.ndarray:
        """保存比較圖像"""
        comparison = self.image_viz.create_comparison_image(img1, img2)
        self.image_viz.save_image(comparison, save_path)
        return comparison
    
    def create_block_distribution_chart(self, block_info: Dict, save_path: str, 
                                      stage_num: int, channel_name: Optional[str] = None):
        """創建區塊分布圖表"""
        data = {size: len(blocks['blocks']) for size, blocks in block_info.items() 
                if len(blocks['blocks']) > 0}
        
        title = f'Block Size Distribution in Stage {stage_num}'
        if channel_name:
            title += f' ({channel_name.capitalize()} Channel)'
        
        self.metrics_viz.create_distribution_chart(data, title, 'Block Size', 'Count', save_path)
    
    def create_metrics_comparison_chart(self, stages: List, metrics: Dict, 
                                      title: str, save_path: str):
        """創建指標比較圖表"""
        self.metrics_viz.create_metrics_comparison(stages, metrics, title, save_path)
    
    def create_histogram_comparison(self, original: np.ndarray, embedded: np.ndarray, save_path: str):
        """創建直方圖比較"""
        self.metrics_viz.create_histogram_comparison(original, embedded, save_path)
    
    def plot_precise_measurements(self, df: pd.DataFrame, img_name: str, 
                                method: str, predictor: str, save_dir: str, 
                                is_color: bool = False):
        """繪製精確測量結果"""
        self.comparison_viz.create_measurement_plots(df, img_name, method, predictor, save_dir, is_color)
    
    def plot_method_comparison(self, results_dict: Dict, img_name: str, 
                             predictor: str, save_dir: str):
        """繪製方法比較"""
        self.comparison_viz.create_method_comparison(results_dict, img_name, predictor, save_dir)
    
    # ================== 批量處理API ==================
    
    def create_all_visualizations(self, block_info: Dict, original_img: np.ndarray, 
                                output_dir: str, stage_num: int) -> Dict[str, str]:
        """創建所有quadtree視覺化"""
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        # 標準區塊大小
        block_sizes = [16, 32, 64, 128, 256, 512, 1024]
        
        for size in block_sizes:
            if str(size) in block_info and len(block_info[str(size)]['blocks']) > 0:
                save_path = f"{output_dir}/stage_{stage_num}_blocks_{size}x{size}.png"
                result = self.image_viz.create_specific_quadtree_blocks_visualization(
                    block_info, original_img, size)
                self.image_viz.save_image(result, save_path)
                results[size] = save_path
        
        # 創建整體視覺化
        combined_path = f"{output_dir}/stage_{stage_num}_all_blocks.png"
        combined_result = self.visualize_quadtree(block_info, original_img.shape[:2], combined_path)
        results['all'] = combined_path
        
        return results
    
    # ================== 配置管理API ==================
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_config(self) -> SystemConfig:
        """獲取當前配置"""
        return self.config

# ================== 向後兼容接口 ==================

# 創建默認實例
_default_system = PEEVisualizationSystem()

# 向後兼容函數
def visualize_split(img, split_size, block_base=False):
    return _default_system.visualize_split(img, split_size, block_base)

def visualize_quadtree(block_info, img_shape):
    return _default_system.visualize_quadtree(block_info, img_shape)

def save_comparison_image(img1, img2, save_path, labels=None):
    return _default_system.save_comparison_image(img1, img2, save_path)

def create_block_size_distribution_chart(block_info, save_path, stage_num, channel_name=None):
    return _default_system.create_block_distribution_chart(block_info, save_path, stage_num, channel_name)

def visualize_embedding_heatmap(original_img, embedded_img, save_path):
    return _default_system.create_heatmap(original_img, embedded_img, save_path)

def is_grayscale(img, threshold=5.0):
    system = PEEVisualizationSystem()
    return system.processor._detect_image_type(img) == ImageType.GRAYSCALE

def split_color_channels(img):
    return _default_system.processor.split_color_channels(img)

def combine_color_channels(b, g, r):
    return _default_system.processor.combine_color_channels(b, g, r)

def calculate_color_metrics(original_img, embedded_img):
    return _default_system.metrics.calculate_color_metrics(original_img, embedded_img)

def get_image_info(img):
    return _default_system.get_image_info(img)

def convert_single_channel_to_color(channel, channel_name):
    """向後兼容：單通道轉彩色"""
    return _default_system.processor.convert_single_channel_to_color(channel, channel_name)

def enhance_with_grid_visualization(combined_stage, b_stage_img, g_stage_img, r_stage_img, 
                                   image_dir, stage_num):
    """向後兼容：增強網格視覺化"""
    return _default_system.enhance_with_grid_visualization(
        combined_stage, b_stage_img, g_stage_img, r_stage_img, image_dir, stage_num
    )

def enhance_block_visualizations(combined_stage, img, image_dir, stage_num):
    """向後兼容：增強區塊視覺化"""
    return _default_system.enhance_block_visualizations(
        combined_stage, img, image_dir, stage_num
    )

def enhance_final_visualizations(color_pee_stages, final_b_img, final_g_img, final_r_img, 
                                output_dir):
    """向後兼容：增強最終視覺化"""
    return _default_system.enhance_final_visualizations(
        color_pee_stages, final_b_img, final_g_img, final_r_img, output_dir
    )

def save_image(img, path):
    """向後兼容：保存圖像"""
    return _default_system.image_viz.save_image(img, path)