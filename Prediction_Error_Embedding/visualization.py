"""
visualization.py - 視覺化相關輔助函數模組

這個模組提供各種視覺化工具，用於產生實驗結果的視覺化呈現，
特別是針對不同的數據隱藏方法（rotation、split 和 quadtree）。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import cupy as cp

# 從現有模組導入可能需要的函數
from common import calculate_psnr, calculate_ssim, histogram_correlation
from image_processing import save_image, generate_histogram

def visualize_split(img, split_size, block_base=False):
    """
    創建分割示意圖，顯示圖像如何被分割為子圖像
    
    Parameters:
    -----------
    img : numpy.ndarray
        原始圖像
    split_size : int
        分割大小
    block_base : bool
        是否使用 block-based 分割
        
    Returns:
    --------
    numpy.ndarray
        分割示意圖，其中分割線使用白色或其他顏色標示
    """
    # 創建示意圖
    height, width = img.shape
    visualization = img.copy()
    
    # 計算子區塊大小
    if block_base:
        # Block-based 分割
        sub_height = height // split_size
        sub_width = width // split_size
        
        # 繪製水平線
        for i in range(1, split_size):
            y = i * sub_height
            cv2.line(visualization, (0, y), (width, y), 255, 2)
        
        # 繪製垂直線
        for j in range(1, split_size):
            x = j * sub_width
            cv2.line(visualization, (x, 0), (x, height), 255, 2)
    else:
        # Quarter-based 分割 (交錯式分割)
        # 這種分割方式較難可視化，我們使用不同的符號標示
        for i in range(0, height, split_size):
            for j in range(0, width, split_size):
                # 繪製點以標記分割的開始位置
                cv2.circle(visualization, (j, i), 3, 255, -1)
    
    return visualization

def visualize_quadtree(block_info, img_shape):
    """
    創建 quadtree 分割視覺化，使用不同顏色標示不同大小的區塊
    
    Parameters:
    -----------
    block_info : dict
        包含區塊資訊的字典，格式為 {'size': {'blocks': [{'position': (y, x), 'size': size}, ...]}}
    img_shape : tuple
        原圖尺寸 (height, width)
        
    Returns:
    --------
    numpy.ndarray
        Quadtree 分割視覺化圖像
    """
    # 創建空白圖像
    height, width = img_shape
    visualization = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白色背景
    
    # 為不同大小的區塊定義不同顏色
    colors = {
        512: (200, 200, 200),   # 淺灰色
        256: (100, 100, 200),   # 淺藍色
        128: (100, 200, 100),   # 淺綠色
        64: (200, 100, 100),    # 淺紅色
        32: (200, 200, 100),    # 淺黃色
        16: (200, 100, 200)     # 淺紫色
    }
    
    # 從大區塊到小區塊繪製，以確保小區塊不被大區塊覆蓋
    for size_str in sorted(block_info.keys(), key=int, reverse=True):
        size = int(size_str)
        blocks = block_info[size_str]['blocks']
        color = colors.get(size, (150, 150, 150))  # 如果沒有定義顏色，使用灰色
        
        for block in blocks:
            y, x = block['position']
            cv2.rectangle(visualization, (x, y), (x + size, y + size), color, -1)  # 填充區塊
            cv2.rectangle(visualization, (x, y), (x + size, y + size), (0, 0, 0), 1)  # 黑色邊框
    
    return visualization

def save_comparison_image(img1, img2, save_path, labels=None):
    """
    將兩張圖像水平拼接並儲存，用於比較
    
    Parameters:
    -----------
    img1 : numpy.ndarray
        第一張圖像
    img2 : numpy.ndarray
        第二張圖像
    save_path : str
        儲存路徑
    labels : tuple, optional
        圖像標籤，格式為 (label1, label2)
    """
    # 確保兩張圖像有相同高度
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    
    if h1 != h2:
        # 調整高度
        max_h = max(h1, h2)
        if h1 < max_h:
            img1 = np.pad(img1, ((0, max_h - h1), (0, 0)), mode='constant', constant_values=0)
        else:
            img2 = np.pad(img2, ((0, max_h - h2), (0, 0)), mode='constant', constant_values=0)
    
    # 在中間加入分隔線
    separator = np.ones((max(h1, h2), 5), dtype=np.uint8) * 128  # 灰色分隔線
    
    # 水平拼接圖像
    comparison = np.hstack((img1, separator, img2))
    
    # 如果有提供標籤，添加文字
    if labels:
        # 轉換為彩色圖像以便添加彩色文字
        comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_GRAY2RGB)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (0, 0, 255)  # 紅色
        font_thickness = 2
        
        # 添加第一個標籤
        cv2.putText(comparison_rgb, labels[0], (10, 30), font, font_scale, font_color, font_thickness)
        
        # 添加第二個標籤
        cv2.putText(comparison_rgb, labels[1], (w1 + 15, 30), font, font_scale, font_color, font_thickness)
        
        # 儲存彩色圖像
        cv2.imwrite(save_path, comparison_rgb)
    else:
        # 儲存灰階圖像
        cv2.imwrite(save_path, comparison)

def create_block_size_distribution_chart(block_info, save_path, stage_num):
    """
    創建區塊大小分布統計圖
    
    Parameters:
    -----------
    block_info : dict
        包含區塊資訊的字典
    save_path : str
        儲存路徑
    stage_num : int
        階段編號
    """
    plt.figure(figsize=(10, 6))
    sizes = []
    counts = []
    
    for size_str in block_info:
        size = int(size_str)
        count = len(block_info[size_str]['blocks'])
        if count > 0:
            sizes.append(f"{size}x{size}")
            counts.append(count)
    
    # 按區塊大小從大到小排序
    sorted_indices = sorted(range(len(sizes)), key=lambda i: int(sizes[i].split('x')[0]), reverse=True)
    sorted_sizes = [sizes[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    
    plt.bar(sorted_sizes, sorted_counts, color='skyblue')
    plt.xlabel('Block Size')
    plt.ylabel('Count')
    plt.title(f'Block Size Distribution in Stage {stage_num}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_rotation_angles(angles, img_shape, save_path):
    """
    視覺化旋轉角度分布
    
    Parameters:
    -----------
    angles : dict
        區塊大小到旋轉角度的映射，例如 {512: 90, 256: 180, ...}
    img_shape : tuple
        原圖尺寸 (height, width)
    save_path : str
        儲存路徑
    """
    height, width = img_shape
    visualization = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白色背景
    
    # 角度對應的顏色
    angle_colors = {
        0: (200, 200, 200),      # 灰色
        90: (100, 200, 100),     # 綠色
        180: (100, 100, 200),    # 藍色
        270: (200, 100, 100),    # 紅色
        -90: (200, 200, 100),    # 黃色
        -180: (200, 100, 200),   # 紫色
        -270: (100, 200, 200)    # 青色
    }
    
    # 繪製圖例
    legend_y = 20
    for angle, color in angle_colors.items():
        cv2.putText(visualization, f"{angle}°", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 2)
        legend_y += 25
    
    # 在圖像上標示每個區塊的旋轉角度
    for size, angle in angles.items():
        color = angle_colors.get(angle, (0, 0, 0))  # 如果沒有對應顏色，使用黑色
        cv2.putText(visualization, f"{size}px: {angle}°", (width//2 - 100, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    cv2.imwrite(save_path, visualization)

def create_metrics_comparison_chart(stages, metrics, save_path, title):
    """
    創建階段指標比較圖表
    
    Parameters:
    -----------
    stages : list
        階段編號列表
    metrics : dict
        指標數據，格式為 {'psnr': [...], 'ssim': [...], ...}
    save_path : str
        儲存路徑
    title : str
        圖表標題
    """
    plt.figure(figsize=(12, 8))
    
    for metric_name, values in metrics.items():
        if metric_name == 'psnr':
            plt.plot(stages, values, 'b.-', linewidth=2, markersize=8, label='PSNR (dB)')
        elif metric_name == 'ssim':
            plt.plot(stages, values, 'r.-', linewidth=2, markersize=8, label='SSIM')
        elif metric_name == 'hist_corr':
            plt.plot(stages, values, 'g.-', linewidth=2, markersize=8, label='Histogram Correlation')
        elif metric_name == 'bpp':
            plt.plot(stages, values, 'k.-', linewidth=2, markersize=8, label='BPP')
    
    plt.xlabel('Stage', fontsize=12)
    plt.ylabel('Metrics Value', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def visualize_embedding_heatmap(original_img, embedded_img, save_path):
    """
    創建嵌入熱圖，顯示圖像中哪些區域有較多修改
    
    Parameters:
    -----------
    original_img : numpy.ndarray
        原始圖像
    embedded_img : numpy.ndarray
        嵌入後的圖像
    save_path : str
        儲存路徑
    """
    # 計算差異
    if isinstance(original_img, cp.ndarray):
        original_img = cp.asnumpy(original_img)
    if isinstance(embedded_img, cp.ndarray):
        embedded_img = cp.asnumpy(embedded_img)
    
    diff = np.abs(embedded_img.astype(np.float32) - original_img.astype(np.float32))
    
    # 正規化差異到 0-255 範圍
    if np.max(diff) > 0:
        diff = diff / np.max(diff) * 255
    
    # 應用色彩映射
    diff_colored = cv2.applyColorMap(diff.astype(np.uint8), cv2.COLORMAP_JET)
    
    # 儲存熱圖
    cv2.imwrite(save_path, diff_colored)
    
    # 將熱圖與原圖融合以便更好地觀察
    alpha = 0.7
    blended = cv2.addWeighted(embedded_img, 1-alpha, diff_colored, alpha, 0)
    
    # 儲存融合圖
    blend_path = save_path.replace('.png', '_blend.png')
    cv2.imwrite(blend_path, blended)
    
    return diff, diff_colored, blended

def create_payload_distribution_chart(pee_stages, save_path):
    """
    創建各階段有效載荷分布圖表
    
    Parameters:
    -----------
    pee_stages : list
        包含所有階段資訊的列表
    save_path : str
        儲存路徑
    """
    plt.figure(figsize=(12, 8))
    
    stages = []
    payloads = []
    accumulated_payloads = []
    total = 0
    
    for stage in pee_stages:
        stages.append(stage['embedding'])
        payloads.append(stage['payload'])
        total += stage['payload']
        accumulated_payloads.append(total)
    
    # 繪製階段有效載荷
    plt.bar(stages, payloads, color='skyblue', alpha=0.7, label='Stage Payload')
    
    # 繪製累積有效載荷曲線
    plt.plot(stages, accumulated_payloads, 'r.-', linewidth=2, markersize=8, label='Accumulated Payload')
    
    # 添加數據標籤
    for i, (stage, payload, acc_payload) in enumerate(zip(stages, payloads, accumulated_payloads)):
        plt.annotate(f"{payload}", (stage, payload), textcoords="offset points", 
                    xytext=(0,10), ha='center')
        plt.annotate(f"{acc_payload}", (stage, acc_payload), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.xlabel('Stage', fontsize=12)
    plt.ylabel('Payload (bits)', fontsize=12)
    plt.title('Payload Distribution Across Stages', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def create_el_distribution_chart(pee_stages, save_path):
    """
    創建嵌入層級 (EL) 分布圖表
    
    Parameters:
    -----------
    pee_stages : list
        包含所有階段資訊的列表
    save_path : str
        儲存路徑
    """
    plt.figure(figsize=(12, 8))
    
    stages = []
    avg_els = []
    max_els = []
    
    for stage in pee_stages:
        stages.append(stage['embedding'])
        
        # 收集所有區塊的 EL 值
        all_els = []
        
        if 'block_params' in stage:
            for block in stage['block_params']:
                if 'EL' in block:
                    all_els.append(block['EL'])
        
        # 計算平均和最大 EL
        if all_els:
            avg_els.append(np.mean(all_els))
            max_els.append(np.max(all_els))
        else:
            avg_els.append(0)
            max_els.append(0)
    
    # 繪製平均 EL
    plt.plot(stages, avg_els, 'b.-', linewidth=2, markersize=8, label='Average EL')
    
    # 繪製最大 EL
    plt.plot(stages, max_els, 'r.-', linewidth=2, markersize=8, label='Maximum EL')
    
    plt.xlabel('Stage', fontsize=12)
    plt.ylabel('Embedding Level (EL)', fontsize=12)
    plt.title('Embedding Level Distribution Across Stages', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def create_histogram_animation(pee_stages, original_img, save_dir, imgName, method):
    """
    創建直方圖變化動畫
    
    Parameters:
    -----------
    pee_stages : list
        包含所有階段資訊的列表
    original_img : numpy.ndarray
        原始圖像
    save_dir : str
        儲存目錄
    imgName : str
        圖像名稱
    method : str
        使用的方法
    """
    # 確保輸出目錄存在
    animation_dir = f"{save_dir}/animation"
    os.makedirs(animation_dir, exist_ok=True)
    
    # 獲取原始圖像直方圖
    orig_hist = generate_histogram(original_img)
    
    # 為每個階段創建直方圖比較
    for i, stage in enumerate(pee_stages):
        stage_img = cp.asnumpy(stage['stage_img'])
        stage_hist = generate_histogram(stage_img)
        
        # 創建直方圖比較圖
        plt.figure(figsize=(12, 6))
        
        # 原始直方圖
        plt.subplot(1, 2, 1)
        plt.bar(range(256), orig_hist, alpha=0.7, color='blue')
        plt.title(f"Original Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        
        # 階段直方圖
        plt.subplot(1, 2, 2)
        plt.bar(range(256), stage_hist, alpha=0.7, color='red')
        plt.title(f"Stage {i} Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(f"{animation_dir}/{imgName}_{method}_histogram_stage_{i}.png")
        plt.close()
    
    # 可以選擇使用 ImageMagick 或其他工具將這些圖像合成動畫
    # 這裡提供命令提示
    print(f"圖像已儲存至 {animation_dir}，可使用以下命令創建動畫：")
    print(f"convert -delay 100 {animation_dir}/{imgName}_{method}_histogram_stage_*.png {animation_dir}/{imgName}_{method}_histogram_animation.gif")