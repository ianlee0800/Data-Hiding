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
import matplotlib.patches as patches

from matplotlib.patches import FancyBboxPatch
from color import combine_color_channels
from common import calculate_psnr, calculate_ssim, histogram_correlation
from image_processing import save_image, generate_histogram, merge_image_flexible

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
    創建 quadtree 分割視覺化，使用不同顏色標示不同大小的區塊（無文字標註）
    
    Parameters:
    -----------
    block_info : dict
        包含區塊資訊的字典，格式為 {'size': {'blocks': [{'position': (y, x), 'size': size}, ...]}}
    img_shape : tuple
        原圖尺寸 (height, width)
        
    Returns:
    --------
    numpy.ndarray
        Quadtree 分割視覺化圖像（純視覺化，無文字）
    """
    # 創建空白圖像
    height, width = img_shape
    visualization = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白色背景
    
    # 為不同大小的區塊定義不同顏色
    colors = {
        1024: (220, 220, 220),  # 更淺的灰色
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
        color = colors.get(size, (150, 150, 150))
        
        for block in blocks:
            y, x = block['position']
            cv2.rectangle(visualization, (x, y), (x + size, y + size), color, -1)  # 填充區塊
            cv2.rectangle(visualization, (x, y), (x + size, y + size), (0, 0, 0), 1)  # 黑色邊框
    
    return visualization

def save_comparison_image(img1, img2, save_path, labels=None):
    """
    將兩張圖像水平拼接並儲存，用於比較（無文字標註版本）
    
    Parameters:
    -----------
    img1 : numpy.ndarray
        第一張圖像
    img2 : numpy.ndarray
        第二張圖像
    save_path : str
        儲存路徑
    labels : tuple, optional
        圖像標籤（忽略，保持API兼容性）
    """
    # 確保兩張圖像有相同高度
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if h1 != h2:
        # 調整高度
        max_h = max(h1, h2)
        if h1 < max_h:
            if len(img1.shape) == 3:
                img1 = np.pad(img1, ((0, max_h - h1), (0, 0), (0, 0)), mode='constant', constant_values=0)
            else:
                img1 = np.pad(img1, ((0, max_h - h1), (0, 0)), mode='constant', constant_values=0)
        else:
            if len(img2.shape) == 3:
                img2 = np.pad(img2, ((0, max_h - h2), (0, 0), (0, 0)), mode='constant', constant_values=0)
            else:
                img2 = np.pad(img2, ((0, max_h - h2), (0, 0)), mode='constant', constant_values=0)
    
    # 在中間加入分隔線
    if len(img1.shape) == 3:  # 彩色圖像
        separator = np.ones((max(h1, h2), 5, 3), dtype=np.uint8) * 128
    else:  # 灰度圖像
        separator = np.ones((max(h1, h2), 5), dtype=np.uint8) * 128
    
    # 水平拼接圖像（無文字標註）
    comparison = np.hstack((img1, separator, img2))
    
    # 直接儲存，不添加任何文字
    cv2.imwrite(save_path, comparison)

def create_block_size_distribution_chart(block_info, save_path, stage_num, channel_name=None):
    """
    創建區塊大小分布統計圖（支持通道名稱）
    
    Parameters:
    -----------
    block_info : dict
        包含區塊資訊的字典
    save_path : str
        儲存路徑
    stage_num : int
        階段編號
    channel_name : str, optional
        通道名稱（如果是彩色圖像的特定通道）
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
    
    # 🔧 修改：根據是否有通道名稱調整標題
    if channel_name:
        plt.title(f'Block Size Distribution in Stage {stage_num} ({channel_name.capitalize()} Channel)')
    else:
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
    
    # 將嵌入圖像轉換為3通道，以便與彩色差異圖混合
    embedded_img_colored = cv2.cvtColor(embedded_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    blended = cv2.addWeighted(embedded_img_colored, 1-alpha, diff_colored, alpha, 0)
    
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
    
    # Add these functions to visualization.py

def visualize_color_histograms(img, save_path, title="Color Image Histograms"):
    """
    Create and save histograms for each channel of a color image.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input color image (BGR format)
    save_path : str
        Path to save the histogram image
    title : str, optional
        Main title for the histogram plot
    """
    
    # Split channels
    b, g, r = cv2.split(img)
    
    # Create figure with subplots
    plt.figure(figsize=(15, 5))
    
    # Blue channel histogram
    plt.subplot(1, 3, 1)
    plt.hist(b.flatten(), bins=256, range=[0,255], color='blue', alpha=0.7)
    plt.title("Blue Channel")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    # Green channel histogram  
    plt.subplot(1, 3, 2)
    plt.hist(g.flatten(), bins=256, range=[0,255], color='green', alpha=0.7)
    plt.title("Green Channel")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    # Red channel histogram
    plt.subplot(1, 3, 3)
    plt.hist(r.flatten(), bins=256, range=[0,255], color='red', alpha=0.7)
    plt.title("Red Channel")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    # Add main title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for main title
    
    # Save figure
    plt.savefig(save_path)
    plt.close()

def create_color_heatmap(original_img, embedded_img, save_path, intensity_scale=1.0):
    """
    Create a heatmap showing the differences between original and embedded color images.
    
    Parameters:
    -----------
    original_img : numpy.ndarray
        Original color image
    embedded_img : numpy.ndarray
        Embedded color image
    save_path : str
        Path to save the heatmap
    intensity_scale : float, optional
        Scale factor for difference visualization (default: 1.0)
    """
    
    # Calculate absolute difference for each channel
    b_diff = np.abs(embedded_img[:,:,0].astype(np.float32) - original_img[:,:,0].astype(np.float32))
    g_diff = np.abs(embedded_img[:,:,1].astype(np.float32) - original_img[:,:,1].astype(np.float32))
    r_diff = np.abs(embedded_img[:,:,2].astype(np.float32) - original_img[:,:,2].astype(np.float32))
    
    # Enhance contrast in the visualization
    b_diff = np.clip(b_diff * intensity_scale, 0, 255)
    g_diff = np.clip(g_diff * intensity_scale, 0, 255)
    r_diff = np.clip(r_diff * intensity_scale, 0, 255)
    
    # Create a combined RGB heatmap
    heatmap = np.stack([r_diff, g_diff, b_diff], axis=2).astype(np.uint8)
    
    # Apply color map for better visualization
    combined_diff = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    heatmap_colored = cv2.applyColorMap(combined_diff, cv2.COLORMAP_JET)
    
    # Save heatmap
    cv2.imwrite(save_path, heatmap_colored)
    
    # Create a blend of original with heatmap overlay
    alpha = 0.7
    embedded_img_colored = embedded_img.copy()
    blended = cv2.addWeighted(embedded_img_colored, 1-alpha, heatmap_colored, alpha, 0)
    
    # Save blended image
    blend_path = save_path.replace('.png', '_blend.png')
    cv2.imwrite(blend_path, blended)
    
    # Create per-channel heatmaps
    blue_heatmap = cv2.applyColorMap(b_diff.astype(np.uint8), cv2.COLORMAP_JET)
    green_heatmap = cv2.applyColorMap(g_diff.astype(np.uint8), cv2.COLORMAP_JET)
    red_heatmap = cv2.applyColorMap(r_diff.astype(np.uint8), cv2.COLORMAP_JET)
    
    # Save channel heatmaps
    cv2.imwrite(save_path.replace('.png', '_blue.png'), blue_heatmap)
    cv2.imwrite(save_path.replace('.png', '_green.png'), green_heatmap)
    cv2.imwrite(save_path.replace('.png', '_red.png'), red_heatmap)
    
    return heatmap_colored, blended

def visualize_color_metrics_comparison(pee_stages, save_path, title="Channel Metrics Comparison"):
    """
    Create visualization comparing image quality metrics across different channels.
    
    Parameters:
    -----------
    pee_stages : list
        List of stages with channel metrics
    save_path : str
        Path to save the visualization
    title : str, optional
        Title for the plot
    """
    
    # Extract data
    stages = []
    blue_psnr = []
    green_psnr = []
    red_psnr = []
    blue_ssim = []
    green_ssim = []
    red_ssim = []
    
    for i, stage in enumerate(pee_stages):
        if 'channel_metrics' in stage:
            stages.append(i)
            blue_psnr.append(stage['channel_metrics']['blue']['psnr'])
            green_psnr.append(stage['channel_metrics']['green']['psnr'])
            red_psnr.append(stage['channel_metrics']['red']['psnr'])
            blue_ssim.append(stage['channel_metrics']['blue']['ssim'])
            green_ssim.append(stage['channel_metrics']['green']['ssim'])
            red_ssim.append(stage['channel_metrics']['red']['ssim'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot PSNR values
    ax1.plot(stages, blue_psnr, 'b-o', linewidth=2, label='Blue Channel')
    ax1.plot(stages, green_psnr, 'g-o', linewidth=2, label='Green Channel')
    ax1.plot(stages, red_psnr, 'r-o', linewidth=2, label='Red Channel')
    ax1.set_xlabel('Stage')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('PSNR Comparison by Channel')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot SSIM values
    ax2.plot(stages, blue_ssim, 'b-o', linewidth=2, label='Blue Channel')
    ax2.plot(stages, green_ssim, 'g-o', linewidth=2, label='Green Channel')
    ax2.plot(stages, red_ssim, 'r-o', linewidth=2, label='Red Channel')
    ax2.set_xlabel('Stage')
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM Comparison by Channel')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Add main title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for main title
    
    # Save figure
    plt.savefig(save_path)
    plt.close()

def create_color_channel_comparison(original_img, embedded_img, save_path):
    """
    創建彩色通道對比的視覺化（無文字標註版本）
    
    Parameters:
    -----------
    original_img : numpy.ndarray
        原始彩色圖像
    embedded_img : numpy.ndarray
        嵌入後的彩色圖像
    save_path : str
        保存路徑
    """
    import numpy as np
    import cv2
    
    # Split channels
    b1, g1, r1 = cv2.split(original_img)
    b2, g2, r2 = cv2.split(embedded_img)
    
    # Create blank canvas for the comparison
    h, w = original_img.shape[:2]
    comparison = np.ones((h*3, w*2 + 10, 3), dtype=np.uint8) * 255  # White background
    
    # Place images in grid (無文字標註，純圖像對比)
    # Blue channel row
    comparison[:h, :w, 0] = b1
    comparison[:h, :w, 1:] = 0
    comparison[:h, w+10:w*2+10, 0] = b2
    comparison[:h, w+10:w*2+10, 1:] = 0
    
    # Green channel row
    comparison[h:h*2, :w, 1] = g1
    comparison[h:h*2, :w, 0] = 0
    comparison[h:h*2, :w, 2] = 0
    comparison[h:h*2, w+10:w*2+10, 1] = g2
    comparison[h:h*2, w+10:w*2+10, 0] = 0
    comparison[h:h*2, w+10:w*2+10, 2] = 0
    
    # Red channel row
    comparison[h*2:h*3, :w, 2] = r1
    comparison[h*2:h*3, :w, :2] = 0
    comparison[h*2:h*3, w+10:w*2+10, 2] = r2
    comparison[h*2:h*3, w+10:w*2+10, :2] = 0
    
    # 直接保存，不添加任何文字標籤
    cv2.imwrite(save_path, comparison)
    
    return comparison

def visualize_specific_quadtree_blocks(block_info, original_img, specific_size, save_path):
    """
    創建僅顯示特定大小區塊的quadtree視覺化，保留原圖中特定大小區塊的內容，其他區塊轉為黑色
    
    Parameters:
    -----------
    block_info : dict
        包含區塊資訊的字典，格式為 {'size': {'blocks': [{'position': (y, x), 'size': size}, ...]}}
    original_img : numpy.ndarray
        原始圖像
    specific_size : int
        要特別顯示的區塊大小（例如 16, 32, 64, 128, 256, 512）
    save_path : str
        儲存路徑
        
    Returns:
    --------
    numpy.ndarray
        特定區塊大小視覺化圖像
    """
    # 創建空白圖像 (黑色背景)
    height, width = original_img.shape[:2]
    
    # 判斷是否為彩色圖像
    is_color = len(original_img.shape) == 3
    
    # 創建適當的黑色背景圖像
    if is_color:
        visualization = np.zeros((height, width, 3), dtype=np.uint8)  # 黑色背景
    else:
        visualization = np.zeros((height, width), dtype=np.uint8)  # 黑色背景
    
    # 檢查是否有指定大小的區塊
    size_str = str(specific_size)
    blocks_count = 0
    
    if size_str in block_info:
        blocks = block_info[size_str]['blocks']
        blocks_count = len(blocks)
        
        # 繪製特定大小的區塊，保留原圖內容
        for block in blocks:
            y, x = block['position']
            
            # 複製原圖這個區塊的內容
            visualization[y:y+specific_size, x:x+specific_size] = original_img[y:y+specific_size, x:x+specific_size]
            
            # 添加邊框 (格線)
            border_width = max(1, specific_size // 64)  # 根據區塊大小調整邊框寬度
            
            # 繪製邊框
            if is_color:
                # 彩色圖像
                # 上邊框
                visualization[y:y+border_width, x:x+specific_size] = [255, 255, 0]  # 黃色邊框
                # 下邊框
                visualization[y+specific_size-border_width:y+specific_size, x:x+specific_size] = [255, 255, 0]
                # 左邊框
                visualization[y:y+specific_size, x:x+border_width] = [255, 255, 0]
                # 右邊框
                visualization[y:y+specific_size, x+specific_size-border_width:x+specific_size] = [255, 255, 0]
            else:
                # 灰階圖像
                # 上邊框
                visualization[y:y+border_width, x:x+specific_size] = 255
                # 下邊框
                visualization[y+specific_size-border_width:y+specific_size, x:x+specific_size] = 255
                # 左邊框
                visualization[y:y+specific_size, x:x+border_width] = 255
                # 右邊框
                visualization[y:y+specific_size, x+specific_size-border_width:x+specific_size] = 255
    
    # Directly save the visualization without adding text
    cv2.imwrite(save_path, visualization)
    
    # Return the visualization
    return visualization

def create_all_quadtree_block_visualizations(block_info, original_img, output_dir, stage_num):
    """
    為所有區塊大小創建單獨的視覺化圖像，保留原圖中特定大小區塊的內容
    
    Parameters:
    -----------
    block_info : dict
        包含區塊資訊的字典
    original_img : numpy.ndarray
        原始圖像
    output_dir : str
        輸出目錄
    stage_num : int
        階段編號
    
    Returns:
    --------
    dict
        各區塊大小圖像的路徑字典
    """
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 標準區塊大小列表
    block_sizes = [16, 32, 64, 128, 256, 512, 1024]
    
    # 儲存各圖像路徑
    visualization_paths = {}
    
    # 為每個出現的區塊大小創建視覺化
    for size in block_sizes:
        size_str = str(size)
        if size_str in block_info and len(block_info[size_str]['blocks']) > 0:
            save_path = f"{output_dir}/stage_{stage_num}_blocks_{size}x{size}.png"
            visualize_specific_quadtree_blocks(block_info, original_img, size, save_path)
            visualization_paths[size] = save_path
    
    # 額外創建一個所有區塊的合併視覺化
    combined_path = f"{output_dir}/stage_{stage_num}_all_blocks.png"
    all_blocks_vis = visualize_quadtree(block_info, original_img.shape[:2])  # 只需要形狀
    cv2.imwrite(combined_path, all_blocks_vis)
    visualization_paths['all'] = combined_path
    
    return visualization_paths

def create_all_quadtree_block_visualizations_color(block_info, original_img, output_dir, stage_num, channel_name=None):
    """
    為彩色圖像的所有區塊大小創建單獨的視覺化圖像
    
    Parameters:
    -----------
    block_info : dict
        包含區塊資訊的字典
    original_img : numpy.ndarray
        原始圖像（彩色或對應通道的灰度圖像）
    output_dir : str
        輸出目錄
    stage_num : int
        階段編號
    channel_name : str, optional
        通道名稱（'red', 'green', 'blue' 或 None）
    
    Returns:
    --------
    dict
        各區塊大小圖像的路徑字典
    """
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 標準區塊大小列表
    block_sizes = [16, 32, 64, 128, 256, 512, 1024]
    
    # 儲存各圖像路徑
    visualization_paths = {}
    
    # 為每個出現的區塊大小創建視覺化
    for size in block_sizes:
        size_str = str(size)
        if size_str in block_info and len(block_info[size_str]['blocks']) > 0:
            if channel_name:
                save_path = f"{output_dir}/stage_{stage_num}_{channel_name}_blocks_{size}x{size}.png"
            else:
                save_path = f"{output_dir}/stage_{stage_num}_blocks_{size}x{size}.png"
            
            visualize_specific_quadtree_blocks_color(block_info, original_img, size, save_path, channel_name)
            visualization_paths[size] = save_path
    
    # 額外創建一個所有區塊的合併視覺化
    if channel_name:
        combined_path = f"{output_dir}/stage_{stage_num}_{channel_name}_all_blocks.png"
    else:
        combined_path = f"{output_dir}/stage_{stage_num}_all_blocks.png"
    
    all_blocks_vis = visualize_quadtree(block_info, original_img.shape[:2])
    cv2.imwrite(combined_path, all_blocks_vis)
    visualization_paths['all'] = combined_path
    
    return visualization_paths

def create_difference_histograms(original_img, pred_img, embedded_img, save_dir, method_name, stage_num, local_el=None):
    """
    創建三種差異直方圖視覺化：
    1. 嵌入前的預測誤差直方圖
    2. 移位後的預測誤差直方圖 (模擬直方圖移位過程)
    3. 嵌入後的預測誤差直方圖
    
    Parameters:
    -----------
    original_img : numpy.ndarray
        原始圖像
    pred_img : numpy.ndarray
        預測圖像
    embedded_img : numpy.ndarray
        嵌入後的圖像
    save_dir : str
        儲存目錄
    method_name : str
        使用的方法名稱 (rotation, split, quadtree)
    stage_num : int or str
        階段編號
    local_el : int or numpy.ndarray, optional
        EL值 (用於直方圖移位模擬)，可以是單一值或像素位置的EL陣列
        
    Returns:
    --------
    tuple
        (before_path, shifted_path, after_path, comparison_path) 四個直方圖圖像的路徑
    """
    # 確保輸入是 numpy 數組
    if not isinstance(original_img, np.ndarray):
        original_img = np.array(original_img)
    if not isinstance(pred_img, np.ndarray):
        pred_img = np.array(pred_img)
    if not isinstance(embedded_img, np.ndarray):
        embedded_img = np.array(embedded_img)
    
    # 確保輸出目錄存在
    hist_dir = os.path.join(save_dir, "difference_histograms")
    os.makedirs(hist_dir, exist_ok=True)
    
    # 計算預測誤差 (before embedding)
    pred_error = original_img.astype(np.int16) - pred_img.astype(np.int16)
    
    # 計算嵌入後的誤差 (after embedding)
    embedded_error = embedded_img.astype(np.int16) - pred_img.astype(np.int16)
    
    # 改進的直方圖移位模擬 - 根據實際EL值
    shifted_error = pred_error.copy()
    
    # 決定使用的最大EL值
    if local_el is None:
        # 如果沒有提供EL值，使用預設值5
        max_el = 5
    elif isinstance(local_el, np.ndarray):
        # 如果是陣列，使用平均值
        max_el = int(np.mean(local_el))
    else:
        # 使用提供的單一值
        max_el = int(local_el)
    
    # 記錄使用的EL值（用於標題）
    el_text = f"EL={max_el}"
    
    # 執行更精確的直方圖移位模擬
    # 正值誤差部分: 根據EL範圍移位
    for i in range(max_el):
        # 將差值為i的像素向右移動至i+1
        shifted_error[pred_error == i] = i + 1
    
    # 負值誤差部分: 類似處理，但可選
    # 注意：根據實際算法，是否移動負值誤差要看具體實現
    # 這裡提供一個選項，預設不移動
    shift_negative = False
    if shift_negative:
        for i in range(1, max_el+1):
            shifted_error[pred_error == -i] = -(i + 1)
    
    # 設定直方圖範圍，以確保三個直方圖使用相同的x軸
    error_min = min(pred_error.min(), shifted_error.min(), embedded_error.min())
    error_max = max(pred_error.max(), shifted_error.max(), embedded_error.max())
    
    # 為了更好的視覺效果，將範圍限制在合理區間內
    hist_range = (max(-50, error_min), min(50, error_max))
    bins = hist_range[1] - hist_range[0] + 1
    
    # 創建 "Before Embedding" 直方圖
    plt.figure(figsize=(10, 6))
    plt.hist(pred_error.flatten(), bins=bins, range=hist_range, 
             alpha=0.7, color='blue', density=True)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.title(f"Prediction Error Histogram Before Embedding\nMethod: {method_name.capitalize()}, Stage: {stage_num}")
    plt.xlabel("Prediction Error Value")
    plt.ylabel("Normalized Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    before_path = os.path.join(hist_dir, f"{method_name}_stage{stage_num}_error_before.png")
    plt.savefig(before_path)
    plt.close()
    
    # 創建 "Shifted" 直方圖
    plt.figure(figsize=(10, 6))
    plt.hist(shifted_error.flatten(), bins=bins, range=hist_range, 
             alpha=0.7, color='green', density=True)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.title(f"Prediction Error Histogram After Shifting\nMethod: {method_name.capitalize()}, Stage: {stage_num}, {el_text}")
    plt.xlabel("Prediction Error Value")
    plt.ylabel("Normalized Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    shifted_path = os.path.join(hist_dir, f"{method_name}_stage{stage_num}_error_shifted.png")
    plt.savefig(shifted_path)
    plt.close()
    
    # 創建 "After Embedding" 直方圖
    plt.figure(figsize=(10, 6))
    plt.hist(embedded_error.flatten(), bins=bins, range=hist_range, 
             alpha=0.7, color='purple', density=True)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.title(f"Prediction Error Histogram After Embedding\nMethod: {method_name.capitalize()}, Stage: {stage_num}")
    plt.xlabel("Prediction Error Value")
    plt.ylabel("Normalized Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    after_path = os.path.join(hist_dir, f"{method_name}_stage{stage_num}_error_after.png")
    plt.savefig(after_path)
    plt.close()
    
    # 創建三個直方圖的對比圖
    plt.figure(figsize=(15, 8))
    plt.hist(pred_error.flatten(), bins=bins, range=hist_range, 
             alpha=0.5, color='blue', density=True, label="Before Embedding")
    plt.hist(shifted_error.flatten(), bins=bins, range=hist_range, 
             alpha=0.5, color='green', density=True, label="After Shifting")
    plt.hist(embedded_error.flatten(), bins=bins, range=hist_range, 
             alpha=0.5, color='purple', density=True, label="After Embedding")
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.title(f"Prediction Error Histogram Comparison\nMethod: {method_name.capitalize()}, Stage: {stage_num}, {el_text}")
    plt.xlabel("Prediction Error Value")
    plt.ylabel("Normalized Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    comparison_path = os.path.join(hist_dir, f"{method_name}_stage{stage_num}_error_comparison.png")
    plt.savefig(comparison_path)
    plt.close()
    
    # 返回四個圖像的路徑，方便後續使用
    return before_path, shifted_path, after_path, comparison_path

def create_rotation_method_flowchart(original_img, imgName, method, prediction_method, output_dir):
    """
    創建旋轉方法的完整流程示意圖
    
    Parameters:
    -----------
    original_img : numpy.ndarray
        原始影像
    imgName : str
        影像名稱
    method : str
        方法名稱 ("rotation")
    prediction_method : str
        預測方法名稱
    output_dir : str
        輸出目錄
    """
    
    # 確保輸出目錄存在
    method_dir = f"{output_dir}/image/{imgName}/{method}"
    os.makedirs(method_dir, exist_ok=True)
    
    # 設定圖像參數
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Multi-Round Rotation Embedding Process Flow', fontsize=16, fontweight='bold')
    
    # 創建網格布局
    gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.2)
    
    # 原始影像
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(original_img, cmap='gray')
    ax_orig.set_title('Original Image\n$I_0$', fontsize=12, fontweight='bold')
    ax_orig.axis('off')
    
    # 四個旋轉階段
    rotation_angles = [90, 180, 270, 360]
    stage_positions = [(0, 1), (0, 2), (0, 3), (0, 4)]
    
    for i, (angle, pos) in enumerate(zip(rotation_angles, stage_positions)):
        # 旋轉後的影像
        ax_rot = fig.add_subplot(gs[pos[0], pos[1]])
        
        # 實際旋轉影像
        if angle == 360:
            rotated_img = original_img
            angle_display = 0
        else:
            rotated_img = np.rot90(original_img, k=angle//90)
            angle_display = angle
        
        ax_rot.imshow(rotated_img, cmap='gray')
        ax_rot.set_title(f'Stage {i+1}\nRotate {angle_display}°\n$I_{i+1}^{{rot}}$', 
                        fontsize=10, fontweight='bold')
        ax_rot.axis('off')
    
    # PEE過程視覺化
    pee_row = 1
    for i in range(4):
        ax_pee = fig.add_subplot(gs[pee_row, i+1])
        ax_pee.set_xlim(0, 10)
        ax_pee.set_ylim(0, 10)
        
        # 預測方塊
        pred_box = FancyBboxPatch((1, 7), 8, 2, boxstyle="round,pad=0.1", 
                                 facecolor='lightblue', edgecolor='blue')
        ax_pee.add_patch(pred_box)
        ax_pee.text(5, 8, 'Weighted\nPrediction', ha='center', va='center', fontsize=9)
        
        # 嵌入方塊
        embed_box = FancyBboxPatch((1, 4), 8, 2, boxstyle="round,pad=0.1", 
                                  facecolor='lightgreen', edgecolor='green')
        ax_pee.add_patch(embed_box)
        ax_pee.text(5, 5, f'Data Embedding\n$D_{i+1}$', ha='center', va='center', fontsize=9)
        
        # EL控制方塊
        el_box = FancyBboxPatch((1, 1), 8, 2, boxstyle="round,pad=0.1", 
                               facecolor='lightyellow', edgecolor='orange')
        ax_pee.add_patch(el_box)
        ax_pee.text(5, 2, f'Adaptive EL\n$EL_{i+1}$', ha='center', va='center', fontsize=9)
        
        # 添加箭頭
        ax_pee.annotate('', xy=(5, 6.8), xytext=(5, 7.2),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        ax_pee.annotate('', xy=(5, 3.8), xytext=(5, 4.2),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
        ax_pee.set_title(f'PEE Process\nStage {i+1}', fontsize=10, fontweight='bold')
        ax_pee.axis('off')
    
    # 最終結果（模擬嵌入效果）
    final_row = 2
    for i in range(4):
        ax_final = fig.add_subplot(gs[final_row, i+1])
        
        # 簡單模擬嵌入後的影像
        if rotation_angles[i] == 360:
            final_img = original_img
        else:
            final_img = np.rot90(original_img, k=-(rotation_angles[i]//90))
        
        ax_final.imshow(final_img, cmap='gray')
        ax_final.set_title(f'Embedded Result\n$I_{i+1}$', fontsize=10, fontweight='bold')
        ax_final.axis('off')
    
    # 添加流程箭頭和說明文字
    for i in range(4):
        fig.text(0.35 + i*0.16, 0.67, '↓', fontsize=20, ha='center', color='blue')
        fig.text(0.35 + i*0.16, 0.64, 'PEE', fontsize=8, ha='center', color='blue')
        
        fig.text(0.35 + i*0.16, 0.37, '↓', fontsize=20, ha='center', color='green')
        angle_back = rotation_angles[i] if rotation_angles[i] != 360 else 0
        fig.text(0.35 + i*0.16, 0.34, f'Rotate\n-{angle_back}°', 
                fontsize=8, ha='center', color='green')
    
    # 圖例
    legend_elements = [
        patches.Patch(color='lightblue', label='Prediction Process'),
        patches.Patch(color='lightgreen', label='Data Embedding'),
        patches.Patch(color='lightyellow', label='Embedding Level Control'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))
    
    plt.tight_layout()
    save_path = f"{method_dir}/rotation_embedding_flowchart_{prediction_method}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def create_rotation_prediction_error_analysis(original_img, imgName, method, prediction_method, output_dir):
    """
    創建旋轉方法的預測誤差分析圖
    
    Parameters:
    -----------
    original_img : numpy.ndarray
        原始影像
    imgName : str
        影像名稱
    method : str
        方法名稱
    prediction_method : str
        預測方法名稱
    output_dir : str
        輸出目錄
    """
    
    # 確保輸出目錄存在
    method_dir = f"{output_dir}/image/{imgName}/{method}"
    os.makedirs(method_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Prediction Error Distribution Across Rotation Stages', fontsize=16, fontweight='bold')
    
    rotation_angles = [0, 90, 180, 270]
    
    # 模擬不同階段的最佳化權重
    stage_weights = {
        0: [0.3, 0.3, 0.2, 0.2],    # 初始權重
        90: [0.25, 0.35, 0.25, 0.15], # 90度最佳化後
        180: [0.4, 0.25, 0.2, 0.15],  # 180度最佳化後
        270: [0.2, 0.4, 0.3, 0.1]     # 270度最佳化後
    }
    
    for i, angle in enumerate(rotation_angles):
        # 旋轉影像
        if angle == 0:
            rotated_img = original_img.astype(np.float32)
        else:
            rotated_img = np.rot90(original_img, k=angle//90).astype(np.float32)
        
        # 使用對應的最佳化權重
        weights = stage_weights[angle]
        
        # 計算預測誤差
        pred_error = compute_prediction_error(rotated_img, weights)
        
        # 顯示旋轉後的影像
        axes[0, i].imshow(rotated_img, cmap='gray')
        axes[0, i].set_title(f'Stage {i+1}: Rotated {angle}°', fontsize=12, fontweight='bold')
        axes[0, i].axis('off')
        
        # 顯示預測誤差熱圖
        im = axes[1, i].imshow(pred_error, cmap='RdBu_r', vmin=-15, vmax=15)
        axes[1, i].set_title(f'Prediction Error\nWeights: {weights}', fontsize=10)
        axes[1, i].axis('off')
        
        # 計算可嵌入像素統計
        embedable_pixels = np.sum(np.abs(pred_error) <= 5)  # 假設EL=5
        total_pixels = pred_error.size
        embed_ratio = embedable_pixels / total_pixels * 100
        
        axes[1, i].text(0.02, 0.98, f'Embeddable: {embed_ratio:.1f}%', 
                       transform=axes[1, i].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=9, verticalalignment='top')
    
    # 添加顏色條
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Prediction Error', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    save_path = f"{method_dir}/rotation_prediction_error_{prediction_method}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def compute_prediction_error(img, weights):
    """
    計算預測誤差的輔助函數
    """
    pred_img = np.zeros_like(img)
    h, w = img.shape
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            up = img[y-1, x]
            left = img[y, x-1]
            up_left = img[y-1, x-1]
            up_right = img[y-1, x+1] if x+1 < w else img[y-1, x-1]
            
            pred_img[y, x] = (weights[0]*up + weights[1]*left + 
                            weights[2]*up_left + weights[3]*up_right)
    
    return img - pred_img

def create_rotation_method_flowchart_color(original_img, imgName, method, prediction_method, output_dir):
    """
    創建彩色圖像旋轉方法的完整流程示意圖
    """
    
    # 確保輸出目錄存在
    method_dir = f"{output_dir}/image/{imgName}/{method}"
    os.makedirs(method_dir, exist_ok=True)
    
    # 轉換BGR到RGB以便matplotlib顯示
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # 設定圖像參數
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('Multi-Round Rotation Embedding Process Flow (Color Image)', fontsize=16, fontweight='bold')
    
    # 創建網格布局，為彩色圖像增加更多空間
    gs = fig.add_gridspec(4, 5, hspace=0.4, wspace=0.2)
    
    # 原始影像
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(original_img_rgb)
    ax_orig.set_title('Original Color Image\n$I_0$', fontsize=12, fontweight='bold')
    ax_orig.axis('off')
    
    # 四個旋轉階段
    rotation_angles = [90, 180, 270, 360]
    stage_positions = [(0, 1), (0, 2), (0, 3), (0, 4)]
    
    for i, (angle, pos) in enumerate(zip(rotation_angles, stage_positions)):
        # 旋轉後的影像
        ax_rot = fig.add_subplot(gs[pos[0], pos[1]])
        
        # 實際旋轉影像
        if angle == 360:
            rotated_img_rgb = original_img_rgb
            angle_display = 0
        else:
            rotated_img_rgb = np.rot90(original_img_rgb, k=angle//90)
            angle_display = angle
        
        ax_rot.imshow(rotated_img_rgb)
        ax_rot.set_title(f'Stage {i+1}\nRotate {angle_display}°\n$I_{i+1}^{{rot}}$', 
                        fontsize=10, fontweight='bold')
        ax_rot.axis('off')
    
    # 通道分離處理示意圖
    channel_row = 1
    channel_names = ['Blue', 'Green', 'Red']
    channel_colors = ['blue', 'green', 'red']
    
    for i in range(4):
        ax_channels = fig.add_subplot(gs[channel_row, i+1])
        ax_channels.set_xlim(0, 10)
        ax_channels.set_ylim(0, 12)
        
        # 通道分離方塊
        for j, (ch_name, ch_color) in enumerate(zip(channel_names, channel_colors)):
            ch_box = FancyBboxPatch((0.5, 9-j*3), 9, 2, boxstyle="round,pad=0.1", 
                                   facecolor=ch_color, alpha=0.3, edgecolor=ch_color)
            ax_channels.add_patch(ch_box)
            ax_channels.text(5, 10-j*3, f'{ch_name} Channel\nPEE Processing', 
                           ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax_channels.set_title(f'Channel Processing\nStage {i+1}', fontsize=10, fontweight='bold')
        ax_channels.axis('off')
    
    # PEE過程詳細展示
    pee_row = 2
    for i in range(4):
        ax_pee = fig.add_subplot(gs[pee_row, i+1])
        ax_pee.set_xlim(0, 10)
        ax_pee.set_ylim(0, 10)
        
        # 預測方塊
        pred_box = FancyBboxPatch((1, 7), 8, 2, boxstyle="round,pad=0.1", 
                                 facecolor='lightblue', edgecolor='blue')
        ax_pee.add_patch(pred_box)
        ax_pee.text(5, 8, 'Weighted\nPrediction', ha='center', va='center', fontsize=9)
        
        # 嵌入方塊
        embed_box = FancyBboxPatch((1, 4), 8, 2, boxstyle="round,pad=0.1", 
                                  facecolor='lightgreen', edgecolor='green')
        ax_pee.add_patch(embed_box)
        ax_pee.text(5, 5, f'Data Embedding\n$D_{i+1}$/3 per channel', ha='center', va='center', fontsize=9)
        
        # EL控制方塊
        el_box = FancyBboxPatch((1, 1), 8, 2, boxstyle="round,pad=0.1", 
                               facecolor='lightyellow', edgecolor='orange')
        ax_pee.add_patch(el_box)
        ax_pee.text(5, 2, f'Adaptive EL\n$EL_{i+1}$', ha='center', va='center', fontsize=9)
        
        # 添加箭頭
        ax_pee.annotate('', xy=(5, 6.8), xytext=(5, 7.2),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        ax_pee.annotate('', xy=(5, 3.8), xytext=(5, 4.2),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
        ax_pee.set_title(f'PEE Process\nStage {i+1}', fontsize=10, fontweight='bold')
        ax_pee.axis('off')
    
    # 最終結果
    final_row = 3
    for i in range(4):
        ax_final = fig.add_subplot(gs[final_row, i+1])
        
        # 簡單模擬嵌入後的影像
        if rotation_angles[i] == 360:
            final_img_rgb = original_img_rgb
        else:
            final_img_rgb = np.rot90(original_img_rgb, k=-(rotation_angles[i]//90))
        
        ax_final.imshow(final_img_rgb)
        ax_final.set_title(f'Color Embedded Result\n$I_{i+1}$', fontsize=10, fontweight='bold')
        ax_final.axis('off')
    
    # 替代方案：更精確的位置計算
    for i in range(4):
        # 計算每個階段的準確x位置
        x_pos = 0.35 + (i * 0.16)  # 均勻分佈在0.2到1.0之間
        
        # 第一組箭頭
        fig.text(x_pos, 0.72, '↓', fontsize=16, ha='center', color='purple')
        fig.text(x_pos, 0.70, 'Split\nChannels', fontsize=7, ha='center', color='purple')
        
        # 第二組箭頭
        fig.text(x_pos, 0.52, '↓', fontsize=16, ha='center', color='blue')
        fig.text(x_pos, 0.50, 'PEE', fontsize=8, ha='center', color='blue')
        
        # 第三組箭頭
        angle_back = rotation_angles[i] if rotation_angles[i] != 360 else 0
        fig.text(x_pos, 0.32, '↓', fontsize=16, ha='center', color='green')
        fig.text(x_pos, 0.30, f'Combine &\nRotate -{angle_back}°', 
                fontsize=7, ha='center', color='green')
    
    # 圖例
    legend_elements = [
        patches.Patch(color='lightblue', label='Prediction Process'),
        patches.Patch(color='lightgreen', label='Data Embedding'),
        patches.Patch(color='lightyellow', label='Embedding Level Control'),
        patches.Patch(color='purple', alpha=0.3, label='Channel Processing'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))
    
    plt.tight_layout()
    save_path = f"{method_dir}/rotation_embedding_flowchart_color_{prediction_method}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def create_rotation_prediction_error_analysis_color(original_img, imgName, method, prediction_method, output_dir):
    """
    創建彩色圖像旋轉方法的預測誤差分析圖
    """
    
    # 確保輸出目錄存在
    method_dir = f"{output_dir}/image/{imgName}/{method}"
    os.makedirs(method_dir, exist_ok=True)
    
    # 分離彩色通道
    b_channel, g_channel, r_channel = cv2.split(original_img)
    channels = [b_channel, g_channel, r_channel]
    channel_names = ['Blue', 'Green', 'Red']
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Color Image Prediction Error Analysis Across Rotation Stages', fontsize=16, fontweight='bold')
    
    rotation_angles = [0, 90, 180, 270]
    stage_weights = {
        0: [0.3, 0.3, 0.2, 0.2],
        90: [0.25, 0.35, 0.25, 0.15],
        180: [0.4, 0.25, 0.2, 0.15],
        270: [0.2, 0.4, 0.3, 0.1]
    }
    
    for ch_idx, (channel, ch_name) in enumerate(zip(channels, channel_names)):
        for i, angle in enumerate(rotation_angles):
            # 旋轉通道
            if angle == 0:
                rotated_channel = channel.astype(np.float32)
            else:
                rotated_channel = np.rot90(channel, k=angle//90).astype(np.float32)
            
            # 計算預測誤差
            weights = stage_weights[angle]
            pred_error = compute_prediction_error(rotated_channel, weights)
            
            # 顯示預測誤差熱圖
            im = axes[ch_idx, i].imshow(pred_error, cmap='RdBu_r', vmin=-15, vmax=15)
            
            if ch_idx == 0:  # 只在第一行添加角度標題
                axes[ch_idx, i].set_title(f'Stage {i+1}: {angle}°', fontsize=12, fontweight='bold')
            
            # 在左側添加通道標籤
            if i == 0:
                axes[ch_idx, i].set_ylabel(f'{ch_name}\nChannel', fontsize=12, fontweight='bold')
            
            axes[ch_idx, i].axis('off')
            
            # 計算可嵌入像素統計
            embedable_pixels = np.sum(np.abs(pred_error) <= 5)
            total_pixels = pred_error.size
            embed_ratio = embedable_pixels / total_pixels * 100
            
            axes[ch_idx, i].text(0.02, 0.98, f'{embed_ratio:.1f}%', 
                               transform=axes[ch_idx, i].transAxes, 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                               fontsize=9, verticalalignment='top')
    
    # 添加顏色條
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Prediction Error', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    save_path = f"{method_dir}/rotation_prediction_error_color_{prediction_method}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def compute_prediction_error(img, weights):
    """
    計算預測誤差的輔助函數
    """
    pred_img = np.zeros_like(img)
    h, w = img.shape
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            up = img[y-1, x]
            left = img[y, x-1]
            up_left = img[y-1, x-1]
            up_right = img[y-1, x+1] if x+1 < w else img[y-1, x-1]
            
            pred_img[y, x] = (weights[0]*up + weights[1]*left + 
                            weights[2]*up_left + weights[3]*up_right)
    
    return img - pred_img

# 替换原有的Split可视化函数

def create_split_rotation_effect_grayscale(sub_images, rotations, split_size, block_base, save_path, stage_num=None):
    """
    創建灰階圖像Split方法的旋轉效果視覺化（修復版本）
    
    Parameters:
    -----------
    sub_images : list
        嵌入後但未旋轉回來的子圖像列表
    rotations : list or np.ndarray
        每個子圖像對應的旋轉角度
    split_size : int
        分割大小 (例如: 2 表示 2x2 分割)
    block_base : bool
        True: Block-based分割, False: Quarter-based分割
    save_path : str
        保存路徑
    stage_num : int, optional
        階段編號，用於文件命名
        
    Returns:
    --------
    tuple
        (merged_image, tiled_image) 合成圖像和拼貼圖像
    """
    
    # 確保目錄存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 檢查輸入數據
    if not sub_images or len(sub_images) == 0:
        print("ERROR: No sub_images provided to create_split_rotation_effect_grayscale")
        return None, None
    
    print(f"Processing {len(sub_images)} sub-images for grayscale rotation effect")
    
    # 轉換CuPy數組為NumPy數組
    numpy_sub_images = []
    for i, sub_img in enumerate(sub_images):
        if isinstance(sub_img, cp.ndarray):
            numpy_sub_img = cp.asnumpy(sub_img)
        else:
            numpy_sub_img = np.array(sub_img)
        
        # 確保是uint8類型
        if numpy_sub_img.dtype != np.uint8:
            numpy_sub_img = numpy_sub_img.astype(np.uint8)
        
        numpy_sub_images.append(numpy_sub_img)
        print(f"  Sub-image {i}: shape={numpy_sub_img.shape}, dtype={numpy_sub_img.dtype}")
    
    try:
        # 1. 創建直接合成的圖像（沒旋轉回0度的子圖像直接合成）
        if isinstance(sub_images[0], cp.ndarray):
            merged_effect_img = merge_image_flexible(sub_images, split_size, block_base)
            merged_effect_img = cp.asnumpy(merged_effect_img)
        else:
            cupy_sub_images = [cp.asarray(sub_img) for sub_img in numpy_sub_images]
            merged_effect_img = merge_image_flexible(cupy_sub_images, split_size, block_base)
            merged_effect_img = cp.asnumpy(merged_effect_img)
        
        # 確保合成圖像是正確的數據類型
        if merged_effect_img.dtype != np.uint8:
            merged_effect_img = merged_effect_img.astype(np.uint8)
        
        print(f"Merged image: shape={merged_effect_img.shape}, dtype={merged_effect_img.dtype}")
        
    except Exception as e:
        print(f"ERROR creating merged image: {e}")
        return None, None
    
    try:
        # 2. 創建拼貼畫方式的圖像（所有子圖像排列成網格）
        tiled_image = create_tiled_subimages(numpy_sub_images, split_size)
        
        # 確保拼貼圖像是正確的數據類型
        if tiled_image.dtype != np.uint8:
            tiled_image = tiled_image.astype(np.uint8)
        
        print(f"Tiled image: shape={tiled_image.shape}, dtype={tiled_image.dtype}")
        
    except Exception as e:
        print(f"ERROR creating tiled image: {e}")
        return merged_effect_img, None
    
    # 生成保存路徑
    base_path = save_path.replace('.png', '')
    split_type = 'block' if block_base else 'quarter'
    
    # 保存兩種類型的圖像
    merged_path = f"{base_path}_{split_type}_merged.png"
    tiled_path = f"{base_path}_{split_type}_tiled.png"
    
    # 實際保存圖像並檢查結果
    try:
        success_merged = cv2.imwrite(merged_path, merged_effect_img)
        if success_merged:
            print(f"✓ Split rotation merged image saved: {merged_path}")
        else:
            print(f"✗ Failed to save merged image: {merged_path}")
    except Exception as e:
        print(f"✗ Exception saving merged image: {e}")
        success_merged = False
    
    try:
        success_tiled = cv2.imwrite(tiled_path, tiled_image)
        if success_tiled:
            print(f"✓ Split rotation tiled image saved: {tiled_path}")
        else:
            print(f"✗ Failed to save tiled image: {tiled_path}")
    except Exception as e:
        print(f"✗ Exception saving tiled image: {e}")
        success_tiled = False
    
    # 檢查文件是否真的存在
    if success_merged and os.path.exists(merged_path):
        file_size = os.path.getsize(merged_path)
        print(f"  Merged file exists, size: {file_size} bytes")
    
    if success_tiled and os.path.exists(tiled_path):
        file_size = os.path.getsize(tiled_path)
        print(f"  Tiled file exists, size: {file_size} bytes")
    
    return merged_effect_img, tiled_image

def create_split_rotation_effect_color(channel_sub_images, rotations, split_size, block_base, 
                                     save_dir, stage_num=None):
    """
    創建彩色圖像Split方法的旋轉效果視覺化（各通道以對應顏色顯示）
    
    Parameters:
    -----------
    channel_sub_images : dict
        包含三個通道的子圖像字典 {'blue': [], 'green': [], 'red': []}
    rotations : list or np.ndarray
        每個子圖像對應的旋轉角度
    split_size : int
        分割大小
    block_base : bool
        分割方式
    save_dir : str
        保存目錄
    stage_num : int, optional
        階段編號
        
    Returns:
    --------
    dict
        包含各通道結果的字典
    """
    
    # 確保目錄存在
    os.makedirs(save_dir, exist_ok=True)
    
    channel_names = ['blue', 'green', 'red']
    channel_results = {}
    split_type = 'block' if block_base else 'quarter'
    
    # 處理每個通道
    merged_channels_gray = []  # 用於最終彩色合成的灰度版本
    tiled_channels_gray = []   # 用於最終彩色合成的灰度版本
    
    for ch_idx, ch_name in enumerate(channel_names):
        if ch_name in channel_sub_images:
            sub_images = channel_sub_images[ch_name]

            
            # 轉換為NumPy數組
            numpy_sub_images = []
            for sub_img in sub_images:
                if isinstance(sub_img, cp.ndarray):
                    numpy_sub_images.append(cp.asnumpy(sub_img))
                else:
                    numpy_sub_images.append(np.array(sub_img))
            
            # 1. 創建合成圖像（灰度版本）
            if isinstance(sub_images[0], cp.ndarray):
                channel_merged_gray = merge_image_flexible(sub_images, split_size, block_base)
                channel_merged_gray = cp.asnumpy(channel_merged_gray)
            else:
                cupy_sub_images = [cp.asarray(sub_img) for sub_img in numpy_sub_images]
                channel_merged_gray = merge_image_flexible(cupy_sub_images, split_size, block_base)
                channel_merged_gray = cp.asnumpy(channel_merged_gray)
            
            # 2. 創建拼貼圖像（灰度版本）
            channel_tiled_gray = create_tiled_subimages(numpy_sub_images, split_size)
            
            # 3. 關鍵步驟：轉換為對應顏色的彩色圖像
            channel_merged_colored = convert_single_channel_to_color(channel_merged_gray, ch_name)
            channel_tiled_colored = convert_single_channel_to_color(channel_tiled_gray, ch_name)
            
            # 保存灰度版本用於最終合成
            merged_channels_gray.append(channel_merged_gray)
            tiled_channels_gray.append(channel_tiled_gray)
            
            # 4. 關鍵修復：實際保存彩色版本的單通道結果
            channel_merged_path = os.path.join(save_dir, f"{ch_name}_channel_{split_type}_merged.png")
            channel_tiled_path = os.path.join(save_dir, f"{ch_name}_channel_{split_type}_tiled.png")
            
            # 實際保存圖像（這是之前缺少的部分）
            success_merged = cv2.imwrite(channel_merged_path, channel_merged_colored)
            success_tiled = cv2.imwrite(channel_tiled_path, channel_tiled_colored)
            
            # 檢查保存是否成功
            if success_merged:
                print(f"{ch_name.capitalize()} channel colored merged saved: {channel_merged_path}")
            else:
                print(f"Failed to save {ch_name} channel merged image: {channel_merged_path}")
                
            if success_tiled:
                print(f"{ch_name.capitalize()} channel colored tiled saved: {channel_tiled_path}")
            else:
                print(f"Failed to save {ch_name} channel tiled image: {channel_tiled_path}")
            
            # 存儲結果
            channel_results[f'{ch_name}_merged'] = channel_merged_colored
            channel_results[f'{ch_name}_tiled'] = channel_tiled_colored
            
        else:
            print(f"WARNING: {ch_name} channel not found in channel_sub_images")
    
    # 創建彩色合成圖像（使用灰度版本合成）
    if len(merged_channels_gray) == 3:
        
        # 合成的彩色圖像（合併方式）
        color_merged = combine_color_channels(
            merged_channels_gray[0],  # blue
            merged_channels_gray[1],  # green
            merged_channels_gray[2]   # red
        )
        
        # 合成的彩色圖像（拼貼方式）
        color_tiled = combine_color_channels(
            tiled_channels_gray[0],   # blue
            tiled_channels_gray[1],   # green
            tiled_channels_gray[2]    # red
        )
        
        # 關鍵修復：實際保存彩色合成結果
        color_merged_path = os.path.join(save_dir, f"color_{split_type}_merged.png")
        color_tiled_path = os.path.join(save_dir, f"color_{split_type}_tiled.png")
        
        # 實際保存圖像（這是之前缺少的部分）
        success_color_merged = cv2.imwrite(color_merged_path, color_merged)
        success_color_tiled = cv2.imwrite(color_tiled_path, color_tiled)
        
        # 檢查保存是否成功
        if success_color_merged:
            print(f"Color merged image saved: {color_merged_path}")
        else:
            print(f"Failed to save color merged image: {color_merged_path}")
            
        if success_color_tiled:
            print(f"Color tiled image saved: {color_tiled_path}")
        else:
            print(f"Failed to save color tiled image: {color_tiled_path}")
        
        channel_results['color_merged'] = color_merged
        channel_results['color_tiled'] = color_tiled
        
    else:
        print(f"WARNING: Expected 3 channels but got {len(merged_channels_gray)}")
    
    return channel_results

def create_tiled_subimages(sub_images, split_size, padding=2):
    """
    將子圖像以拼貼畫方式排列成網格
    
    Parameters:
    -----------
    sub_images : list
        子圖像列表
    split_size : int
        分割大小
    padding : int
        子圖像之間的間距（像素）
        
    Returns:
    --------
    numpy.ndarray
        拼貼畫圖像
    """
    if not sub_images:
        return np.array([])
    
    # 獲取子圖像尺寸
    sub_height, sub_width = sub_images[0].shape[:2]
    
    # 計算網格尺寸
    grid_width = split_size * sub_width + (split_size - 1) * padding
    grid_height = split_size * sub_height + (split_size - 1) * padding
    
    # 創建空白畫布
    if len(sub_images[0].shape) == 3:  # 彩色圖像
        tiled_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    else:  # 灰度圖像
        tiled_image = np.zeros((grid_height, grid_width), dtype=np.uint8)
    
    # 將子圖像放置到網格中
    for i in range(split_size):
        for j in range(split_size):
            idx = i * split_size + j
            if idx < len(sub_images):
                # 計算放置位置
                start_y = i * (sub_height + padding)
                end_y = start_y + sub_height
                start_x = j * (sub_width + padding)
                end_x = start_x + sub_width
                
                # 放置子圖像
                tiled_image[start_y:end_y, start_x:end_x] = sub_images[idx]
    
    return tiled_image

def save_split_rotation_effects(pee_stages, method, imgName, output_dir, is_color_image=False):
    """
    批量保存Split方法的所有旋轉效果圖像（修改版，無文字標注）
    
    Parameters:
    -----------
    pee_stages : list
        PEE階段結果列表
    method : str
        方法名稱 ("split")
    imgName : str
        圖像名稱
    output_dir : str
        輸出目錄
    is_color_image : bool
        是否為彩色圖像
    """
    if method != "split":
        return
    
    # 創建Split專用的輸出目錄
    split_effect_dir = os.path.join(output_dir, "image", imgName, "split", "rotation_effects")
    os.makedirs(split_effect_dir, exist_ok=True)
    
    for i, stage in enumerate(pee_stages):
        if 'rotated_sub_images' in stage or 'channel_rotated_sub_images' in stage:
            stage_dir = os.path.join(split_effect_dir, f"stage_{i}")
            os.makedirs(stage_dir, exist_ok=True)
            
            # 獲取旋轉角度資訊
            rotations = stage.get('rotations', [0] * (stage.get('split_size', 2) ** 2))
            split_size = stage.get('split_size', 2)
            block_base = stage.get('block_base', True)
            
            if is_color_image:
                # 彩色圖像處理
                if 'channel_rotated_sub_images' in stage:
                    color_results = create_split_rotation_effect_color(
                        stage['channel_rotated_sub_images'],
                        rotations, split_size, block_base,
                        stage_dir, i
                    )
                    print(f"Created color rotation effects for stage {i}")
            else:
                # 灰階圖像處理
                if 'rotated_sub_images' in stage:
                    grayscale_save_path = os.path.join(stage_dir, "grayscale_rotation_effect.png")
                    merged_img, tiled_img = create_split_rotation_effect_grayscale(
                        stage['rotated_sub_images'],
                        rotations, split_size, block_base,
                        grayscale_save_path, i
                    )
                    print(f"Created grayscale rotation effects for stage {i}")

# 新增: 創建簡潔的比較圖像函數
def create_split_comparison_simple(original_img, merged_img, tiled_img, save_path, split_type):
    """
    創建簡潔的Split方法比較圖像（無文字）
    
    Parameters:
    -----------
    original_img : numpy.ndarray
        原始圖像
    merged_img : numpy.ndarray
        合成的旋轉效果圖像
    tiled_img : numpy.ndarray
        拼貼的旋轉效果圖像
    save_path : str
        保存路徑
    split_type : str
        分割類型 ('block' 或 'quarter')
    """
    # 確保所有圖像尺寸一致（調整tiled_img以匹配原始圖像）
    h, w = original_img.shape[:2]
    
    # 調整拼貼圖像大小以匹配原始圖像
    tiled_resized = cv2.resize(tiled_img, (w, h))
    
    # 水平拼接三張圖像
    if len(original_img.shape) == 3:  # 彩色圖像
        combined = np.hstack([original_img, merged_img, tiled_resized])
    else:  # 灰度圖像
        combined = np.hstack([original_img, merged_img, tiled_resized])
    
    cv2.imwrite(save_path, combined)
    print(f"Split comparison ({split_type}) saved: {save_path}")
    
def convert_single_channel_to_color(single_channel_img, channel_name):
    """
    將單通道圖像轉換為對應顏色的彩色圖像
    
    Parameters:
    -----------
    single_channel_img : numpy.ndarray
        單通道灰度圖像
    channel_name : str
        通道名稱 ('blue', 'green', 'red')
        
    Returns:
    --------
    numpy.ndarray
        對應顏色的彩色圖像 (BGR格式)
    """
    
    # 確保輸入是2D數組
    if len(single_channel_img.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape: {single_channel_img.shape}")
    
    height, width = single_channel_img.shape
    colored_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if channel_name == 'blue':
        # 藍色通道：(B, G, R) = (blue_value, 0, 0)
        colored_img[:, :, 0] = single_channel_img  # B通道
        colored_img[:, :, 1] = 0                   # G通道
        colored_img[:, :, 2] = 0                   # R通道
    elif channel_name == 'green':
        # 綠色通道：(B, G, R) = (0, green_value, 0)
        colored_img[:, :, 0] = 0                   # B通道
        colored_img[:, :, 1] = single_channel_img  # G通道
        colored_img[:, :, 2] = 0                   # R通道
    elif channel_name == 'red':
        # 紅色通道：(B, G, R) = (0, 0, red_value)
        colored_img[:, :, 0] = 0                   # B通道
        colored_img[:, :, 1] = 0                   # G通道
        colored_img[:, :, 2] = single_channel_img  # R通道
    else:
        raise ValueError(f"Unknown channel name: {channel_name}")
    
    return colored_img

def enhance_with_grid_visualization(combined_stage, b_img, g_img, r_img, image_dir, stage_num):
    """
    🎨 增強 with_grid 視覺化：生成三通道彩色版本和組合版本（僅彩色）
    """
    from visualization import convert_single_channel_to_color
    from image_processing import add_grid_lines
    
    # 創建目錄
    with_grid_dir = f"{image_dir}/with_grid"
    colored_grid_dir = f"{with_grid_dir}/colored"
    combined_grid_dir = f"{with_grid_dir}/combined"
    os.makedirs(colored_grid_dir, exist_ok=True)
    os.makedirs(combined_grid_dir, exist_ok=True)
    
    channel_names = ['blue', 'green', 'red']
    channel_imgs = [b_img, g_img, r_img]
    colored_grids = []
    
    # 處理每個通道
    for ch_name, ch_img in zip(channel_names, channel_imgs):
        if ch_name in combined_stage['channel_block_info']:
            block_info = combined_stage['channel_block_info'][ch_name]
            
            # 生成帶網格的灰階圖像
            grid_img_gray = add_grid_lines(ch_img.copy(), block_info)
            
            # 🎨 轉換為對應顏色並保存
            grid_img_colored = convert_single_channel_to_color(grid_img_gray, ch_name)
            colored_grids.append(grid_img_colored)
            
            # 🎨 僅保存彩色網格圖像
            cv2.imwrite(f"{with_grid_dir}/stage_{stage_num}_{ch_name}_grid.png", grid_img_colored)
    
    # 🎨 生成三通道組合的彩色網格圖像
    if len(colored_grids) == 3:
        # 方法1：直接疊加三個彩色通道（創造混合效果）
        combined_overlay = np.zeros_like(colored_grids[0], dtype=np.float32)
        for colored_grid in colored_grids:
            combined_overlay += colored_grid.astype(np.float32)
        combined_overlay = np.clip(combined_overlay / 3, 0, 255).astype(np.uint8)
        
        cv2.imwrite(f"{combined_grid_dir}/stage_{stage_num}_overlay_grid.png", combined_overlay)
        
        # 方法2：使用原始通道組合後再添加統一網格
        from color import combine_color_channels
        combined_img = combine_color_channels(b_img, g_img, r_img)
        
        # 使用藍色通道的block_info作為代表（因為三通道通常有相似的分割）
        if 'blue' in combined_stage['channel_block_info']:
            combined_grid = add_grid_lines_color(combined_img.copy(), combined_stage['channel_block_info']['blue'])
            cv2.imwrite(f"{combined_grid_dir}/stage_{stage_num}_unified_grid.png", combined_grid)

def enhance_block_visualizations(combined_stage, original_img, image_dir, stage_num):
    """
    🎨 增強 block_size_visualizations：生成三通道彩色版本和組合版本
    """
    from visualization import (convert_single_channel_to_color, 
                             visualize_specific_quadtree_blocks)
    from color import split_color_channels, combine_color_channels
    
    # 創建目錄結構
    blocks_viz_dir = f"{image_dir}/block_size_visualizations"
    colored_blocks_dir = f"{blocks_viz_dir}/colored"
    combined_blocks_dir = f"{blocks_viz_dir}/combined"
    
    for ch_name in ['blue', 'green', 'red']:
        ch_colored_dir = f"{colored_blocks_dir}/{ch_name}"
        os.makedirs(ch_colored_dir, exist_ok=True)
    os.makedirs(combined_blocks_dir, exist_ok=True)
    
    # 分離原始圖像通道
    b_orig, g_orig, r_orig = split_color_channels(original_img)
    channel_origs = {'blue': b_orig, 'green': g_orig, 'red': r_orig}
    
    # 標準區塊大小列表
    block_sizes = [16, 32, 64, 128, 256, 512, 1024]
    
    # 為每個區塊大小創建視覺化
    for size in block_sizes:
        size_str = str(size)
        
        # 檢查是否有這個大小的區塊
        has_blocks = any(
            size_str in combined_stage['channel_block_info'][ch] and 
            len(combined_stage['channel_block_info'][ch][size_str]['blocks']) > 0
            for ch in ['blue', 'green', 'red']
            if ch in combined_stage['channel_block_info']
        )
        
        if has_blocks:
            colored_channel_imgs = []
            
            # 處理每個通道
            for ch_name, ch_orig in channel_origs.items():
                if (ch_name in combined_stage['channel_block_info'] and 
                    size_str in combined_stage['channel_block_info'][ch_name] and
                    len(combined_stage['channel_block_info'][ch_name][size_str]['blocks']) > 0):
                    
                    # 生成灰階的區塊視覺化
                    block_viz_gray = visualize_specific_quadtree_blocks(
                        combined_stage['channel_block_info'][ch_name], 
                        ch_orig, size, 
                        f"{blocks_viz_dir}/{ch_name}/stage_{stage_num}_blocks_{size}x{size}.png"
                    )
                    
                    # 🎨 轉換為對應顏色
                    block_viz_colored = convert_single_channel_to_color(block_viz_gray, ch_name)
                    colored_channel_imgs.append(block_viz_colored)
                    
                    # 保存單通道彩色版本
                    colored_save_path = f"{colored_blocks_dir}/{ch_name}/stage_{stage_num}_blocks_{size}x{size}_colored.png"
                    cv2.imwrite(colored_save_path, block_viz_colored)
                    
                else:
                    # 如果該通道沒有這個大小的區塊，創建空的佔位符
                    empty_colored = np.zeros((ch_orig.shape[0], ch_orig.shape[1], 3), dtype=np.uint8)
                    colored_channel_imgs.append(empty_colored)
            
            # 🎨 生成三通道組合的彩色視覺化
            if len(colored_channel_imgs) == 3:
                # 方法1：疊加三個彩色通道
                combined_overlay = np.zeros_like(colored_channel_imgs[0], dtype=np.float32)
                for colored_img in colored_channel_imgs:
                    combined_overlay += colored_img.astype(np.float32)
                combined_overlay = np.clip(combined_overlay, 0, 255).astype(np.uint8)
                
                overlay_path = f"{combined_blocks_dir}/stage_{stage_num}_blocks_{size}x{size}_overlay.png"
                cv2.imwrite(overlay_path, combined_overlay)
                
                # 方法2：使用實際的彩色組合（基於原始圖像的區塊分割）
                # 這需要更複雜的邏輯，暫時使用藍色通道的分割資訊作為代表
                if ('blue' in combined_stage['channel_block_info'] and 
                    size_str in combined_stage['channel_block_info']['blue']):
                    
                    color_blocks_viz = visualize_specific_quadtree_blocks_color(
                        combined_stage['channel_block_info']['blue'], 
                        original_img, size
                    )
                    
                    unified_path = f"{combined_blocks_dir}/stage_{stage_num}_blocks_{size}x{size}_unified.png"
                    cv2.imwrite(unified_path, color_blocks_viz)

def enhance_final_visualizations(pee_stages, final_b_img, final_g_img, final_r_img, image_dir):
    """
    🎨 增強最終結果的視覺化
    """
    from visualization import convert_single_channel_to_color
    from image_processing import add_grid_lines
    from color import combine_color_channels
    
    if not pee_stages:
        return
    
    final_stage = pee_stages[-1]
    
    # 🎨 最終 with_grid 視覺化
    if 'channel_block_info' in final_stage:
        with_grid_dir = f"{image_dir}/with_grid"
        colored_grid_dir = f"{with_grid_dir}/colored"
        combined_grid_dir = f"{with_grid_dir}/combined"
        os.makedirs(colored_grid_dir, exist_ok=True)
        os.makedirs(combined_grid_dir, exist_ok=True)
        
        channel_names = ['blue', 'green', 'red']
        channel_imgs = [final_b_img, final_g_img, final_r_img]
        colored_final_grids = []
        
        for ch_name, ch_img in zip(channel_names, channel_imgs):
            if ch_name in final_stage['channel_block_info']:
                # 生成帶網格的灰階圖像
                final_grid_gray = add_grid_lines(ch_img.copy(), final_stage['channel_block_info'][ch_name])
                
                # 🎨 轉換為對應顏色
                final_grid_colored = convert_single_channel_to_color(final_grid_gray, ch_name)
                colored_final_grids.append(final_grid_colored)
                
                # 保存最終單通道彩色網格
                cv2.imwrite(f"{colored_grid_dir}/final_{ch_name}_grid_colored.png", final_grid_colored)
                
                # 保存原始灰階版本（向後兼容）
                cv2.imwrite(f"{with_grid_dir}/final_{ch_name}_channel_grid.png", final_grid_gray)
        
        # 🎨 生成最終組合彩色網格
        if len(colored_final_grids) == 3:
            # 疊加版本
            final_overlay = np.zeros_like(colored_final_grids[0], dtype=np.float32)
            for colored_grid in colored_final_grids:
                final_overlay += colored_grid.astype(np.float32)
            final_overlay = np.clip(final_overlay / 3, 0, 255).astype(np.uint8)
            cv2.imwrite(f"{combined_grid_dir}/final_overlay_grid.png", final_overlay)
            
            # 統一版本
            final_combined_img = combine_color_channels(final_b_img, final_g_img, final_r_img)
            if 'blue' in final_stage['channel_block_info']:
                final_unified_grid = add_grid_lines_color(final_combined_img.copy(), 
                                                        final_stage['channel_block_info']['blue'])
                cv2.imwrite(f"{combined_grid_dir}/final_unified_grid.png", final_unified_grid)

def add_grid_lines_color(img, block_info):
    """
    為彩色圖像添加網格線的輔助函數
    """
    grid_img = img.copy()
    grid_color = [128, 128, 128]  # 灰色格線
    
    # 線寬設定
    line_widths = {
        1024: 4,
        512: 3,
        256: 3,
        128: 2,
        64: 2,
        32: 1,
        16: 1
    }
    
    # 從大到小處理各個區塊
    for size_str in sorted(block_info.keys(), key=lambda x: int(x), reverse=True):
        size = int(size_str)
        line_width = line_widths.get(size, 1)
        blocks = block_info[size_str]['blocks']
        
        for block in blocks:
            y, x = block['position']
            block_size = block['size']
            
            # 繪製邊框
            for i in range(line_width):
                # 上下邊框
                grid_img[y+i:y+i+1, x:x+block_size] = grid_color
                grid_img[y+block_size-i-1:y+block_size-i, x:x+block_size] = grid_color
                # 左右邊框
                grid_img[y:y+block_size, x+i:x+i+1] = grid_color
                grid_img[y:y+block_size, x+block_size-i-1:x+block_size-i] = grid_color
    
    return grid_img

def visualize_specific_quadtree_blocks_color(block_info, original_color_img, specific_size):
    """
    為彩色圖像創建特定大小區塊的視覺化
    """
    height, width = original_color_img.shape[:2]
    visualization = np.zeros((height, width, 3), dtype=np.uint8)  # 黑色背景
    
    size_str = str(specific_size)
    if size_str in block_info:
        blocks = block_info[size_str]['blocks']
        
        for block in blocks:
            y, x = block['position']
            
            # 複製原圖這個區塊的內容
            visualization[y:y+specific_size, x:x+specific_size] = original_color_img[y:y+specific_size, x:x+specific_size]
            
            # 添加彩色邊框
            border_width = max(1, specific_size // 64)
            border_color = [255, 255, 0]  # 黃色邊框
            
            # 繪製邊框
            visualization[y:y+border_width, x:x+specific_size] = border_color
            visualization[y+specific_size-border_width:y+specific_size, x:x+specific_size] = border_color
            visualization[y:y+specific_size, x:x+border_width] = border_color
            visualization[y:y+specific_size, x+specific_size-border_width:x+specific_size] = border_color
    
    return visualization