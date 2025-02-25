import numpy as np
import struct
import cupy as cp
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import pandas as pd
import os
import time
from datetime import datetime

from common import (
    calculate_psnr,
    calculate_ssim,
    histogram_correlation
)

from image_processing import (
    save_image,
    generate_histogram,
    PredictionMethod
)

def generate_embedding_data(total_embeddings, sub_images_per_stage, max_capacity_per_subimage, 
                           ratio_of_ones=0.5, target_payload_size=-1):
    """
    更靈活的嵌入數據生成函數，不強制平均分配 payload
    
    Parameters:
    -----------
    total_embeddings : int
        總嵌入階段數
    sub_images_per_stage : int
        每個stage的子圖像數量
    max_capacity_per_subimage : int
        每個子圖像的最大容量
    ratio_of_ones : float, optional
        生成數據中1的比例，默認為0.5
    target_payload_size : int, optional
        目標總payload大小，設為-1或0時使用最大容量
        
    Returns:
    --------
    dict
        包含每個stage的數據生成資訊
    """
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
    # 為每個stage生成足夠大的數據，讓它們能夠靈活地達到目標payload
    total_remaining = target_payload_size
    stage_data = []
    
    # 為每個stage分配潛在的最大容量
    potential_capacity_per_stage = max_capacity_per_subimage * sub_images_per_stage
    
    for stage in range(total_embeddings):
        sub_data_list = []
        
        # 為每個子圖像生成數據
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
            'remaining_target': total_remaining  # 記錄當前階段還需要嵌入多少數據
        })
    
    return {
        'stage_data': stage_data,
        'total_target': target_payload_size
    }

# 保留原有函數以保持向後兼容性
def generate_random_binary_array(size, ratio_of_ones=0.5):
    """生成指定大小的随机二进制数组，可调整1的比例"""
    return np.random.choice([0, 1], size=size, p=[1-ratio_of_ones, ratio_of_ones])

def encode_pee_info(pee_info):
    encoded = struct.pack('B', pee_info['total_rotations'])
    for stage in pee_info['stages']:
        for block in stage['block_params']:
            # 编码权重（每个权重使用4位，4个权重共16位）
            weights_packed = sum(w << (4 * i) for i, w in enumerate(block['weights']))
            encoded += struct.pack('>HBH', 
                weights_packed,
                block['EL'],
                block['payload']
            )
    return encoded

def decode_pee_info(encoded_data):
    total_rotations = struct.unpack('B', encoded_data[:1])[0]
    stages = []
    offset = 1

    for _ in range(total_rotations):
        block_params = []
        for _ in range(4):  # 每次旋转有4个块
            weights_packed, EL, payload = struct.unpack('>HBH', encoded_data[offset:offset+5])
            weights = [(weights_packed >> (4 * i)) & 0xF for i in range(4)]
            block_params.append({
                'weights': weights,
                'EL': EL,
                'payload': payload
            })
            offset += 5
        stages.append({'block_params': block_params})

    return {
        'total_rotations': total_rotations,
        'stages': stages
    }

def create_pee_info_table(pee_stages, use_different_weights, total_pixels, split_size, quad_tree=False):
    """
    創建 PEE 資訊表格的完整函數
    
    Parameters:
    -----------
    pee_stages : list
        包含所有 PEE 階段資訊的列表
    use_different_weights : bool
        是否對每個子圖像使用不同的權重
    total_pixels : int
        圖像總像素數
    split_size : int
        分割大小
    quad_tree : bool, optional
        是否使用 quad tree 模式
        
    Returns:
    --------
    PrettyTable
        格式化的表格，包含所有階段的詳細資訊
    """
    table = PrettyTable()
    
    if quad_tree:
        # Quad tree 模式的表格欄位
        table.field_names = [
            "Embedding", "Block Size", "Block Position", "Payload", "BPP",
            "PSNR", "SSIM", "Hist Corr", "Weights", "EL", "Note"
        ]
    else:
        # 標準模式的表格欄位
        table.field_names = [
            "Embedding", "Sub-image", "Payload", "BPP", "PSNR", "SSIM",
            "Hist Corr", "Weights", "EL", "Rotation", "Note"
        ]
    
    # 設置列寬以確保更好的可讀性
    for field in table.field_names:
        table.max_width[field] = 20
    
    for stage in pee_stages:
        # 添加整體 stage 資訊
        if quad_tree:
            # Quad tree 模式的整體資訊
            table.add_row([
                f"{stage['embedding']} (Overall)",
                "-",
                "-",
                stage['payload'],
                f"{stage['bpp']:.4f}",
                f"{stage['psnr']:.2f}",
                f"{stage['ssim']:.4f}",
                f"{stage['hist_corr']:.4f}",
                "-",
                "-",
                "Stage Summary"
            ])
        else:
            # 標準模式的整體資訊
            table.add_row([
                f"{stage['embedding']} (Overall)",
                "-",
                stage['payload'],
                f"{stage['bpp']:.4f}",
                f"{stage['psnr']:.2f}",
                f"{stage['ssim']:.4f}",
                f"{stage['hist_corr']:.4f}",
                "-",
                "-",
                "-",
                "Stage Summary"
            ])
        
        # 添加分隔線
        table.add_row(["-" * 5] * len(table.field_names))
        
        if quad_tree:
            # 處理 quad tree 模式的區塊資訊
            for size in sorted(stage['block_info'].keys(), key=int, reverse=True):
                blocks = stage['block_info'][size]['blocks']
                for block in blocks:
                    block_pixels = block['size'] * block['size']
                    
                    # 處理權重顯示
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
        else:
            # 處理標準模式的區塊資訊
            total_blocks = split_size * split_size
            sub_image_pixels = total_pixels // total_blocks
            
            for i, block in enumerate(stage['block_params']):
                # 處理權重顯示，考慮不同預測方法的情況
                weights_display = (
                    "N/A" if block['weights'] == "N/A"
                    else ", ".join([f"{w:.2f}" for w in block['weights']]) if block['weights']
                    else "-"
                )
                
                table.add_row([
                    stage['embedding'] if i == 0 else "",
                    i,
                    block['payload'],
                    f"{block['payload'] / sub_image_pixels:.4f}",
                    f"{block['psnr']:.2f}",
                    f"{block['ssim']:.4f}",
                    f"{block['hist_corr']:.4f}",
                    weights_display,
                    block['EL'],
                    f"{block['rotation']}°",
                    "Different weights" if use_different_weights else ""
                ])
        
        # 添加分隔線
        table.add_row(["-" * 5] * len(table.field_names))
    
    return table

def analyze_and_plot_results(bpp_psnr_data, imgName, split_size):
    """
    分析結果並繪製圖表
    """
    plt.figure(figsize=(12, 8))
    
    # 繪製 BPP-PSNR 曲線
    bpps = [data['bpp'] for data in bpp_psnr_data]
    psnrs = [data['psnr'] for data in bpp_psnr_data]
    
    plt.plot(bpps, psnrs, 'b.-', linewidth=2, markersize=8, 
             label=f'Split Size: {split_size}x{split_size}')
    
    plt.xlabel('Bits Per Pixel (BPP)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title(f'BPP-PSNR Curve for {imgName}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 添加數據標籤
    for i, (bpp, psnr) in enumerate(zip(bpps, psnrs)):
        plt.annotate(f'Stage {i}\n({bpp:.3f}, {psnr:.2f})',
                    (bpp, psnr), textcoords="offset points",
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    return plt.gcf()

def calculate_metrics_with_rotation(original_img, stage_img, current_rotation):
    """
    計算考慮旋轉的圖像品質指標
    
    Parameters:
    -----------
    original_img : numpy.ndarray
        原始圖像
    stage_img : numpy.ndarray
        處理後的圖像
    current_rotation : int
        當前旋轉角度（度數）
    
    Returns:
    --------
    tuple
        (psnr, ssim, hist_corr) 三個品質指標
    """
    # 確保輸入是numpy array
    if isinstance(original_img, cp.ndarray):
        original_img = cp.asnumpy(original_img)
    if isinstance(stage_img, cp.ndarray):
        stage_img = cp.asnumpy(stage_img)
    
    # 如果需要旋轉，將圖像旋轉回原始方向
    if current_rotation != 0:
        k = (-current_rotation // 90) % 4
        stage_img = np.rot90(stage_img, k=k)
    
    # 計算各種指標
    psnr = calculate_psnr(original_img, stage_img)
    ssim = calculate_ssim(original_img, stage_img)
    hist_corr = histogram_correlation(
        np.histogram(original_img, bins=256, range=(0, 255))[0],
        np.histogram(stage_img, bins=256, range=(0, 255))[0]
    )
    
    return psnr, ssim, hist_corr

def generate_interval_statistics(original_img, stages, total_payload, segments=15):
    """
    根據總嵌入容量生成均勻分布的統計數據表格
    
    Parameters:
    -----------
    original_img : numpy.ndarray
        原始圖像
    stages : list
        包含嵌入各階段資訊的列表
    total_payload : int
        總嵌入容量
    segments : int
        要生成的數據點數量，默認為15
        
    Returns:
    --------
    tuple
        (DataFrame, PrettyTable) 包含統計數據的DataFrame和格式化表格
    """
    # 確保輸入數據類型正確
    if isinstance(original_img, cp.ndarray):
        original_img = cp.asnumpy(original_img)
    
    # 總像素數用於計算BPP
    total_pixels = original_img.size
    
    # 計算間隔和數據點
    if total_payload <= 0:
        print("Warning: Total payload is zero or negative. No statistics generated.")
        return None, None
        
    # 確保至少有2個段
    segments = max(2, min(segments, total_payload))
    
    # 計算每個數據點的嵌入量
    payload_interval = total_payload / segments
    payload_points = [int(i * payload_interval) for i in range(1, segments + 1)]
    
    # 最後一個點確保是總嵌入量
    payload_points[-1] = total_payload
    
    # 初始化結果數據
    results = []
    
    # 累計各階段的嵌入量
    accumulated_payload = 0
    current_stage_index = 0
    current_stage_img = None
    
    for target_payload in payload_points:
        # 模擬嵌入到目標嵌入量的圖像狀態
        while accumulated_payload < target_payload and current_stage_index < len(stages):
            current_stage = stages[current_stage_index]
            stage_payload = current_stage['payload']
            current_stage_img = cp.asnumpy(current_stage['stage_img'])
            
            if accumulated_payload + stage_payload <= target_payload:
                # 完整包含當前階段
                accumulated_payload += stage_payload
                current_stage_index += 1
            else:
                # 部分包含當前階段 - 需要進行插值
                # 注意：這裡使用線性插值來估計PSNR和SSIM，實際上可能需要更精確的模擬
                break
        
        # 確保current_stage_img不為None
        if current_stage_img is None and current_stage_index > 0:
            current_stage_img = cp.asnumpy(stages[current_stage_index-1]['stage_img'])
        elif current_stage_img is None:
            print("Warning: No valid stage image found.")
            continue
            
        # 計算性能指標
        psnr = calculate_psnr(original_img, current_stage_img)
        ssim = calculate_ssim(original_img, current_stage_img)
        hist_corr = histogram_correlation(
            np.histogram(original_img, bins=256, range=(0, 255))[0],
            np.histogram(current_stage_img, bins=256, range=(0, 255))[0]
        )
        bpp = target_payload / total_pixels
        
        # 添加到結果列表
        results.append({
            'Payload': target_payload,
            'BPP': bpp,
            'PSNR': psnr,
            'SSIM': ssim,
            'Hist_Corr': hist_corr
        })
    
    # 創建DataFrame
    df = pd.DataFrame(results)
    
    # 創建PrettyTable
    table = PrettyTable()
    table.field_names = ["Payload", "BPP", "PSNR", "SSIM", "Hist_Corr"]
    
    for result in results:
        table.add_row([
            result['Payload'],
            f"{result['BPP']:.6f}",
            f"{result['PSNR']:.2f}",
            f"{result['SSIM']:.4f}",
            f"{result['Hist_Corr']:.4f}"
        ])
    
    return df, table

def save_interval_statistics(df, imgName, method, prediction_method):
    """
    保存統計數據到CSV和NPY文件
    
    Parameters:
    -----------
    df : pandas.DataFrame
        包含統計數據的DataFrame
    imgName : str
        圖像名稱
    method : str
        使用的方法
    prediction_method : str
        使用的預測方法
    """
    # 保存CSV
    csv_path = f"./pred_and_QR/outcome/plots/{imgName}/interval_stats_{method}_{prediction_method}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Interval statistics saved to {csv_path}")
    
    # 保存NPY
    npy_path = f"./pred_and_QR/outcome/plots/{imgName}/interval_stats_{method}_{prediction_method}.npy"
    np.save(npy_path, df.to_dict('records'))
    print(f"Interval statistics saved to {npy_path}")
    
    return csv_path, npy_path

def plot_interval_statistics(df, imgName, method, prediction_method):
    """
    繪製統計數據曲線圖
    
    Parameters:
    -----------
    df : pandas.DataFrame
        包含統計數據的DataFrame
    imgName : str
        圖像名稱
    method : str
        使用的方法
    prediction_method : str
        使用的預測方法
        
    Returns:
    --------
    tuple
        (fig_psnr, fig_ssim) PSNR和SSIM圖表
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline
    import numpy as np
    
    # 將數據排序
    df_sorted = df.sort_values('BPP')
    
    # 繪製BPP-PSNR曲線
    fig_psnr = plt.figure(figsize=(10, 6))
    
    x = df_sorted['BPP'].values
    y = df_sorted['PSNR'].values
    
    # 如果數據點足夠多，則使用平滑曲線
    if len(x) > 3:
        # 創建平滑曲線 - 增加插值點數量
        x_smooth = np.linspace(min(x), max(x), 300)
        try:
            # 使用三次樣條插值
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(x_smooth)
            
            # 繪製平滑曲線
            plt.plot(x_smooth, y_smooth, 
                     color='blue',
                     linewidth=2.5,
                     label=f'Method: {method}, Predictor: {prediction_method}')
        except:
            # 如果插值失敗，則退回到簡單的平滑方法
            print(f"Warning: Spline interpolation failed. Using basic smoothing.")
            plt.plot(x, y, 
                     color='blue',
                     linewidth=2.5,
                     label=f'Method: {method}, Predictor: {prediction_method}')
    else:
        # 數據點太少，直接連線
        plt.plot(x, y, 
                 color='blue',
                 linewidth=2.5,
                 label=f'Method: {method}, Predictor: {prediction_method}')
    
    # 在原始數據點添加標記
    plt.scatter(x, y, 
               color='blue',
               marker='o',
               s=80, 
               alpha=0.7)
    
    # 添加數據標籤
    for i, (bpp, psnr) in enumerate(zip(x, y)):
        if i % 3 == 0 or i == len(x) - 1:  # 只標記部分點，避免擁擠
            plt.annotate(f'({bpp:.4f}, {psnr:.2f})',
                        (bpp, psnr), 
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                 fc='yellow', 
                                 alpha=0.3),
                        fontsize=8)
    
    plt.xlabel('Bits Per Pixel (BPP)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title(f'BPP-PSNR Curve for {imgName}\nMethod: {method}, Predictor: {prediction_method}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 保存圖表
    plt.savefig(f"./pred_and_QR/outcome/plots/{imgName}/bpp_psnr_{method}_{prediction_method}.png", 
               dpi=300, bbox_inches='tight')
    
    # 繪製BPP-SSIM曲線
    fig_ssim = plt.figure(figsize=(10, 6))
    
    x = df_sorted['BPP'].values
    y = df_sorted['SSIM'].values
    
    # 如果數據點足夠多，則使用平滑曲線
    if len(x) > 3:
        # 創建平滑曲線 - 增加插值點數量
        x_smooth = np.linspace(min(x), max(x), 300)
        try:
            # 使用三次樣條插值
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(x_smooth)
            
            # 繪製平滑曲線
            plt.plot(x_smooth, y_smooth, 
                     color='red',
                     linewidth=2.5,
                     label=f'Method: {method}, Predictor: {prediction_method}')
        except:
            # 如果插值失敗，則退回到簡單的平滑方法
            print(f"Warning: Spline interpolation failed. Using basic smoothing.")
            plt.plot(x, y, 
                     color='red',
                     linewidth=2.5,
                     label=f'Method: {method}, Predictor: {prediction_method}')
    else:
        # 數據點太少，直接連線
        plt.plot(x, y, 
                 color='red',
                 linewidth=2.5,
                 label=f'Method: {method}, Predictor: {prediction_method}')
    
    # 在原始數據點添加標記
    plt.scatter(x, y, 
               color='red',
               marker='o',
               s=80, 
               alpha=0.7)
    
    # 添加數據標籤
    for i, (bpp, ssim) in enumerate(zip(x, y)):
        if i % 3 == 0 or i == len(x) - 1:  # 只標記部分點，避免擁擠
            plt.annotate(f'({bpp:.4f}, {ssim:.4f})',
                        (bpp, ssim), 
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                 fc='yellow', 
                                 alpha=0.3),
                        fontsize=8)
    
    plt.xlabel('Bits Per Pixel (BPP)', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.title(f'BPP-SSIM Curve for {imgName}\nMethod: {method}, Predictor: {prediction_method}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 保存圖表
    plt.savefig(f"./pred_and_QR/outcome/plots/{imgName}/bpp_ssim_{method}_{prediction_method}.png", 
               dpi=300, bbox_inches='tight')
    
    # 如果有直方圖相關性數據，也繪製相應的曲線圖
    if 'Hist_Corr' in df.columns:
        fig_hist = plt.figure(figsize=(10, 6))
        
        x = df_sorted['BPP'].values
        y = df_sorted['Hist_Corr'].values
        
        # 如果數據點足夠多，則使用平滑曲線
        if len(x) > 3:
            # 創建平滑曲線 - 增加插值點數量
            x_smooth = np.linspace(min(x), max(x), 300)
            try:
                # 使用三次樣條插值
                spl = make_interp_spline(x, y, k=3)
                y_smooth = spl(x_smooth)
                
                # 繪製平滑曲線
                plt.plot(x_smooth, y_smooth, 
                         color='green',
                         linewidth=2.5,
                         label=f'Method: {method}, Predictor: {prediction_method}')
            except:
                # 如果插值失敗，則退回到簡單的平滑方法
                print(f"Warning: Spline interpolation failed. Using basic smoothing.")
                plt.plot(x, y, 
                         color='green',
                         linewidth=2.5,
                         label=f'Method: {method}, Predictor: {prediction_method}')
        else:
            # 數據點太少，直接連線
            plt.plot(x, y, 
                     color='green',
                     linewidth=2.5,
                     label=f'Method: {method}, Predictor: {prediction_method}')
        
        # 在原始數據點添加標記
        plt.scatter(x, y, 
                   color='green',
                   marker='o',
                   s=80, 
                   alpha=0.7)
        
        plt.xlabel('Bits Per Pixel (BPP)', fontsize=12)
        plt.ylabel('Histogram Correlation', fontsize=12)
        plt.title(f'BPP-Histogram Correlation Curve for {imgName}\nMethod: {method}, Predictor: {prediction_method}', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # 保存圖表
        plt.savefig(f"./pred_and_QR/outcome/plots/{imgName}/bpp_histcorr_{method}_{prediction_method}.png", 
                   dpi=300, bbox_inches='tight')
        
        return fig_psnr, fig_ssim, fig_hist
    
    return fig_psnr, fig_ssim

def run_multiple_predictors(imgName, filetype="png", method="quadtree", total_embeddings=5, 
                           predictor_ratios=None, el_mode=0, use_different_weights=False,
                           split_size=2, block_base=False, 
                           quad_tree_params=None, stats_segments=15):
    """
    自動運行多種預測方法並生成比較結果
    
    Parameters:
    -----------
    imgName : str
        圖像名稱
    filetype : str
        圖像檔案類型
    method : str
        使用的方法（"rotation", "split", "quadtree"）
    total_embeddings : int
        總嵌入次數
    predictor_ratios : dict
        各預測器的ratio_of_ones設置，鍵為預測器名稱，值為ratio值
    el_mode : int
        EL模式 (0:無限制, 1:漸增, 2:漸減)
    use_different_weights : bool
        是否對每個子圖像使用不同的權重
    split_size : int
        分割大小
    block_base : bool
        是否使用 block-base 分割
    quad_tree_params : dict
        四叉樹參數
    stats_segments : int
        統計分段數量
    """
    from embedding import (
        pee_process_with_rotation_cuda,
        pee_process_with_split_cuda
    )
    from quadtree import pee_process_with_quadtree_cuda
    import time
    from datetime import datetime
    import pandas as pd
    import os
    import cupy as cp
    import numpy as np
    from image_processing import save_image, generate_histogram, PredictionMethod
    from common import calculate_psnr, calculate_ssim, histogram_correlation
    import cv2
    
    # 設置默認的預測器ratio字典
    if predictor_ratios is None:
        predictor_ratios = {
            "PROPOSED": 0.5,
            "MED": 1.0,
            "GAP": 1.0,
            "RHOMBUS": 1.0
        }
    
    # 創建必要的目錄
    comparison_dir = f"./pred_and_QR/outcome/plots/{imgName}/comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 預測方法列表
    prediction_methods = [
        PredictionMethod.PROPOSED,
        PredictionMethod.MED,
        PredictionMethod.GAP,
        PredictionMethod.RHOMBUS
    ]
    
    # 儲存所有預測方法的統計數據
    all_stats = {}
    all_results = {}
    
    # 記錄開始時間
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 創建記錄檔案
    log_file = f"{comparison_dir}/multi_predictor_run_{timestamp}.log"
    with open(log_file, 'w') as f:
        f.write(f"Multi-predictor comparison run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Image: {imgName}.{filetype}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Parameters: embeddings={total_embeddings}, el_mode={el_mode}\n")
        f.write("Predictor ratio settings:\n")
        for pred, ratio in predictor_ratios.items():
            f.write(f"  {pred}: {ratio}\n")
            
        if method == "quadtree":
            f.write(f"Quadtree params: min_block_size={quad_tree_params['min_block_size']}, variance_threshold={quad_tree_params['variance_threshold']}\n")
        else:
            f.write(f"Split size: {split_size}x{split_size}, block_base={block_base}\n")
        f.write("\n" + "="*80 + "\n\n")
    
    # 讀取原始圖像 - 只需要讀取一次
    origImg = cv2.imread(f"./pred_and_QR/image/{imgName}.{filetype}", cv2.IMREAD_GRAYSCALE)
    if origImg is None:
        raise ValueError(f"Failed to read image: ./pred_and_QR/image/{imgName}.{filetype}")
    origImg = np.array(origImg).astype(np.uint8)
    total_pixels = origImg.size
    
    # 依次運行每種預測方法
    for prediction_method in prediction_methods:
        method_name = prediction_method.value.upper()
        print(f"\n{'='*80}")
        print(f"Running with {method_name.lower()} predictor...")
        print(f"{'='*80}\n")
        
        # 記錄到日誌
        with open(log_file, 'a') as f:
            f.write(f"Starting run with {method_name.lower()} predictor at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 重置 GPU 記憶體
        cp.get_default_memory_pool().free_all_blocks()
        
        # 針對 MED、GAP 和 RHOMBUS，強制設置 use_different_weights = False
        current_use_weights = use_different_weights
        if prediction_method in [PredictionMethod.MED, PredictionMethod.GAP, PredictionMethod.RHOMBUS]:
            current_use_weights = False
            print(f"Note: Weight optimization disabled for {method_name.lower()} prediction method")
        
        # 獲取當前預測器的ratio_of_ones
        current_ratio_of_ones = predictor_ratios.get(method_name, 0.5)
        print(f"Using ratio_of_ones = {current_ratio_of_ones} for {method_name.lower()} predictor")
        
        try:
            # 執行選定的方法
            if method == "rotation":
                final_pee_img, total_payload, pee_stages = pee_process_with_rotation_cuda(
                    origImg,
                    total_embeddings,
                    current_ratio_of_ones,  # 使用當前預測器的ratio_of_ones
                    current_use_weights,
                    split_size,
                    el_mode,
                    prediction_method=prediction_method,
                    target_payload_size=-1  # 使用最大嵌入量
                )
            elif method == "split":
                final_pee_img, total_payload, pee_stages = pee_process_with_split_cuda(
                    origImg,
                    total_embeddings,
                    current_ratio_of_ones,  # 使用當前預測器的ratio_of_ones
                    current_use_weights,
                    split_size,
                    el_mode,
                    block_base,
                    prediction_method=prediction_method,
                    target_payload_size=-1  # 使用最大嵌入量
                )
            elif method == "quadtree":
                final_pee_img, total_payload, pee_stages = pee_process_with_quadtree_cuda(
                    origImg,
                    total_embeddings,
                    current_ratio_of_ones,  # 使用當前預測器的ratio_of_ones
                    current_use_weights,
                    quad_tree_params['min_block_size'],
                    quad_tree_params['variance_threshold'],
                    el_mode,
                    rotation_mode='random',
                    prediction_method=prediction_method,
                    target_payload_size=-1  # 使用最大嵌入量
                )
            
            # 生成統計數據
            print("\nGenerating statistics...")
            stats_df, stats_table = generate_interval_statistics(
                origImg, pee_stages, total_payload, segments=stats_segments
            )
            
            # 保存統計數據
            if stats_df is not None:
                # 添加預測器和ratio_of_ones資訊到DataFrame
                stats_df['Predictor'] = method_name.lower()
                stats_df['Ratio_of_Ones'] = current_ratio_of_ones
                
                # 記錄統計數據
                all_stats[method_name.lower()] = stats_df
                
                # 保存為CSV
                csv_path = f"{comparison_dir}/{method_name.lower()}_stats.csv"
                stats_df.to_csv(csv_path, index=False)
                print(f"Statistics saved to {csv_path}")
                
                # 將結果添加到字典中
                final_psnr = calculate_psnr(origImg, final_pee_img)
                final_ssim = calculate_ssim(origImg, final_pee_img)
                hist_orig = generate_histogram(origImg)
                hist_final = generate_histogram(final_pee_img)
                final_hist_corr = histogram_correlation(hist_orig, hist_final)
                
                all_results[method_name.lower()] = {
                    'predictor': method_name.lower(),
                    'ratio_of_ones': current_ratio_of_ones,
                    'total_payload': total_payload,
                    'bpp': total_payload / total_pixels,
                    'psnr': final_psnr,
                    'ssim': final_ssim,
                    'hist_corr': final_hist_corr
                }
                
                # 記錄到日誌
                with open(log_file, 'a') as f:
                    f.write(f"Run completed for {method_name.lower()} predictor (ratio_of_ones={current_ratio_of_ones})\n")
                    f.write(f"Total payload: {total_payload}\n")
                    f.write(f"PSNR: {final_psnr:.2f}\n")
                    f.write(f"SSIM: {final_ssim:.4f}\n")
                    f.write(f"Processing time: {time.time() - start_time:.2f} seconds\n")
                    f.write("\n" + "-"*60 + "\n\n")
                
                # 保存處理後的圖像
                save_image(final_pee_img, 
                          f"{comparison_dir}/{method_name.lower()}_final.png")
            
            else:
                print(f"Warning: No statistics generated for {method_name.lower()}")
                with open(log_file, 'a') as f:
                    f.write(f"Warning: No statistics generated for {method_name.lower()}\n\n")
        
        except Exception as e:
            print(f"Error processing {method_name.lower()}: {str(e)}")
            with open(log_file, 'a') as f:
                f.write(f"Error processing {method_name.lower()}: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
                f.write("\n\n")
    
    # 所有預測方法處理完成後，創建比較結果
    if all_stats:
        try:
            # 創建結果摘要表
            results_df = pd.DataFrame(list(all_results.values()))
            results_df = results_df[['predictor', 'ratio_of_ones', 'total_payload', 'bpp', 'psnr', 'ssim', 'hist_corr']]
            
            # 保存結果摘要
            results_csv = f"{comparison_dir}/summary_results.csv"
            results_df.to_csv(results_csv, index=False)
            
            # 創建統一的折線圖
            create_unified_plots(all_stats, imgName, method, comparison_dir)
            
            # 記錄到日誌
            with open(log_file, 'a') as f:
                f.write(f"\nComparison completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total processing time: {time.time() - start_time:.2f} seconds\n")
                f.write("\nSummary Results:\n")
                f.write(results_df.to_string())
            
            print(f"\nComparison completed and saved to {comparison_dir}")
            print(f"Total processing time: {time.time() - start_time:.2f} seconds")
            print("\nSummary Results:")
            print(results_df)
            
            return results_df, all_stats
        
        except Exception as e:
            print(f"Error generating comparison: {str(e)}")
            with open(log_file, 'a') as f:
                f.write(f"\nError generating comparison: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
    
    else:
        print("No valid statistics available for comparison")
        with open(log_file, 'a') as f:
            f.write("\nNo valid statistics available for comparison\n")
    
    return None, None

# 修改utils.py中的create_unified_plots函數，從曲線圖改回線性折線圖

def create_unified_plots(all_stats, imgName, method, output_dir):
    """
    創建統一的線性折線圖，顯示所有預測方法的比較
    
    Parameters:
    -----------
    all_stats : dict
        字典，包含各預測方法的統計數據框
    imgName : str
        圖像名稱
    method : str
        使用的方法
    output_dir : str
        輸出目錄
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not all_stats:
        print("No data available for plotting")
        return
    
    # 設置不同預測方法的顏色和標記
    colors = {
        'proposed': 'blue',
        'med': 'red',
        'gap': 'green',
        'rhombus': 'purple'
    }
    
    markers = {
        'proposed': 'o',
        'med': 's',
        'gap': '^',
        'rhombus': 'D'
    }
    
    # 創建統一的BPP-PSNR折線圖
    plt.figure(figsize=(12, 8))
    
    for predictor, df in all_stats.items():
        # 將數據排序為升序
        df_sorted = df.sort_values('BPP')
        x = df_sorted['BPP'].values
        y = df_sorted['PSNR'].values
        
        # 使用直接連接的折線圖（而非曲線圖）
        plt.plot(x, y, 
                 color=colors.get(predictor, 'black'),
                 linewidth=2.5,
                 marker=markers.get(predictor, 'x'),
                 markersize=8,
                 label=f'Predictor: {predictor}')
    
    plt.xlabel('Bits Per Pixel (BPP)', fontsize=14)
    plt.ylabel('PSNR (dB)', fontsize=14)
    plt.title(f'Comparison of Different Predictors\nMethod: {method}, Image: {imgName}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/unified_bpp_psnr.png", dpi=300)
    
    # 創建統一的BPP-SSIM折線圖
    plt.figure(figsize=(12, 8))
    
    for predictor, df in all_stats.items():
        # 將數據排序為升序
        df_sorted = df.sort_values('BPP')
        x = df_sorted['BPP'].values
        y = df_sorted['SSIM'].values
        
        # 使用直接連接的折線圖（而非曲線圖）
        plt.plot(x, y, 
                 color=colors.get(predictor, 'black'),
                 linewidth=2.5,
                 marker=markers.get(predictor, 'x'),
                 markersize=8,
                 label=f'Predictor: {predictor}')
    
    plt.xlabel('Bits Per Pixel (BPP)', fontsize=14)
    plt.ylabel('SSIM', fontsize=14)
    plt.title(f'Comparison of Different Predictors (SSIM)\nMethod: {method}, Image: {imgName}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/unified_bpp_ssim.png", dpi=300)
    
    # 創建統一的直方圖相關性折線圖
    if 'Hist_Corr' in next(iter(all_stats.values())).columns:
        plt.figure(figsize=(12, 8))
        
        for predictor, df in all_stats.items():
            # 將數據排序為升序
            df_sorted = df.sort_values('BPP')
            x = df_sorted['BPP'].values
            y = df_sorted['Hist_Corr'].values
            
            # 使用直接連接的折線圖（而非曲線圖）
            plt.plot(x, y, 
                     color=colors.get(predictor, 'black'),
                     linewidth=2.5,
                     marker=markers.get(predictor, 'x'),
                     markersize=8,
                     label=f'Predictor: {predictor}')
        
        plt.xlabel('Bits Per Pixel (BPP)', fontsize=14)
        plt.ylabel('Histogram Correlation', fontsize=14)
        plt.title(f'Comparison of Different Predictors (Histogram Correlation)\nMethod: {method}, Image: {imgName}', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/unified_bpp_histcorr.png", dpi=300)
    
    # 創建合併的CSV統計數據
    create_combined_csv(all_stats, output_dir)

# 修改plot_interval_statistics函數，從曲線圖改回線性折線圖
def plot_interval_statistics(df, imgName, method, prediction_method):
    """
    繪製統計數據線性折線圖
    
    Parameters:
    -----------
    df : pandas.DataFrame
        包含統計數據的DataFrame
    imgName : str
        圖像名稱
    method : str
        使用的方法
    prediction_method : str
        使用的預測方法
        
    Returns:
    --------
    tuple
        (fig_psnr, fig_ssim) PSNR和SSIM圖表
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 將數據排序
    df_sorted = df.sort_values('BPP')
    
    # 繪製BPP-PSNR折線圖
    fig_psnr = plt.figure(figsize=(10, 6))
    
    x = df_sorted['BPP'].values
    y = df_sorted['PSNR'].values
    
    # 使用直接連接的折線圖（而非曲線圖）
    plt.plot(x, y, 
             color='blue',
             linewidth=2.5,
             marker='o',
             markersize=8,
             label=f'Method: {method}, Predictor: {prediction_method}')
    
    # 添加數據標籤
    for i, (bpp, psnr) in enumerate(zip(x, y)):
        if i % 3 == 0 or i == len(x) - 1:  # 只標記部分點，避免擁擠
            plt.annotate(f'({bpp:.4f}, {psnr:.2f})',
                        (bpp, psnr), 
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                 fc='yellow', 
                                 alpha=0.3),
                        fontsize=8)
    
    plt.xlabel('Bits Per Pixel (BPP)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title(f'BPP-PSNR Curve for {imgName}\nMethod: {method}, Predictor: {prediction_method}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 保存圖表
    plt.savefig(f"./pred_and_QR/outcome/plots/{imgName}/bpp_psnr_{method}_{prediction_method}.png", 
               dpi=300, bbox_inches='tight')
    
    # 繪製BPP-SSIM折線圖
    fig_ssim = plt.figure(figsize=(10, 6))
    
    x = df_sorted['BPP'].values
    y = df_sorted['SSIM'].values
    
    # 使用直接連接的折線圖（而非曲線圖）
    plt.plot(x, y, 
             color='red',
             linewidth=2.5,
             marker='o',
             markersize=8,
             label=f'Method: {method}, Predictor: {prediction_method}')
    
    # 添加數據標籤
    for i, (bpp, ssim) in enumerate(zip(x, y)):
        if i % 3 == 0 or i == len(x) - 1:  # 只標記部分點，避免擁擠
            plt.annotate(f'({bpp:.4f}, {ssim:.4f})',
                        (bpp, ssim), 
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                 fc='yellow', 
                                 alpha=0.3),
                        fontsize=8)
    
    plt.xlabel('Bits Per Pixel (BPP)', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.title(f'BPP-SSIM Curve for {imgName}\nMethod: {method}, Predictor: {prediction_method}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 保存圖表
    plt.savefig(f"./pred_and_QR/outcome/plots/{imgName}/bpp_ssim_{method}_{prediction_method}.png", 
               dpi=300, bbox_inches='tight')
    
    # 如果有直方圖相關性數據，也繪製相應的折線圖
    if 'Hist_Corr' in df.columns:
        fig_hist = plt.figure(figsize=(10, 6))
        
        x = df_sorted['BPP'].values
        y = df_sorted['Hist_Corr'].values
        
        # 使用直接連接的折線圖（而非曲線圖）
        plt.plot(x, y, 
                 color='green',
                 linewidth=2.5,
                 marker='o',
                 markersize=8,
                 label=f'Method: {method}, Predictor: {prediction_method}')
        
        plt.xlabel('Bits Per Pixel (BPP)', fontsize=12)
        plt.ylabel('Histogram Correlation', fontsize=12)
        plt.title(f'BPP-Histogram Correlation Curve for {imgName}\nMethod: {method}, Predictor: {prediction_method}', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # 保存圖表
        plt.savefig(f"./pred_and_QR/outcome/plots/{imgName}/bpp_histcorr_{method}_{prediction_method}.png", 
                   dpi=300, bbox_inches='tight')
        
        return fig_psnr, fig_ssim, fig_hist
    
    return fig_psnr, fig_ssim

def create_combined_csv(all_stats, output_dir):
    """
    創建合併的CSV統計數據，方便論文使用
    
    Parameters:
    -----------
    all_stats : dict
        字典，包含各預測方法的統計數據框
    output_dir : str
        輸出目錄
    """
    # 初始化合併數據的列表
    combined_data = []
    
    # 添加所有預測方法的數據
    for predictor, df in all_stats.items():
        for _, row in df.iterrows():
            data_point = {
                'Predictor': predictor,
                'Payload': row['Payload'],
                'BPP': row['BPP'],
                'PSNR': row['PSNR'],
                'SSIM': row['SSIM']
            }
            if 'Hist_Corr' in row:
                data_point['Hist_Corr'] = row['Hist_Corr']
            
            combined_data.append(data_point)
    
    # 創建並保存合併的DataFrame
    combined_df = pd.DataFrame(combined_data)
    combined_csv = f"{output_dir}/combined_statistics.csv"
    combined_df.to_csv(combined_csv, index=False)
    
    print(f"Combined statistics saved to {combined_csv}")
    
    # 創建寬格式的DataFrame (對於論文中的表格可能更有用)
    # 對於每個指標類型 (PSNR, SSIM等)，創建一個表格，其中各列是不同的BPP值，各行是不同的預測方法
    
    # 首先找出所有唯一的BPP值
    all_bpp = set()
    for df in all_stats.values():
        all_bpp.update(df['BPP'].values)
    all_bpp = sorted(list(all_bpp))
    
    # 標準化BPP值（選擇最接近的值）
    # 這對於生成表格很有用，因為不同預測方法可能產生略微不同的BPP值
    standard_bpp = []
    
    # 如果BPP值較多，選擇15個均勻分布的點
    if len(all_bpp) > 15:
        min_bpp = min(all_bpp)
        max_bpp = max(all_bpp)
        standard_bpp = np.linspace(min_bpp, max_bpp, 15).tolist()
    else:
        standard_bpp = all_bpp
    
    # 對於每個指標類型，創建一個寬格式的DataFrame
    for metric in ['PSNR', 'SSIM', 'Hist_Corr']:
        if metric not in next(iter(all_stats.values())).columns and metric != 'PSNR' and metric != 'SSIM':
            continue
        
        # 初始化結果字典
        wide_data = {'BPP': standard_bpp}
        
        # 對於每個預測方法，尋找與標準BPP最接近的值
        for predictor, df in all_stats.items():
            metric_values = []
            
            for bpp in standard_bpp:
                # 找到最接近的BPP值
                closest_idx = (df['BPP'] - bpp).abs().idxmin()
                closest_row = df.loc[closest_idx]
                
                metric_values.append(closest_row[metric])
            
            wide_data[predictor] = metric_values
        
        # 創建並保存寬格式的DataFrame
        wide_df = pd.DataFrame(wide_data)
        wide_csv = f"{output_dir}/wide_format_{metric.lower()}.csv"
        wide_df.to_csv(wide_csv, index=False)
        
        print(f"Wide format {metric} data saved to {wide_csv}")