# utils.py - 重新整理後的版本

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import cupy as cp
from prettytable import PrettyTable
from common import calculate_psnr, calculate_ssim, histogram_correlation, cleanup_memory
from image_processing import save_image, generate_histogram, PredictionMethod

# =============================================================================
# 第一部分：基本工具函數
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
# 第二部分：PEE 資訊表格相關功能
# =============================================================================

def create_pee_info_table(pee_stages, use_different_weights, total_pixels, 
                         split_size, quad_tree=False):
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

# =============================================================================
# 第三部分：嵌入數據生成相關函數
# =============================================================================

def generate_embedding_data(total_embeddings, sub_images_per_stage, max_capacity_per_subimage, 
                           ratio_of_ones=0.5, target_payload_size=-1):
    """
    生成嵌入數據
    
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
            'remaining_target': total_remaining  # 記錄當前階段還需要嵌入多少數據
        })
    
    return {
        'stage_data': stage_data,
        'total_target': target_payload_size
    }

# =============================================================================
# 第四部分：精確測量相關函數
# =============================================================================

def run_embedding_with_target(origImg, method, prediction_method, ratio_of_ones, 
                             total_embeddings, el_mode, target_payload_size,
                             split_size=2, block_base=False, quad_tree_params=None,
                             use_different_weights=False):
    """
    執行特定嵌入算法，針對特定的目標payload
    
    Parameters:
    -----------
    origImg : numpy.ndarray
        原始圖像
    method : str
        使用的方法 ("rotation", "split", "quadtree")
    prediction_method : PredictionMethod
        預測方法
    ratio_of_ones : float
        嵌入數據中1的比例
    total_embeddings : int
        總嵌入次數
    el_mode : int
        EL模式
    target_payload_size : int
        目標嵌入量
    split_size : int, optional
        分割大小
    block_base : bool, optional
        是否使用block base方式
    quad_tree_params : dict, optional
        四叉樹參數
    use_different_weights : bool, optional
        是否使用不同權重
        
    Returns:
    --------
    tuple
        (final_img, actual_payload, stages)
    """
    from embedding import (
        pee_process_with_rotation_cuda,
        pee_process_with_split_cuda
    )
    from quadtree import pee_process_with_quadtree_cuda
    
    # 重置GPU記憶體
    cp.get_default_memory_pool().free_all_blocks()
    
    # 根據方法選擇相應的嵌入算法
    if method == "rotation":
        final_img, actual_payload, stages = pee_process_with_rotation_cuda(
            origImg,
            total_embeddings,
            ratio_of_ones,
            use_different_weights,
            split_size,
            el_mode,
            prediction_method=prediction_method,
            target_payload_size=target_payload_size
        )
    elif method == "split":
        final_img, actual_payload, stages = pee_process_with_split_cuda(
            origImg,
            total_embeddings,
            ratio_of_ones,
            use_different_weights,
            split_size,
            el_mode,
            block_base,
            prediction_method=prediction_method,
            target_payload_size=target_payload_size
        )
    elif method == "quadtree":
        if quad_tree_params is None:
            quad_tree_params = {
                'min_block_size': 16,
                'variance_threshold': 300
            }
        
        final_img, actual_payload, stages = pee_process_with_quadtree_cuda(
            origImg,
            total_embeddings,
            ratio_of_ones,
            use_different_weights,
            quad_tree_params['min_block_size'],
            quad_tree_params['variance_threshold'],
            el_mode,
            rotation_mode='random',
            prediction_method=prediction_method,
            target_payload_size=target_payload_size
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return final_img, actual_payload, stages

def run_precise_measurements(origImg, imgName, method, prediction_method, ratio_of_ones, 
                            total_embeddings=5, el_mode=0, segments=15, use_different_weights=False,
                            split_size=2, block_base=False, quad_tree_params=None):
    """
    運行精確的數據點測量，為均勻分布的payload目標單獨執行嵌入算法
    
    Parameters:
    -----------
    origImg : numpy.ndarray
        原始圖像
    imgName : str
        圖像名稱 (用於保存結果)
    method : str
        使用的方法
    prediction_method : PredictionMethod
        預測方法
    ratio_of_ones : float
        嵌入數據中1的比例
    total_embeddings : int
        總嵌入次數
    el_mode : int
        EL模式
    segments : int
        要測量的數據點數量
    use_different_weights : bool
        是否使用不同權重
    split_size : int
        分割大小
    block_base : bool
        是否使用block base方式
    quad_tree_params : dict
        四叉樹參數
        
    Returns:
    --------
    pandas.DataFrame
        包含所有測量結果的DataFrame
    """
    # 總運行開始時間
    total_start_time = time.time()
    
    # 創建結果目錄
    if isinstance(prediction_method, str):
        method_name = prediction_method
    else:
        method_name = prediction_method.value
        
    result_dir = f"./Prediction_Error_Embedding/outcome/plots/{imgName}/precise_{method_name}"
    os.makedirs(result_dir, exist_ok=True)
    
    # 記錄運行設置
    log_file = f"{result_dir}/precise_measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, 'w') as f:
        f.write(f"Precise measurement run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Image: {imgName}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Prediction method: {method_name}\n")
        f.write(f"Ratio of ones: {ratio_of_ones}\n")
        f.write(f"Total embeddings: {total_embeddings}\n")
        f.write(f"EL mode: {el_mode}\n")
        f.write(f"Segments: {segments}\n")
        f.write(f"Use different weights: {use_different_weights}\n")
        f.write("\n" + "="*80 + "\n\n")
    
    # 步驟1: 找出最大嵌入容量
    print(f"\n{'='*80}")
    print(f"Step 1: Finding maximum payload capacity")
    print(f"{'='*80}")
    
    with open(log_file, 'a') as f:
        f.write(f"Step 1: Finding maximum payload capacity\n")
    
    start_time = time.time()
    final_img_max, max_payload, stages_max = run_embedding_with_target(
        origImg, method, prediction_method, ratio_of_ones, 
        total_embeddings, el_mode, target_payload_size=-1,
        split_size=split_size, block_base=block_base, 
        quad_tree_params=quad_tree_params,
        use_different_weights=use_different_weights
    )
    
    max_run_time = time.time() - start_time
    total_pixels = origImg.size
    
    print(f"Maximum payload: {max_payload} bits")
    print(f"Max BPP: {max_payload/total_pixels:.6f}")
    print(f"Time taken: {max_run_time:.2f} seconds")
    
    with open(log_file, 'a') as f:
        f.write(f"Maximum payload: {max_payload} bits\n")
        f.write(f"Max BPP: {max_payload/total_pixels:.6f}\n")
        f.write(f"Time taken: {max_run_time:.2f} seconds\n\n")
    
    # 步驟2: 計算均勻分布的payload點
    print(f"\n{'='*80}")
    print(f"Step 2: Calculating {segments} evenly distributed payload points")
    print(f"{'='*80}")
    
    with open(log_file, 'a') as f:
        f.write(f"Step 2: Calculating {segments} evenly distributed payload points\n")
    
    # 計算每個級距的目標嵌入量 (從10%到100%)
    payload_points = [int(max_payload * (i+1) / segments) for i in range(segments)]
    
    print("Target payload points:")
    for i, target in enumerate(payload_points):
        print(f"  Point {i+1}: {target} bits ({target/max_payload*100:.1f}% of max)")
    
    with open(log_file, 'a') as f:
        f.write("Target payload points:\n")
        for i, target in enumerate(payload_points):
            f.write(f"  Point {i+1}: {target} bits ({target/max_payload*100:.1f}% of max)\n")
        f.write("\n")
    
    # 步驟3: 為每個目標點運行嵌入算法
    print(f"\n{'='*80}")
    print(f"Step 3: Running embedding algorithm for each target point")
    print(f"{'='*80}")
    
    with open(log_file, 'a') as f:
        f.write(f"Step 3: Running embedding algorithm for each target point\n")
    
    # 記錄結果
    results = []
    
    # 添加最大嵌入容量的結果
    psnr_max = calculate_psnr(origImg, final_img_max)
    ssim_max = calculate_ssim(origImg, final_img_max)
    hist_corr_max = histogram_correlation(
        np.histogram(origImg, bins=256, range=(0, 255))[0],
        np.histogram(final_img_max, bins=256, range=(0, 255))[0]
    )
    
    # 將100%結果加入列表
    results.append({
        'Target_Percentage': 100.0,
        'Target_Payload': max_payload,
        'Actual_Payload': max_payload,
        'BPP': max_payload / total_pixels,
        'PSNR': psnr_max,
        'SSIM': ssim_max,
        'Hist_Corr': hist_corr_max,
        'Processing_Time': max_run_time
    })
    
    # 保存最大容量的嵌入圖像
    save_image(final_img_max, f"{result_dir}/embedded_100pct.png")
    
    with open(log_file, 'a') as f:
        f.write(f"100.0% target (Max capacity):\n")
        f.write(f"  Target: {max_payload} bits\n")
        f.write(f"  Actual: {max_payload} bits\n")
        f.write(f"  BPP: {max_payload/total_pixels:.6f}\n")
        f.write(f"  PSNR: {psnr_max:.2f}\n")
        f.write(f"  SSIM: {ssim_max:.4f}\n")
        f.write(f"  Hist_Corr: {hist_corr_max:.4f}\n")
        f.write(f"  Time: {max_run_time:.2f} seconds\n\n")
    
    # 運行其餘級距的測量 (1到segments-1，跳過最後一個因為已經有了max結果)
    for i, target in enumerate(payload_points[:-1]):
        percentage = (i+1) / segments * 100
        
        print(f"\nRunning point {i+1}/{segments}: {target} bits ({percentage:.1f}% of max)")
        
        with open(log_file, 'a') as f:
            f.write(f"{percentage:.1f}% target:\n")
            f.write(f"  Target: {target} bits\n")
        
        start_time = time.time()
        final_img, actual_payload, stages = run_embedding_with_target(
            origImg, method, prediction_method, ratio_of_ones, 
            total_embeddings, el_mode, target_payload_size=target,
            split_size=split_size, block_base=block_base, 
            quad_tree_params=quad_tree_params,
            use_different_weights=use_different_weights
        )
        
        run_time = time.time() - start_time
        
        # 計算質量指標
        psnr = calculate_psnr(origImg, final_img)
        ssim = calculate_ssim(origImg, final_img)
        hist_corr = histogram_correlation(
            np.histogram(origImg, bins=256, range=(0, 255))[0],
            np.histogram(final_img, bins=256, range=(0, 255))[0]
        )
        
        # 記錄結果
        results.append({
            'Target_Percentage': percentage,
            'Target_Payload': target,
            'Actual_Payload': actual_payload,
            'BPP': actual_payload / total_pixels,
            'PSNR': psnr,
            'SSIM': ssim,
            'Hist_Corr': hist_corr,
            'Processing_Time': run_time
        })
        
        # 保存嵌入圖像
        save_image(final_img, f"{result_dir}/embedded_{int(percentage)}pct.png")
        
        print(f"  Actual: {actual_payload} bits")
        print(f"  BPP: {actual_payload/total_pixels:.6f}")
        print(f"  PSNR: {psnr:.2f}")
        print(f"  SSIM: {ssim:.4f}")
        print(f"  Time: {run_time:.2f} seconds")
        
        with open(log_file, 'a') as f:
            f.write(f"  Actual: {actual_payload} bits\n")
            f.write(f"  BPP: {actual_payload/total_pixels:.6f}\n")
            f.write(f"  PSNR: {psnr:.2f}\n")
            f.write(f"  SSIM: {ssim:.4f}\n")
            f.write(f"  Hist_Corr: {hist_corr:.4f}\n")
            f.write(f"  Time: {run_time:.2f} seconds\n\n")
    
    # 按照正確順序排序結果
    results.sort(key=lambda x: x['Target_Percentage'])
    
    # 轉換為DataFrame
    df = pd.DataFrame(results)
    
    # 步驟4: 整理結果
    print(f"\n{'='*80}")
    print(f"Step 4: Results summary")
    print(f"{'='*80}")
    
    total_time = time.time() - total_start_time
    
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per point: {total_time/segments:.2f} seconds")
    print(f"Results saved to {result_dir}")
    
    with open(log_file, 'a') as f:
        f.write(f"Results summary:\n")
        f.write(f"Total processing time: {total_time:.2f} seconds\n")
        f.write(f"Average time per point: {total_time/segments:.2f} seconds\n")
        f.write(f"Results saved to {result_dir}\n\n")
        
        f.write("Data table:\n")
        f.write(df.to_string(index=False))
    
    # 保存結果
    csv_path = f"{result_dir}/precise_measurements.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # 繪製圖表
    plot_precise_measurements(df, imgName, method, method_name, result_dir)
    
    return df

def plot_precise_measurements(df, imgName, method, prediction_method, output_dir):
    """
    繪製精確測量結果的折線圖，並確保所有圖表資源都被正確釋放
    
    Parameters:
    -----------
    df : pandas.DataFrame
        包含測量結果的DataFrame
    imgName : str
        圖像名稱
    method : str
        使用的方法
    prediction_method : str
        預測方法
    output_dir : str
        輸出目錄
    """
    # 繪製BPP-PSNR折線圖
    plt.figure(figsize=(12, 8))
    
    plt.plot(df['BPP'], df['PSNR'], 
             color='blue',
             linewidth=2.5,
             marker='o',
             markersize=8,
             label=f'Method: {method}, Predictor: {prediction_method}')
    
    # 添加數據標籤
    for i, row in enumerate(df.itertuples()):
        if i % 3 == 0 or i == len(df) - 1:  # 只標記部分點，避免擁擠
            plt.annotate(f'({row.BPP:.4f}, {row.PSNR:.2f})',
                        (row.BPP, row.PSNR), 
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                 fc='yellow', 
                                 alpha=0.3),
                        fontsize=8)
    
    plt.xlabel('Bits Per Pixel (BPP)', fontsize=14)
    plt.ylabel('PSNR (dB)', fontsize=14)
    plt.title(f'Precise BPP-PSNR Measurements for {imgName}\n'
              f'Method: {method}, Predictor: {prediction_method}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precise_bpp_psnr.png", dpi=300)
    plt.close()  # 關閉圖表
    
    # 繪製BPP-SSIM折線圖
    plt.figure(figsize=(12, 8))
    
    plt.plot(df['BPP'], df['SSIM'], 
             color='red',
             linewidth=2.5,
             marker='o',
             markersize=8,
             label=f'Method: {method}, Predictor: {prediction_method}')
    
    # 添加數據標籤
    for i, row in enumerate(df.itertuples()):
        if i % 3 == 0 or i == len(df) - 1:  # 只標記部分點，避免擁擠
            plt.annotate(f'({row.BPP:.4f}, {row.SSIM:.4f})',
                        (row.BPP, row.SSIM), 
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                 fc='yellow', 
                                 alpha=0.3),
                        fontsize=8)
    
    plt.xlabel('Bits Per Pixel (BPP)', fontsize=14)
    plt.ylabel('SSIM', fontsize=14)
    plt.title(f'Precise BPP-SSIM Measurements for {imgName}\n'
              f'Method: {method}, Predictor: {prediction_method}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precise_bpp_ssim.png", dpi=300)
    plt.close()  # 關閉圖表
    
    # 繪製BPP-Histogram Correlation折線圖
    plt.figure(figsize=(12, 8))
    
    plt.plot(df['BPP'], df['Hist_Corr'], 
             color='green',
             linewidth=2.5,
             marker='o',
             markersize=8,
             label=f'Method: {method}, Predictor: {prediction_method}')
    
    # 添加數據標籤
    for i, row in enumerate(df.itertuples()):
        if i % 3 == 0 or i == len(df) - 1:  # 只標記部分點，避免擁擠
            plt.annotate(f'({row.BPP:.4f}, {row.Hist_Corr:.4f})',
                        (row.BPP, row.Hist_Corr), 
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                 fc='yellow', 
                                 alpha=0.3),
                        fontsize=8)
    
    plt.xlabel('Bits Per Pixel (BPP)', fontsize=14)
    plt.ylabel('Histogram Correlation', fontsize=14)
    plt.title(f'Precise BPP-Histogram Correlation Measurements for {imgName}\n'
              f'Method: {method}, Predictor: {prediction_method}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precise_bpp_hist_corr.png", dpi=300)
    plt.close()  # 關閉圖表
    
    # 繪製Target vs Actual Payload折線圖
    plt.figure(figsize=(12, 8))
    
    # 理想線 (Target = Actual)
    ideal_line = np.linspace(0, df['Target_Payload'].max(), 100)
    plt.plot(ideal_line, ideal_line, 'k--', alpha=0.5, label='Target = Actual')
    
    # 實際結果
    plt.scatter(df['Target_Payload'], df['Actual_Payload'], 
               color='purple',
               s=100, 
               alpha=0.7,
               label='Actual Results')
    
    # 添加數據標籤
    for i, row in enumerate(df.itertuples()):
        if i % 3 == 0 or i == len(df) - 1:  # 只標記部分點，避免擁擠
            plt.annotate(f'({row.Target_Percentage:.0f}%)',
                        (row.Target_Payload, row.Actual_Payload), 
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                 fc='yellow', 
                                 alpha=0.3),
                        fontsize=8)
    
    plt.xlabel('Target Payload (bits)', fontsize=14)
    plt.ylabel('Actual Payload (bits)', fontsize=14)
    plt.title(f'Target vs Actual Payload for {imgName}\n'
              f'Method: {method}, Predictor: {prediction_method}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/target_vs_actual.png", dpi=300)
    plt.close()  # 關閉圖表
    
    # 繪製Performance vs Percentage折線圖 (多Y軸)
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    color1 = 'blue'
    ax1.set_xlabel('Capacity Percentage (%)', fontsize=14)
    ax1.set_ylabel('PSNR (dB)', color=color1, fontsize=14)
    ax1.plot(df['Target_Percentage'], df['PSNR'], 
            color=color1, marker='o', linewidth=2.5, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'red'
    ax2.set_ylabel('SSIM', color=color2, fontsize=14)
    ax2.plot(df['Target_Percentage'], df['SSIM'], 
            color=color2, marker='s', linewidth=2.5, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title(f'Performance vs Capacity Percentage for {imgName}\n'
              f'Method: {method}, Predictor: {prediction_method}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, ['PSNR', 'SSIM'], loc='best', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_vs_percentage.png", dpi=300)
    plt.close()  # 關閉圖表
    
    # 繪製處理時間統計
    plt.figure(figsize=(12, 8))
    
    plt.plot(df['Target_Percentage'], df['Processing_Time'], 
             color='purple',
             linewidth=2.5,
             marker='o',
             markersize=8)
    
    plt.xlabel('Capacity Percentage (%)', fontsize=14)
    plt.ylabel('Processing Time (seconds)', fontsize=14)
    plt.title(f'Processing Time vs Capacity Percentage for {imgName}\n'
              f'Method: {method}, Predictor: {prediction_method}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/processing_time.png", dpi=300)
    plt.close()  # 關閉圖表

def run_multi_predictor_precise_measurements(imgName, filetype="png", method="quadtree", 
                                           predictor_ratios=None, total_embeddings=5, 
                                           el_mode=0, segments=15, use_different_weights=False,
                                           split_size=2, block_base=False, quad_tree_params=None):
    """
    為多個預測器運行精確測量並生成比較結果，只為 proposed 預測器儲存詳細資料
    
    Parameters:
    -----------
    imgName : str
        圖像名稱
    filetype : str
        圖像檔案類型
    method : str
        使用的方法
    predictor_ratios : dict
        各預測器的ratio_of_ones設置
    total_embeddings : int
        總嵌入次數
    el_mode : int
        EL模式
    segments : int
        要測量的數據點數量
    use_different_weights : bool
        是否使用不同權重
    split_size : int
        分割大小
    block_base : bool
        是否使用block base方式
    quad_tree_params : dict
        四叉樹參數
        
    Returns:
    --------
    dict
        包含各預測器測量結果的字典
    """
    import cv2
    from tqdm import tqdm
    
    # 設置默認的預測器ratio字典
    if predictor_ratios is None:
        predictor_ratios = {
            "PROPOSED": 0.5,
            "MED": 1.0,
            "GAP": 1.0,
            "RHOMBUS": 1.0
        }
    
    # 讀取原始圖像
    origImg = cv2.imread(f"./Prediction_Error_Embedding/image/{imgName}.{filetype}", cv2.IMREAD_GRAYSCALE)
    if origImg is None:
        raise ValueError(f"Failed to read image: ./Prediction_Error_Embedding/image/{imgName}.{filetype}")
    origImg = np.array(origImg).astype(np.uint8)
    
    # 預測方法列表
    prediction_methods = [
        PredictionMethod.PROPOSED,
        PredictionMethod.MED,
        PredictionMethod.GAP,
        PredictionMethod.RHOMBUS
    ]
    
    # 創建比較結果目錄
    comparison_dir = f"./Prediction_Error_Embedding/outcome/plots/{imgName}/precise_comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 記錄總運行開始時間
    total_start_time = time.time()
    
    # 創建記錄檔案
    log_file = f"{comparison_dir}/multi_predictor_precise_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, 'w') as f:
        f.write(f"Multi-predictor precise measurement run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Image: {imgName}.{filetype}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Total embeddings: {total_embeddings}\n")
        f.write(f"EL mode: {el_mode}\n")
        f.write(f"Segments: {segments}\n")
        f.write("Predictor ratio settings:\n")
        for pred, ratio in predictor_ratios.items():
            f.write(f"  {pred}: {ratio}\n")
            
        if method == "quadtree":
            f.write(f"Quadtree params: min_block_size={quad_tree_params['min_block_size']}, variance_threshold={quad_tree_params['variance_threshold']}\n")
        else:
            f.write(f"Split size: {split_size}x{split_size}, block_base={block_base}\n")
        f.write("\n" + "="*80 + "\n\n")
    
    # 儲存所有預測方法的結果
    all_results = {}
    
    # 依次運行每種預測方法
    for prediction_method in tqdm(prediction_methods, desc="處理預測器"):
        method_name = prediction_method.value.upper()
        is_proposed = method_name.upper() == "PROPOSED"  # 檢查是否為 proposed 預測器
        
        print(f"\n{'='*80}")
        print(f"Running precise measurements for {method_name.lower()} predictor")
        print(f"{'='*80}")
        
        # 獲取當前預測器的ratio_of_ones
        current_ratio_of_ones = predictor_ratios.get(method_name, 0.5)
        print(f"Using ratio_of_ones = {current_ratio_of_ones}")
        
        with open(log_file, 'a') as f:
            f.write(f"Starting precise measurements for {method_name.lower()} predictor\n")
            f.write(f"Using ratio_of_ones = {current_ratio_of_ones}\n\n")
        
        try:
            # 為了簡化數據處理，創建一個自定義的測量函數
            if is_proposed:
                # 對於 proposed 預測器，儲存所有詳細資料
                result_dir = f"./Prediction_Error_Embedding/outcome/plots/{imgName}/precise_{method_name.lower()}"
                os.makedirs(result_dir, exist_ok=True)
                
                # 執行精確測量
                predictor_start_time = time.time()
                results_df = run_precise_measurements(
                    origImg, imgName, method, prediction_method, 
                    current_ratio_of_ones, total_embeddings, 
                    el_mode, segments, use_different_weights,
                    split_size, block_base, quad_tree_params
                )
            else:
                # 對於其他預測器，僅儲存數據而不儲存圖像和圖表
                predictor_start_time = time.time()
                results_df = run_simplified_precise_measurements(
                    origImg, imgName, method, prediction_method, 
                    current_ratio_of_ones, total_embeddings, 
                    el_mode, segments, use_different_weights,
                    split_size, block_base, quad_tree_params
                )
            
            predictor_time = time.time() - predictor_start_time
            
            # 保存結果
            all_results[method_name.lower()] = results_df
            
            with open(log_file, 'a') as f:
                f.write(f"Completed measurements for {method_name.lower()} predictor\n")
                f.write(f"Time taken: {predictor_time:.2f} seconds\n\n")
                
            # 保存CSV到比較目錄
            results_df.to_csv(f"{comparison_dir}/{method_name.lower()}_precise.csv", index=False)
            
            # 清理記憶體
            cleanup_memory()
            
        except Exception as e:
            print(f"Error processing {method_name.lower()}: {str(e)}")
            with open(log_file, 'a') as f:
                f.write(f"Error processing {method_name.lower()}: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
                f.write("\n\n")
    
    # 生成比較圖表
    try:
        if all_results:
            plot_predictor_comparison(all_results, imgName, method, comparison_dir)
            
            # 記錄運行時間
            total_time = time.time() - total_start_time
            
            with open(log_file, 'a') as f:
                f.write(f"\nComparison completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total processing time: {total_time:.2f} seconds\n\n")
                
            print(f"\nComparison completed and saved to {comparison_dir}")
            print(f"Total processing time: {total_time:.2f} seconds")
            
            # 創建寬格式表格，便於論文使用
            create_wide_format_tables(all_results, comparison_dir)
            
            return all_results
            
    except Exception as e:
        print(f"Error generating comparison: {str(e)}")
        with open(log_file, 'a') as f:
            f.write(f"\nError generating comparison: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
    
    return all_results

def plot_predictor_comparison(all_results, imgName, method, output_dir):
    """
    繪製多預測器精確測量結果的比較圖表，並修復 DataFrame 警告
    
    Parameters:
    -----------
    all_results : dict
        包含各預測器測量結果的字典
    imgName : str
        圖像名稱
    method : str
        使用的方法
    output_dir : str
        輸出目錄
    """
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
    
    # 創建BPP-PSNR比較圖
    plt.figure(figsize=(12, 8))
    
    for predictor, df in all_results.items():
        plt.plot(df['BPP'], df['PSNR'], 
                 color=colors.get(predictor, 'black'),
                 linewidth=2.5,
                 marker=markers.get(predictor, 'x'),
                 markersize=8,
                 label=f'Predictor: {predictor}')
    
    plt.xlabel('Bits Per Pixel (BPP)', fontsize=14)
    plt.ylabel('PSNR (dB)', fontsize=14)
    plt.title(f'Precise Measurement Comparison of Different Predictors\n'
              f'Method: {method}, Image: {imgName}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_bpp_psnr.png", dpi=300)
    plt.close()  # 關閉圖表
    
    # 創建BPP-SSIM比較圖
    plt.figure(figsize=(12, 8))
    
    for predictor, df in all_results.items():
        plt.plot(df['BPP'], df['SSIM'], 
                 color=colors.get(predictor, 'black'),
                 linewidth=2.5,
                 marker=markers.get(predictor, 'x'),
                 markersize=8,
                 label=f'Predictor: {predictor}')
    
    plt.xlabel('Bits Per Pixel (BPP)', fontsize=14)
    plt.ylabel('SSIM', fontsize=14)
    plt.title(f'Precise Measurement Comparison of Different Predictors\n'
              f'Method: {method}, Image: {imgName}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_bpp_ssim.png", dpi=300)
    plt.close()  # 關閉圖表
    
    # 創建BPP-Histogram Correlation比較圖
    plt.figure(figsize=(12, 8))
    
    for predictor, df in all_results.items():
        plt.plot(df['BPP'], df['Hist_Corr'], 
                 color=colors.get(predictor, 'black'),
                 linewidth=2.5,
                 marker=markers.get(predictor, 'x'),
                 markersize=8,
                 label=f'Predictor: {predictor}')
    
    plt.xlabel('Bits Per Pixel (BPP)', fontsize=14)
    plt.ylabel('Histogram Correlation', fontsize=14)
    plt.title(f'Precise Measurement Comparison of Different Predictors\n'
              f'Method: {method}, Image: {imgName}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_bpp_hist_corr.png", dpi=300)
    plt.close()  # 關閉圖表
    
    # 創建Capacity Percentage-PSNR比較圖
    plt.figure(figsize=(12, 8))
    
    for predictor, df in all_results.items():
        plt.plot(df['Target_Percentage'], df['PSNR'], 
                 color=colors.get(predictor, 'black'),
                 linewidth=2.5,
                 marker=markers.get(predictor, 'x'),
                 markersize=8,
                 label=f'Predictor: {predictor}')
    
    plt.xlabel('Capacity Percentage (%)', fontsize=14)
    plt.ylabel('PSNR (dB)', fontsize=14)
    plt.title(f'PSNR vs Capacity Percentage Comparison\n'
              f'Method: {method}, Image: {imgName}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_percentage_psnr.png", dpi=300)
    plt.close()  # 關閉圖表
    
    # 創建Capacity Percentage-SSIM比較圖
    plt.figure(figsize=(12, 8))
    
    for predictor, df in all_results.items():
        plt.plot(df['Target_Percentage'], df['SSIM'], 
                 color=colors.get(predictor, 'black'),
                 linewidth=2.5,
                 marker=markers.get(predictor, 'x'),
                 markersize=8,
                 label=f'Predictor: {predictor}')
    
    plt.xlabel('Capacity Percentage (%)', fontsize=14)
    plt.ylabel('SSIM', fontsize=14)
    plt.title(f'SSIM vs Capacity Percentage Comparison\n'
              f'Method: {method}, Image: {imgName}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_percentage_ssim.png", dpi=300)
    plt.close()  # 關閉圖表
    
    # 創建處理時間比較圖
    plt.figure(figsize=(12, 8))
    
    for predictor, df in all_results.items():
        plt.plot(df['Target_Percentage'], df['Processing_Time'], 
                 color=colors.get(predictor, 'black'),
                 linewidth=2.5,
                 marker=markers.get(predictor, 'x'),
                 markersize=8,
                 label=f'Predictor: {predictor}')
    
    plt.xlabel('Capacity Percentage (%)', fontsize=14)
    plt.ylabel('Processing Time (seconds)', fontsize=14)
    plt.title(f'Processing Time Comparison\n'
              f'Method: {method}, Image: {imgName}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_processing_time.png", dpi=300)
    plt.close()  # 關閉圖表
    
    # 創建最大嵌入容量比較圖 (條形圖)
    plt.figure(figsize=(12, 8))
    
    max_payloads = []
    predictor_names = []
    
    for predictor, df in all_results.items():
        max_row = df.loc[df['Target_Percentage'] == 100.0]
        if not max_row.empty:
            # 修正：使用 iloc[0] 來取得 Series 中的單一值
            max_payloads.append(float(max_row['Actual_Payload'].iloc[0]))
            predictor_names.append(predictor)
    
    bars = plt.bar(predictor_names, max_payloads, color=[colors.get(p, 'gray') for p in predictor_names])
    
    # 在條形上添加數值標籤
    for bar, payload in zip(bars, max_payloads):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{int(payload)}',
                ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Predictor', fontsize=14)
    plt.ylabel('Maximum Payload (bits)', fontsize=14)
    plt.title(f'Maximum Payload Comparison\n'
              f'Method: {method}, Image: {imgName}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_max_payload.png", dpi=300)
    plt.close()  # 關閉圖表

def create_wide_format_tables(all_results, output_dir):
    """
    創建寬格式表格，便於論文使用
    
    Parameters:
    -----------
    all_results : dict
        包含各預測器測量結果的字典
    output_dir : str
        輸出目錄
    """
    # 創建PSNR表格 (列：百分比，列：預測器)
    psnr_table = {'Percentage': []}
    ssim_table = {'Percentage': []}
    hist_corr_table = {'Percentage': []}
    
    # 確定所有百分比值
    percentages = []
    for df in all_results.values():
        percentages.extend(df['Target_Percentage'].tolist())
    
    # 去重並排序
    percentages = sorted(list(set(percentages)))
    psnr_table['Percentage'] = percentages
    ssim_table['Percentage'] = percentages
    hist_corr_table['Percentage'] = percentages
    
    # 填充各預測器的數據
    for predictor, df in all_results.items():
        psnr_values = []
        ssim_values = []
        hist_corr_values = []
        
        for percentage in percentages:
            # 找到最接近的百分比行
            closest_idx = (df['Target_Percentage'] - percentage).abs().idxmin()
            row = df.loc[closest_idx]
            
            psnr_values.append(row['PSNR'])
            ssim_values.append(row['SSIM'])
            hist_corr_values.append(row['Hist_Corr'])
        
        psnr_table[predictor] = psnr_values
        ssim_table[predictor] = ssim_values
        hist_corr_table[predictor] = hist_corr_values
    
    # 創建DataFrame
    psnr_df = pd.DataFrame(psnr_table)
    ssim_df = pd.DataFrame(ssim_table)
    hist_corr_df = pd.DataFrame(hist_corr_table)
    
    # 保存表格
    psnr_df.to_csv(f"{output_dir}/wide_format_psnr.csv", index=False)
    ssim_df.to_csv(f"{output_dir}/wide_format_ssim.csv", index=False)
    hist_corr_df.to_csv(f"{output_dir}/wide_format_hist_corr.csv", index=False)
    
    # 創建LaTeX格式表格
    with open(f"{output_dir}/latex_table_psnr.txt", 'w') as f:
        f.write(psnr_df.to_latex(index=False, float_format="%.2f"))
    
    with open(f"{output_dir}/latex_table_ssim.txt", 'w') as f:
        f.write(ssim_df.to_latex(index=False, float_format="%.4f"))
    
    with open(f"{output_dir}/latex_table_hist_corr.txt", 'w') as f:
        f.write(hist_corr_df.to_latex(index=False, float_format="%.4f"))
    
    print(f"Wide format tables saved to {output_dir}")

# =============================================================================
# 第五部分：舊版測量和繪圖函數（保留以保持向後兼容性）
# =============================================================================

def generate_interval_statistics(original_img, stages, total_payload, segments=15):
    """
    根據總嵌入容量生成均勻分布的統計數據表格
    (近似方法，使用已有的階段結果進行插值)
    
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
    
    # 計算每個級距的目標嵌入量
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

def save_interval_statistics(df, imgName, method, prediction_method, base_dir="./Prediction_Error_Embedding/outcome/plots"):
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
    base_dir : str, optional
        基本儲存目錄，默認為 "./Prediction_Error_Embedding/outcome/plots"
    """
    # 確保目錄存在
    os.makedirs(f"{base_dir}/{imgName}", exist_ok=True)
    
    # 保存CSV
    csv_path = f"{base_dir}/{imgName}/interval_stats_{method}_{prediction_method}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Interval statistics saved to {csv_path}")
    
    # 保存NPY
    npy_path = f"{base_dir}/{imgName}/interval_stats_{method}_{prediction_method}.npy"
    np.save(npy_path, df.to_dict('records'))
    print(f"Interval statistics saved to {npy_path}")
    
    return csv_path, npy_path

def plot_interval_statistics(df, imgName, method, prediction_method):
    """
    繪製統計數據折線圖
    
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
    # 將數據排序
    df_sorted = df.sort_values('BPP')
    
    # 繪製BPP-PSNR折線圖
    fig_psnr = plt.figure(figsize=(10, 6))
    
    plt.plot(df_sorted['BPP'], df_sorted['PSNR'], 
             color='blue',
             linewidth=2.5,
             marker='o',
             markersize=8,
             label=f'Method: {method}, Predictor: {prediction_method}')
    
    # 添加數據標籤
    for i, row in enumerate(df_sorted.itertuples()):
        if i % 3 == 0 or i == len(df_sorted) - 1:  # 只標記部分點，避免擁擠
            plt.annotate(f'({row.BPP:.4f}, {row.PSNR:.2f})',
                        (row.BPP, row.PSNR), 
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
    plt.savefig(f"./Prediction_Error_Embedding/outcome/plots/{imgName}/bpp_psnr_{method}_{prediction_method}.png", 
               dpi=300, bbox_inches='tight')
    
    # 繪製BPP-SSIM折線圖
    fig_ssim = plt.figure(figsize=(10, 6))
    
    plt.plot(df_sorted['BPP'], df_sorted['SSIM'], 
             color='red',
             linewidth=2.5,
             marker='o',
             markersize=8,
             label=f'Method: {method}, Predictor: {prediction_method}')
    
    # 添加數據標籤
    for i, row in enumerate(df_sorted.itertuples()):
        if i % 3 == 0 or i == len(df_sorted) - 1:  # 只標記部分點，避免擁擠
            plt.annotate(f'({row.BPP:.4f}, {row.SSIM:.4f})',
                        (row.BPP, row.SSIM), 
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
    plt.savefig(f"./Prediction_Error_Embedding/outcome/plots/{imgName}/bpp_ssim_{method}_{prediction_method}.png", 
               dpi=300, bbox_inches='tight')
    
    # 如果有直方圖相關性數據，也繪製相應的折線圖
    if 'Hist_Corr' in df.columns:
        fig_hist = plt.figure(figsize=(10, 6))
        
        plt.plot(df_sorted['BPP'], df_sorted['Hist_Corr'], 
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
        plt.savefig(f"./Prediction_Error_Embedding/outcome/plots/{imgName}/bpp_histcorr_{method}_{prediction_method}.png", 
                   dpi=300, bbox_inches='tight')
        
        return fig_psnr, fig_ssim, fig_hist
    
    return fig_psnr, fig_ssim

def run_multiple_predictors(imgName, filetype="png", method="quadtree", 
                           predictor_ratios=None, total_embeddings=5, 
                           el_mode=0, use_different_weights=False,
                           split_size=2, block_base=False, 
                           quad_tree_params=None, stats_segments=15):
    """
    自動運行多種預測方法並生成比較結果
    (使用近似方法)
    
    Parameters:
    -----------
    imgName : str
        圖像名稱
    filetype : str
        圖像檔案類型
    method : str
        使用的方法
    predictor_ratios : dict
        各預測器的ratio_of_ones設置
    total_embeddings : int
        總嵌入次數
    el_mode : int
        EL模式
    use_different_weights : bool
        是否對每個子圖像使用不同的權重
    split_size : int
        分割大小
    block_base : bool
        是否使用block base方式
    quad_tree_params : dict
        四叉樹參數
    stats_segments : int
        統計分段數量
        
    Returns:
    --------
    tuple
        (results_df, all_stats) 包含比較結果的DataFrame和統計數據
    """
    from embedding import (
        pee_process_with_rotation_cuda,
        pee_process_with_split_cuda
    )
    from quadtree import pee_process_with_quadtree_cuda
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
    comparison_dir = f"./Prediction_Error_Embedding/outcome/plots/{imgName}/comparison"
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
    origImg = cv2.imread(f"./Prediction_Error_Embedding/image/{imgName}.{filetype}", cv2.IMREAD_GRAYSCALE)
    if origImg is None:
        raise ValueError(f"Failed to read image: ./Prediction_Error_Embedding/image/{imgName}.{filetype}")
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
        
        # 獲取當前預測器的ratio_of_ones
        current_ratio_of_ones = predictor_ratios.get(method_name, 0.5)
        print(f"Using ratio_of_ones = {current_ratio_of_ones}")
        
        # 針對 MED、GAP 和 RHOMBUS，強制設置 use_different_weights = False
        current_use_weights = use_different_weights
        if prediction_method in [PredictionMethod.MED, PredictionMethod.GAP, PredictionMethod.RHOMBUS]:
            current_use_weights = False
            print(f"Note: Weight optimization disabled for {method_name.lower()} prediction method")
        
        try:
            # 執行選定的方法
            if method == "rotation":
                final_pee_img, total_payload, pee_stages = pee_process_with_rotation_cuda(
                    origImg,
                    total_embeddings,
                    current_ratio_of_ones,
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
                    current_ratio_of_ones,
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
                    current_ratio_of_ones,
                    current_use_weights,
                    quad_tree_params['min_block_size'],
                    quad_tree_params['variance_threshold'],
                    el_mode,
                    rotation_mode='random',
                    prediction_method=prediction_method,
                    target_payload_size=-1  # 使用最大嵌入量
                )
            
            # 生成統計數據
            print("\nGenerating interval statistics...")
            stats_df, stats_table = generate_interval_statistics(
                origImg, pee_stages, total_payload, segments=stats_segments
            )
            
            # 保存統計數據
            if stats_df is not None:
                # 記錄統計數據
                stats_df['Predictor'] = method_name.lower()
                stats_df['Ratio_of_Ones'] = current_ratio_of_ones
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
                    f.write(f"Run completed for {method_name.lower()} predictor\n")
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
            plot_predictor_comparison(all_stats, imgName, method, comparison_dir)
            
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

def run_simplified_precise_measurements(origImg, imgName, method, prediction_method, ratio_of_ones, 
                                      total_embeddings=5, el_mode=0, segments=15, use_different_weights=False,
                                      split_size=2, block_base=False, quad_tree_params=None):
    """
    運行精確的數據點測量，但僅儲存數據而不產生圖像和圖表
    適用於非 proposed 預測器
    
    Parameters:
    -----------
    origImg : numpy.ndarray
        原始圖像
    imgName : str
        圖像名稱 (用於保存結果)
    method : str
        使用的方法
    prediction_method : PredictionMethod
        預測方法
    ratio_of_ones : float
        嵌入數據中1的比例
    total_embeddings : int
        總嵌入次數
    el_mode : int
        EL模式
    segments : int
        要測量的數據點數量
    use_different_weights : bool
        是否使用不同權重
    split_size : int
        分割大小
    block_base : bool
        是否使用block base方式
    quad_tree_params : dict
        四叉樹參數
        
    Returns:
    --------
    pandas.DataFrame
        包含所有測量結果的DataFrame
    """
    # 總運行開始時間
    total_start_time = time.time()
    method_name = prediction_method.value
    
    # 創建結果目錄 (僅用於儲存CSV，不儲存圖像和圖表)
    result_dir = f"./Prediction_Error_Embedding/outcome/plots/{imgName}/data_{method_name.lower()}"
    os.makedirs(result_dir, exist_ok=True)
    
    # 記錄運行設置
    log_file = f"{result_dir}/simplified_measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, 'w') as f:
        f.write(f"Simplified measurement run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Image: {imgName}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Prediction method: {method_name}\n")
        f.write(f"Ratio of ones: {ratio_of_ones}\n")
        f.write(f"Total embeddings: {total_embeddings}\n")
        f.write(f"EL mode: {el_mode}\n")
        f.write(f"Segments: {segments}\n")
        f.write(f"Use different weights: {use_different_weights}\n")
        f.write("\n" + "="*80 + "\n\n")
    
    # 步驟1: 找出最大嵌入容量
    print(f"\n{'='*80}")
    print(f"Step 1: Finding maximum payload capacity for {method_name}")
    print(f"{'='*80}")
    
    with open(log_file, 'a') as f:
        f.write(f"Step 1: Finding maximum payload capacity\n")
    
    start_time = time.time()
    final_img_max, max_payload, stages_max = run_embedding_with_target(
        origImg, method, prediction_method, ratio_of_ones, 
        total_embeddings, el_mode, target_payload_size=-1,
        split_size=split_size, block_base=block_base, 
        quad_tree_params=quad_tree_params,
        use_different_weights=use_different_weights
    )
    
    max_run_time = time.time() - start_time
    total_pixels = origImg.size
    
    print(f"Maximum payload: {max_payload} bits")
    print(f"Max BPP: {max_payload/total_pixels:.6f}")
    print(f"Time taken: {max_run_time:.2f} seconds")
    
    with open(log_file, 'a') as f:
        f.write(f"Maximum payload: {max_payload} bits\n")
        f.write(f"Max BPP: {max_payload/total_pixels:.6f}\n")
        f.write(f"Time taken: {max_run_time:.2f} seconds\n\n")
    
    # 清理記憶體
    cleanup_memory()
    
    # 步驟2: 計算均勻分布的payload點
    print(f"\n{'='*80}")
    print(f"Step 2: Calculating {segments} evenly distributed payload points")
    print(f"{'='*80}")
    
    with open(log_file, 'a') as f:
        f.write(f"Step 2: Calculating {segments} evenly distributed payload points\n")
    
    # 計算每個級距的目標嵌入量 (從10%到100%)
    payload_points = [int(max_payload * (i+1) / segments) for i in range(segments)]
    
    print("Target payload points:")
    for i, target in enumerate(payload_points):
        print(f"  Point {i+1}: {target} bits ({target/max_payload*100:.1f}% of max)")
    
    with open(log_file, 'a') as f:
        f.write("Target payload points:\n")
        for i, target in enumerate(payload_points):
            f.write(f"  Point {i+1}: {target} bits ({target/max_payload*100:.1f}% of max)\n")
        f.write("\n")
    
    # 步驟3: 為每個目標點運行嵌入算法
    print(f"\n{'='*80}")
    print(f"Step 3: Running embedding algorithm for each target point")
    print(f"{'='*80}")
    
    with open(log_file, 'a') as f:
        f.write(f"Step 3: Running embedding algorithm for each target point\n")
    
    # 記錄結果
    results = []
    
    # 添加最大嵌入容量的結果
    psnr_max = calculate_psnr(origImg, final_img_max)
    ssim_max = calculate_ssim(origImg, final_img_max)
    hist_corr_max = histogram_correlation(
        np.histogram(origImg, bins=256, range=(0, 255))[0],
        np.histogram(final_img_max, bins=256, range=(0, 255))[0]
    )
    
    # 將100%結果加入列表
    results.append({
        'Target_Percentage': 100.0,
        'Target_Payload': max_payload,
        'Actual_Payload': max_payload,
        'BPP': max_payload / total_pixels,
        'PSNR': psnr_max,
        'SSIM': ssim_max,
        'Hist_Corr': hist_corr_max,
        'Processing_Time': max_run_time
    })
    
    # 注意：我們不保存圖像，只記錄數據
    
    with open(log_file, 'a') as f:
        f.write(f"100.0% target (Max capacity):\n")
        f.write(f"  Target: {max_payload} bits\n")
        f.write(f"  Actual: {max_payload} bits\n")
        f.write(f"  BPP: {max_payload/total_pixels:.6f}\n")
        f.write(f"  PSNR: {psnr_max:.2f}\n")
        f.write(f"  SSIM: {ssim_max:.4f}\n")
        f.write(f"  Hist_Corr: {hist_corr_max:.4f}\n")
        f.write(f"  Time: {max_run_time:.2f} seconds\n\n")
    
    # 清理記憶體
    cleanup_memory()
    
    # 使用 tqdm 添加進度條
    from tqdm import tqdm
    
    # 運行其餘級距的測量 (1到segments-1，跳過最後一個因為已經有了max結果)
    for i, target in enumerate(tqdm(payload_points[:-1], desc=f"處理 {method_name} 數據點")):
        percentage = (i+1) / segments * 100
        
        print(f"\nRunning point {i+1}/{segments}: {target} bits ({percentage:.1f}% of max)")
        
        with open(log_file, 'a') as f:
            f.write(f"{percentage:.1f}% target:\n")
            f.write(f"  Target: {target} bits\n")
        
        start_time = time.time()
        final_img, actual_payload, stages = run_embedding_with_target(
            origImg, method, prediction_method, ratio_of_ones, 
            total_embeddings, el_mode, target_payload_size=target,
            split_size=split_size, block_base=block_base, 
            quad_tree_params=quad_tree_params,
            use_different_weights=use_different_weights
        )
        
        run_time = time.time() - start_time
        
        # 計算質量指標
        psnr = calculate_psnr(origImg, final_img)
        ssim = calculate_ssim(origImg, final_img)
        hist_corr = histogram_correlation(
            np.histogram(origImg, bins=256, range=(0, 255))[0],
            np.histogram(final_img, bins=256, range=(0, 255))[0]
        )
        
        # 記錄結果
        results.append({
            'Target_Percentage': percentage,
            'Target_Payload': target,
            'Actual_Payload': actual_payload,
            'BPP': actual_payload / total_pixels,
            'PSNR': psnr,
            'SSIM': ssim,
            'Hist_Corr': hist_corr,
            'Processing_Time': run_time
        })
        
        # 注意：我們不保存圖像，只記錄數據
        
        print(f"  Actual: {actual_payload} bits")
        print(f"  BPP: {actual_payload/total_pixels:.6f}")
        print(f"  PSNR: {psnr:.2f}")
        print(f"  SSIM: {ssim:.4f}")
        print(f"  Time: {run_time:.2f} seconds")
        
        with open(log_file, 'a') as f:
            f.write(f"  Actual: {actual_payload} bits\n")
            f.write(f"  BPP: {actual_payload/total_pixels:.6f}\n")
            f.write(f"  PSNR: {psnr:.2f}\n")
            f.write(f"  SSIM: {ssim:.4f}\n")
            f.write(f"  Hist_Corr: {hist_corr:.4f}\n")
            f.write(f"  Time: {run_time:.2f} seconds\n\n")
        
        # 清理記憶體
        cleanup_memory()
    
    # 按照正確順序排序結果
    results.sort(key=lambda x: x['Target_Percentage'])
    
    # 轉換為DataFrame
    df = pd.DataFrame(results)
    
    # 步驟4: 整理結果
    print(f"\n{'='*80}")
    print(f"Step 4: Results summary for {method_name}")
    print(f"{'='*80}")
    
    total_time = time.time() - total_start_time
    
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per point: {total_time/segments:.2f} seconds")
    print(f"Results saved to {result_dir}")
    
    with open(log_file, 'a') as f:
        f.write(f"Results summary:\n")
        f.write(f"Total processing time: {total_time:.2f} seconds\n")
        f.write(f"Average time per point: {total_time/segments:.2f} seconds\n")
        f.write(f"Results saved to {result_dir}\n\n")
        
        f.write("Data table:\n")
        f.write(df.to_string(index=False))
    
    # 保存結果
    csv_path = f"{result_dir}/simplified_measurements.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # 注意：我們不生成圖表，只保存數據
    
    return df

def create_wide_format_tables(all_results, output_dir):
    """
    創建寬格式表格，便於論文使用，並修復可能的警告
    
    Parameters:
    -----------
    all_results : dict
        包含各預測器測量結果的字典
    output_dir : str
        輸出目錄
    """
    # 創建PSNR表格 (列：百分比，列：預測器)
    psnr_table = {'Percentage': []}
    ssim_table = {'Percentage': []}
    hist_corr_table = {'Percentage': []}
    
    # 確定所有百分比值
    percentages = []
    for df in all_results.values():
        percentages.extend(df['Target_Percentage'].tolist())
    
    # 去重並排序
    percentages = sorted(list(set(percentages)))
    psnr_table['Percentage'] = percentages
    ssim_table['Percentage'] = percentages
    hist_corr_table['Percentage'] = percentages
    
    # 填充各預測器的數據
    for predictor, df in all_results.items():
        psnr_values = []
        ssim_values = []
        hist_corr_values = []
        
        for percentage in percentages:
            # 找到最接近的百分比行
            closest_idx = (df['Target_Percentage'] - percentage).abs().idxmin()
            # 使用 .loc[idx, 'column'] 而不是 .loc[idx]['column'] 來避免警告
            psnr_values.append(df.loc[closest_idx, 'PSNR'])
            ssim_values.append(df.loc[closest_idx, 'SSIM'])
            hist_corr_values.append(df.loc[closest_idx, 'Hist_Corr'])
        
        psnr_table[predictor] = psnr_values
        ssim_table[predictor] = ssim_values
        hist_corr_table[predictor] = hist_corr_values
    
    # 創建DataFrame
    psnr_df = pd.DataFrame(psnr_table)
    ssim_df = pd.DataFrame(ssim_table)
    hist_corr_df = pd.DataFrame(hist_corr_table)
    
    # 保存表格
    psnr_df.to_csv(f"{output_dir}/wide_format_psnr.csv", index=False)
    ssim_df.to_csv(f"{output_dir}/wide_format_ssim.csv", index=False)
    hist_corr_df.to_csv(f"{output_dir}/wide_format_hist_corr.csv", index=False)
    
    # 創建LaTeX格式表格
    with open(f"{output_dir}/latex_table_psnr.txt", 'w') as f:
        f.write(psnr_df.to_latex(index=False, float_format="%.2f"))
    
    with open(f"{output_dir}/latex_table_ssim.txt", 'w') as f:
        f.write(ssim_df.to_latex(index=False, float_format="%.4f"))
    
    with open(f"{output_dir}/latex_table_hist_corr.txt", 'w') as f:
        f.write(hist_corr_df.to_latex(index=False, float_format="%.4f"))
    
    print(f"Wide format tables saved to {output_dir}")
    
    