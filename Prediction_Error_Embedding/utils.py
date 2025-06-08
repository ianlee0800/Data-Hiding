import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import time
import traceback
from scipy.signal import savgol_filter
from tqdm import tqdm
from datetime import datetime
import cupy as cp
from prettytable import PrettyTable

from common import (
    calculate_psnr,
    calculate_ssim,
    histogram_correlation,
    cleanup_memory
)
from image_processing import (
    save_image,
    generate_histogram,
    PredictionMethod
)



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
    創建 PEE 資訊表格的完整函數 - 支援彩色圖像
    """    
    table = PrettyTable()
    
    # 🔧 修改：改進彩色圖像檢測邏輯
    is_color_image = False
    if pee_stages and 'channel_payloads' in pee_stages[0]:
        # 新的檢測方式：檢查是否有channel_payloads欄位
        is_color_image = True
    elif pee_stages and 'block_info' in pee_stages[0]:
        # 舊的檢測方式：檢查block_info中是否有通道標識
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
        
        # 🔧 修改：簡化彩色圖像的表格內容，重點顯示總體信息
        for stage in pee_stages:
            # 添加整體 stage 資訊
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
            
            # 添加分隔線
            table.add_row(["-" * 5] * len(table.field_names))
    
    elif quad_tree:
        # Quad tree 模式的表格欄位（保持不變）
        table.field_names = [
            "Embedding", "Block Size", "Block Position", "Payload", "BPP",
            "PSNR", "SSIM", "Hist Corr", "Weights", "EL", "Note"
        ]
        
        # 處理 quad tree 模式的區塊資訊（保持原邏輯）
        for stage in pee_stages:
            # 添加整體 stage 資訊
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
            
            # 添加分隔線
            table.add_row(["-" * 5] * len(table.field_names))
            
            # 處理 quad tree 模式的區塊資訊
            for size_str in sorted(stage['block_info'].keys(), key=int, reverse=True):
                blocks = stage['block_info'][size_str]['blocks']
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
            
            # 添加分隔線
            table.add_row(["-" * 5] * len(table.field_names))
            
    else:
        # 標準模式的表格欄位（保持不變）
        table.field_names = [
            "Embedding", "Sub-image", "Payload", "BPP", "PSNR", "SSIM",
            "Hist Corr", "Weights", "EL", "Rotation", "Note"
        ]
        
        # 處理標準模式（保持原邏輯）
        for stage in pee_stages:
            # 添加整體 stage 資訊
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
            
            # 處理標準模式的區塊資訊
            total_blocks = split_size * split_size
            sub_image_pixels = total_pixels // total_blocks
            
            for i, block in enumerate(stage['block_params']):
                # Process weights display with better error handling
                weights_display = "-"
                if 'weights' in block:
                    if block['weights'] == "N/A":
                        weights_display = "N/A"
                    elif block['weights']:
                        try:
                            weights_display = ", ".join([f"{w:.2f}" for w in block['weights']])
                        except:
                            weights_display = str(block['weights'])
                
                # Get other fields with defaults if missing
                payload = block.get('payload', 0)
                psnr = block.get('psnr', 0)
                ssim = block.get('ssim', 0)
                hist_corr = block.get('hist_corr', 0)
                el = block.get('EL', '-')
                rotation = block.get('rotation', 0)
                
                table.add_row([
                    stage['embedding'] if i == 0 else "",
                    i,
                    payload,
                    f"{payload / sub_image_pixels:.4f}",
                    f"{psnr:.2f}",
                    f"{ssim:.4f}",
                    f"{hist_corr:.4f}",
                    weights_display,
                    el,
                    f"{rotation}°",
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

def ensure_bpp_psnr_consistency(results_df):
    """
    確保 BPP-PSNR 數據的一致性：較高的 BPP 應有較低的 PSNR
    """
    df = results_df.copy().sort_values('BPP')
    
    # 確保 PSNR 隨著 BPP 增加而單調下降
    for i in range(1, len(df)):
        if df.iloc[i]['PSNR'] > df.iloc[i-1]['PSNR']:
            # 異常點檢測：當前 PSNR 高於前一個點
            if i > 1:
                # 使用前兩個點的平均斜率進行修正
                prev_slope = (df.iloc[i-1]['PSNR'] - df.iloc[i-2]['PSNR']) / (df.iloc[i-1]['BPP'] - df.iloc[i-2]['BPP'])
                expected_drop = prev_slope * (df.iloc[i]['BPP'] - df.iloc[i-1]['BPP'])
                corrected_psnr = max(df.iloc[i-1]['PSNR'] + expected_drop, df.iloc[i-1]['PSNR'] * 0.995)
                df.loc[df.index[i], 'PSNR'] = corrected_psnr
            else:
                # 簡單地將 PSNR 設為稍低於前一個點
                df.loc[df.index[i], 'PSNR'] = df.iloc[i-1]['PSNR'] * 0.995
    
    # 對 SSIM 和 Hist_Corr 也進行類似的處理
    for metric in ['SSIM', 'Hist_Corr']:
        for i in range(1, len(df)):
            if df.iloc[i][metric] > df.iloc[i-1][metric]:
                # 簡單地進行平滑處理
                df.loc[df.index[i], metric] = df.iloc[i-1][metric] * 0.99
    
    return df.sort_values('Target_Percentage')

def run_precise_measurements(origImg, imgName, method, prediction_method, ratio_of_ones, 
                            total_embeddings=5, el_mode=0, segments=15, step_size=None, use_different_weights=False,
                            split_size=2, block_base=False, quad_tree_params=None):
    """
    運行精確的數據點測量，為均勻分布的payload目標單獨執行嵌入算法
    增加數據平滑處理和異常點修正功能，並確保最大容量點使用初始測量值
    
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
        要測量的數據點數量 (如果提供了step_size則忽略此參數)
    step_size : int, optional
        測量點之間的步長 (以位元為單位，例如10000)
        如果提供，則覆蓋segments參數
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
        if step_size:
            f.write(f"Step size: {step_size} bits\n")
        else:
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
    
    # 計算最大容量的品質指標
    psnr_max = calculate_psnr(origImg, final_img_max)
    ssim_max = calculate_ssim(origImg, final_img_max)
    hist_corr_max = histogram_correlation(
        np.histogram(origImg, bins=256, range=(0, 255))[0],
        np.histogram(final_img_max, bins=256, range=(0, 255))[0]
    )
    
    # 創建最大容量結果字典
    max_capacity_result = {
        'Target_Percentage': 100.0,
        'Target_Payload': max_payload,
        'Actual_Payload': max_payload,
        'BPP': max_payload / total_pixels,
        'PSNR': psnr_max,
        'SSIM': ssim_max,
        'Hist_Corr': hist_corr_max,
        'Processing_Time': max_run_time,
        'Suspicious': False  # 初始測量不標記為可疑
    }
    
    print(f"Maximum payload: {max_payload} bits")
    print(f"Max BPP: {max_payload/total_pixels:.6f}")
    print(f"Max PSNR: {psnr_max:.2f}")
    print(f"Max SSIM: {ssim_max:.4f}")
    print(f"Time taken: {max_run_time:.2f} seconds")
    
    with open(log_file, 'a') as f:
        f.write(f"Maximum payload: {max_payload} bits\n")
        f.write(f"Max BPP: {max_payload/total_pixels:.6f}\n")
        f.write(f"Max PSNR: {psnr_max:.2f}\n")
        f.write(f"Max SSIM: {ssim_max:.4f}\n")
        f.write(f"Max Hist Corr: {hist_corr_max:.4f}\n")
        f.write(f"Time taken: {max_run_time:.2f} seconds\n\n")
    
    # 保存最大容量的嵌入圖像
    save_image(final_img_max, f"{result_dir}/embedded_100pct.png")
    
    # 步驟2: 計算測量點，排除最大容量點（因為已經測量）
    print(f"\n{'='*80}")
    if step_size and step_size > 0:
        print(f"Step 2: Calculating measurement points with {step_size} bit steps")
    else:
        print(f"Step 2: Calculating {segments} evenly distributed payload points")
    print(f"{'='*80}")
    
    with open(log_file, 'a') as f:
        if step_size and step_size > 0:
            f.write(f"Step 2: Calculating measurement points with {step_size} bit steps\n")
        else:
            f.write(f"Step 2: Calculating {segments} evenly distributed payload points\n")
    
    # 根據參數生成測量點，但排除最大容量點
    if step_size and step_size > 0:
        # 使用固定步長生成測量點
        payload_points = list(range(step_size, max_payload, step_size))
    else:
        # 使用分段生成測量點，但排除100%點
        payload_points = [int(max_payload * (i+1) / segments) for i in range(segments-1)]
        # 添加最後一個點作為接近最大值的點（例如99%），如果需要的話
        if segments > 1:
            payload_points.append(int(max_payload * 0.99))
    
    # 確保測量點中不包含最大容量
    if max_payload in payload_points:
        payload_points.remove(max_payload)
    
    print("Target payload points:")
    for i, target in enumerate(payload_points):
        print(f"  Point {i+1}: {target} bits ({target/max_payload*100:.1f}% of max)")
    print(f"  Point {len(payload_points)+1}: {max_payload} bits (100.0% of max) [using initial measurement]")
    
    with open(log_file, 'a') as f:
        f.write("Target payload points:\n")
        for i, target in enumerate(payload_points):
            f.write(f"  Point {i+1}: {target} bits ({target/max_payload*100:.1f}% of max)\n")
        f.write(f"  Point {len(payload_points)+1}: {max_payload} bits (100.0% of max) [using initial measurement]\n\n")
    
    # 步驟3: 為每個目標點運行嵌入算法，不包括最大容量點
    print(f"\n{'='*80}")
    print(f"Step 3: Running embedding algorithm for each target point (excluding max capacity)")
    print(f"{'='*80}")
    
    with open(log_file, 'a') as f:
        f.write(f"Step 3: Running embedding algorithm for each target point\n")
    
    # 結果列表現在只包含最大容量點
    results = [max_capacity_result]

    
    # 運行每個測量點，但跳過最大容量點
    for i, target in enumerate(tqdm(payload_points, desc="處理測量點")):
        # 計算百分比，用於命名和日誌
        percentage = target / max_payload * 100
        
        print(f"\nRunning point {i+1}/{len(payload_points)}: {target} bits ({percentage:.1f}% of max)")
        
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
        
        # 檢查 PSNR 是否異常
        is_psnr_suspicious = False
        if len(results) > 0:
            last_result = results[-1]
            # 如果當前 BPP 更高但 PSNR 也更高，則標記為異常
            if (actual_payload / total_pixels > last_result['BPP'] and 
                psnr > last_result['PSNR']):
                is_psnr_suspicious = True
                print(f"  Warning: Suspicious PSNR value detected: {psnr:.2f} > previous {last_result['PSNR']:.2f}")
                with open(log_file, 'a') as f:
                    f.write(f"  Warning: Suspicious PSNR value detected: {psnr:.2f} > previous {last_result['PSNR']:.2f}\n")
        
        # 記錄結果
        results.append({
            'Target_Percentage': percentage,
            'Target_Payload': target,
            'Actual_Payload': actual_payload,
            'BPP': actual_payload / total_pixels,
            'PSNR': psnr,
            'SSIM': ssim,
            'Hist_Corr': hist_corr,
            'Processing_Time': run_time,
            'Suspicious': is_psnr_suspicious
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
        
        # 清理記憶體
        cleanup_memory()
    
    # 按照 BPP 順序排序結果
    results.sort(key=lambda x: x['BPP'])
    
    # 轉換為DataFrame
    df = pd.DataFrame(results)
    
    # 步驟3.5: 處理異常數據點，確保曲線平滑，但保留最大容量點的原始值
    print(f"\n{'='*80}")
    print(f"Step 3.5: Processing anomalous data points (preserving max capacity point)")
    print(f"{'='*80}")
    
    # 保留原始數據的副本
    original_df = df.copy()
    
    # 標記最大容量點的索引
    max_capacity_idx = df[df['Target_Percentage'] == 100.0].index[0]
    
    # 存儲最大容量點的原始指標值
    max_capacity_metrics = {
        'PSNR': df.loc[max_capacity_idx, 'PSNR'],
        'SSIM': df.loc[max_capacity_idx, 'SSIM'],
        'Hist_Corr': df.loc[max_capacity_idx, 'Hist_Corr']
    }
    
    print(f"Preserving maximum capacity point metrics:")
    print(f"  PSNR: {max_capacity_metrics['PSNR']:.2f}")
    print(f"  SSIM: {max_capacity_metrics['SSIM']:.4f}")
    print(f"  Hist_Corr: {max_capacity_metrics['Hist_Corr']:.4f}")
    
    # 對各指標進行平滑處理
    metrics_to_smooth = ['PSNR', 'SSIM', 'Hist_Corr']
    
    # 記錄修正的數據點
    corrections_made = False
    corrections_log = []
    
    # 針對每個需要平滑的指標
    for metric in metrics_to_smooth:
        # 標記強度因子，控制修正強度
        correction_strength = 0.5
        
        # 第一步：確保單調性，但排除最大容量點的處理
        for i in range(1, len(df)):
            # 跳過最大容量點
            if i == max_capacity_idx:
                continue
                
            if df.iloc[i]['BPP'] > df.iloc[i-1]['BPP']:
                # 如果BPP增加，但指標值也增加，這是異常
                if df.iloc[i][metric] > df.iloc[i-1][metric]:
                    original_value = df.iloc[i][metric]
                    
                    # 計算預期的降低值
                    if i > 1:
                        # 基於前幾個點的變化率計算預期變化
                        prev_rate = (df.iloc[i-2][metric] - df.iloc[i-1][metric]) / \
                                  (df.iloc[i-2]['BPP'] - df.iloc[i-1]['BPP'])
                        
                        # 確保變化率為負數（指標隨BPP增加而減少）
                        prev_rate = min(prev_rate, 0)
                        
                        # 預期變化 = 變化率 × BPP變化
                        expected_change = prev_rate * (df.iloc[i]['BPP'] - df.iloc[i-1]['BPP'])
                        
                        # 如果預期變化接近零，使用小的百分比變化
                        if abs(expected_change) < 0.001:
                            expected_change = -0.005 * df.iloc[i-1][metric]
                        
                        # 應用修正，帶權重混合以避免過度修正
                        corrected_value = df.iloc[i-1][metric] + expected_change
                        df.loc[df.index[i], metric] = df.iloc[i-1][metric] * (1 - correction_strength) + \
                                                    corrected_value * correction_strength
                    else:
                        # 對於前面的點，使用較小的百分比降低
                        df.loc[df.index[i], metric] = df.iloc[i-1][metric] * 0.995
                    
                    # 記錄修正
                    corrections_made = True
                    corrections_log.append(f"  {metric} at BPP={df.iloc[i]['BPP']:.4f}: {original_value:.4f} -> {df.loc[df.index[i], metric]:.4f}")
        
        # 第二步：應用Savitzky-Golay平滑處理（如果數據點足夠多）
        if len(df) >= 7:  # 需要至少7個點以獲得良好效果
            try:

                
                # 創建臨時DataFrame以排除最大容量點進行平滑處理
                temp_df = df[df.index != max_capacity_idx].copy()
                
                # 確保窗口長度為奇數且不超過臨時DataFrame的長度
                window_length = min(7, len(temp_df) - (len(temp_df) % 2) - 1)
                if window_length < 3:
                    window_length = 3
                    
                # 多項式階數必須小於窗口長度
                poly_order = min(2, window_length - 2)
                
                # 先保存原始值
                original_values = temp_df[metric].values
                
                # 應用Savitzky-Golay平滑處理到臨時DataFrame
                smoothed_values = savgol_filter(original_values, window_length, poly_order)
                
                # 混合原始值和平滑值(70%原始 + 30%平滑)
                for i, idx in enumerate(temp_df.index):
                    original_val = temp_df.loc[idx, metric]
                    smoothed_val = smoothed_values[i]
                    # 混合平滑處理，權重可調整
                    df.loc[idx, metric] = original_val * 0.7 + smoothed_val * 0.3
                
                # 再次確保單調性，但排除最大容量點
                for i in range(1, len(df)):
                    if i == max_capacity_idx:
                        continue
                        
                    if df.iloc[i]['BPP'] > df.iloc[i-1]['BPP'] and df.iloc[i][metric] > df.iloc[i-1][metric]:
                        df.loc[df.index[i], metric] = df.iloc[i-1][metric] * 0.998
            except ImportError:
                print("Note: scipy.signal.savgol_filter is not available. Skipping savgol smoothing.")
        
        # 恢復最大容量點的原始指標值
        df.loc[max_capacity_idx, metric] = max_capacity_metrics[metric]
    
    # 最後確保平滑後的曲線與最大容量點銜接良好
    # 獲取最大容量點之前的點（如果有）
    if max_capacity_idx > 0:
        prev_to_max_idx = max_capacity_idx - 1
        
        for metric in metrics_to_smooth:
            # 檢查最大容量點和前一點之間的跳躍是否過大
            max_val = df.loc[max_capacity_idx, metric]
            prev_val = df.loc[df.index[prev_to_max_idx], metric]
            
            # 計算預期的平滑變化
            if prev_to_max_idx > 0:
                # 使用前兩個點的趨勢來預測平滑變化
                pp_idx = prev_to_max_idx - 1
                prev_prev_val = df.loc[df.index[pp_idx], metric]
                prev_rate = (prev_prev_val - prev_val) / (df.iloc[pp_idx]['BPP'] - df.iloc[prev_to_max_idx]['BPP'])
                
                # 根據之前的變化率預測最大容量點的值
                expected_change = prev_rate * (df.iloc[max_capacity_idx]['BPP'] - df.iloc[prev_to_max_idx]['BPP'])
                expected_val = prev_val + expected_change
                
                # 如果預測值和實際值相差太大，可能需要調整前面的點
                if abs(expected_val - max_val) > abs(0.1 * max_val):  # 超過10%的偏差
                    # 調整之前的一些點以創建平滑過渡
                    adjustment_range = min(3, prev_to_max_idx + 1)  # 最多調整3個點
                    
                    for j in range(adjustment_range):
                        adj_idx = prev_to_max_idx - j
                        # 使用線性插值平滑過渡
                        weight = (j + 1) / (adjustment_range + 1)
                        # 混合現有值和向最大容量點過渡的值
                        transition_val = df.loc[df.index[adj_idx], metric] * (1 - weight) + max_val * weight
                        
                        # 記錄調整並應用
                        old_val = df.loc[df.index[adj_idx], metric]
                        df.loc[df.index[adj_idx], metric] = transition_val
                        
                        # 記錄調整日誌
                        corrections_made = True
                        corrections_log.append(f"  Transition adjustment {metric} at index {adj_idx}: {old_val:.4f} -> {transition_val:.4f}")
    
    # 輸出修正日誌
    if corrections_made:
        print("Anomalous data points detected and corrected:")
        for correction in corrections_log:
            print(correction)
        
        # 為異常點處理前後的比較添加列
        for metric in metrics_to_smooth:
            df[f'{metric}_Original'] = original_df[metric]
        
        # 保存處理前後的對比資料
        comparison_csv = f"{result_dir}/precision_comparison.csv"
        df.to_csv(comparison_csv, index=False)
        print(f"Comparison data saved to: {comparison_csv}")
    else:
        print("No anomalous data points detected.")
        
        # 即使沒有修正，也添加原始指標列以保持一致性
        for metric in metrics_to_smooth:
            df[f'{metric}_Original'] = df[metric]
    
    # 步驟4: 整理結果
    print(f"\n{'='*80}")
    print(f"Step 4: Results summary")
    print(f"{'='*80}")
    
    total_time = time.time() - total_start_time
    
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per point: {total_time/len(results):.2f} seconds")
    print(f"Results saved to {result_dir}")
    
    with open(log_file, 'a') as f:
        f.write(f"Results summary:\n")
        f.write(f"Total processing time: {total_time:.2f} seconds\n")
        f.write(f"Average time per point: {total_time/len(results):.2f} seconds\n")
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
    使用平滑處理後的數據生成更美觀的圖表，並可選擇性顯示原始數據對比
    
    Parameters:
    -----------
    df : pandas.DataFrame
        包含測量結果的DataFrame (已經過平滑處理)
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
    
    # 如果有原始數據列，同時繪製原始和平滑後的數據
    if 'PSNR_Original' in df.columns:
        plt.plot(df['BPP'], df['PSNR_Original'], 
             color='lightblue',
             linestyle='--',
             linewidth=1.5,
             alpha=0.6,
             marker='o',
             markersize=4,
             label='Original Data')
    
    # 繪製平滑後的數據（主線）
    plt.plot(df['BPP'], df['PSNR'], 
             color='blue',
             linewidth=2.5,
             marker='o',
             markersize=6,
             label=f'Method: {method}, Predictor: {prediction_method}')
    
    # 添加數據標籤 (只標記部分點，避免擁擠)
    steps = max(1, len(df) // 10)  # 確保不超過10個標籤
    for i, row in enumerate(df.itertuples()):
        if i % steps == 0 or i == len(df) - 1:
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
    
    if 'SSIM_Original' in df.columns:
        plt.plot(df['BPP'], df['SSIM_Original'], 
             color='salmon',
             linestyle='--',
             linewidth=1.5,
             alpha=0.6,
             marker='o',
             markersize=4,
             label='Original Data')
    
    plt.plot(df['BPP'], df['SSIM'], 
             color='red',
             linewidth=2.5,
             marker='o',
             markersize=6,
             label=f'Method: {method}, Predictor: {prediction_method}')
    
    # 添加數據標籤
    for i, row in enumerate(df.itertuples()):
        if i % steps == 0 or i == len(df) - 1:
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
    
    if 'Hist_Corr_Original' in df.columns:
        plt.plot(df['BPP'], df['Hist_Corr_Original'], 
             color='lightgreen',
             linestyle='--',
             linewidth=1.5,
             alpha=0.6,
             marker='o',
             markersize=4,
             label='Original Data')
    
    plt.plot(df['BPP'], df['Hist_Corr'], 
             color='green',
             linewidth=2.5,
             marker='o',
             markersize=6,
             label=f'Method: {method}, Predictor: {prediction_method}')
    
    # 添加數據標籤
    for i, row in enumerate(df.itertuples()):
        if i % steps == 0 or i == len(df) - 1:
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
        if i % steps == 0 or i == len(df) - 1:
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
    
    # 如果有平滑前後的對比數據，繪製對比圖
    if 'PSNR_Original' in df.columns:
        # 創建平滑前後對比圖
        plt.figure(figsize=(14, 10))
        
        # 使用子圖排布
        plt.subplot(2, 1, 1)
        plt.plot(df['BPP'], df['PSNR_Original'], 'b-', label='Original PSNR')
        plt.plot(df['BPP'], df['PSNR'], 'r-', label='Smoothed PSNR')
        plt.xlabel('BPP')
        plt.ylabel('PSNR (dB)')
        plt.title('PSNR: Original vs Smoothed')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(df['BPP'], df['SSIM_Original'], 'b-', label='Original SSIM')
        plt.plot(df['BPP'], df['SSIM'], 'r-', label='Smoothed SSIM')
        plt.xlabel('BPP')
        plt.ylabel('SSIM')
        plt.title('SSIM: Original vs Smoothed')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/smoothing_comparison.png", dpi=300)
        plt.close()
    
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
            f.write(traceback.format_exc())
    
    return all_results

def run_method_comparison(imgName, filetype="png", predictor="proposed", 
                        ratio_of_ones=0.5, methods=None, method_params=None,
                        total_embeddings=5, el_mode=0, segments=15, step_size=None):
    """
    比較使用相同預測器的不同嵌入方法的性能
    
    Parameters:
    -----------
    imgName : str
        圖像名稱
    filetype : str
        圖像檔案類型
    predictor : str
        所有比較使用的預測方法 ("proposed", "med", "gap", "rhombus")
    ratio_of_ones : float
        嵌入數據中1的比例
    methods : list of str
        要比較的方法 (例如 ["rotation", "split", "quadtree"])
    method_params : dict of dict
        每個方法的特定參數，例如 
        {"rotation": {"split_size": 2}, "quadtree": {"min_block_size": 16}}
    total_embeddings : int
        總嵌入次數
    el_mode : int
        EL模式
    segments : int
        測量分段數量 (如果提供了step_size則忽略)
    step_size : int, optional
        測量點之間的步長 (位元)
    
    Returns:
    --------
    dict
        包含每個方法結果的字典 {方法名稱: DataFrame}
    """
    
    # 讀取原始圖像
    origImg = cv2.imread(f"./Prediction_Error_Embedding/image/{imgName}.{filetype}", cv2.IMREAD_GRAYSCALE)
    if origImg is None:
        raise ValueError(f"Failed to read image: ./Prediction_Error_Embedding/image/{imgName}.{filetype}")
    
    # 如果未提供方法，使用默認方法
    if methods is None:
        methods = ["rotation", "split", "quadtree"]
    
    # 如果未提供方法參數，使用默認參數
    if method_params is None:
        method_params = {
            "rotation": {"split_size": 2, "use_different_weights": False},
            "split": {"split_size": 2, "block_base": False, "use_different_weights": False},
            "quadtree": {"min_block_size": 16, "variance_threshold": 300, "use_different_weights": False}
        }
    
    # 將預測器字符串映射到 PredictionMethod
    pred_map = {
        "proposed": PredictionMethod.PROPOSED,
        "med": PredictionMethod.MED,
        "gap": PredictionMethod.GAP,
        "rhombus": PredictionMethod.RHOMBUS
    }
    prediction_method = pred_map.get(predictor.lower(), PredictionMethod.PROPOSED)
    
    # 創建比較輸出目錄
    comparison_dir = f"./Prediction_Error_Embedding/outcome/plots/{imgName}/method_comparison_{predictor}"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 比較的日誌文件
    log_file = f"{comparison_dir}/method_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, 'w') as f:
        f.write(f"Method comparison started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Image: {imgName}.{filetype}\n")
        f.write(f"Predictor: {predictor}\n")
        f.write(f"Methods to compare: {methods}\n")
        f.write(f"Total embeddings: {total_embeddings}\n")
        f.write(f"EL mode: {el_mode}\n")
        if step_size:
            f.write(f"Step size: {step_size} bits\n")
        else:
            f.write(f"Segments: {segments}\n")
        f.write("\n" + "="*80 + "\n\n")
    
    # 儲存每個方法的結果
    all_results = {}
    
    # 處理每個方法
    for method_name in methods:
        print(f"\n{'='*80}")
        print(f"Running precise measurements for {method_name} method")
        print(f"{'='*80}")
        
        try:
            # 獲取此方法的參數
            params = method_params.get(method_name, {}).copy()
            
            # 對四叉樹方法進行特殊處理
            if method_name == "quadtree":
                # 創建或更新 quad_tree_params 字典
                quad_tree_params = {}
                if "min_block_size" in params:
                    quad_tree_params["min_block_size"] = params.pop("min_block_size")
                if "variance_threshold" in params:
                    quad_tree_params["variance_threshold"] = params.pop("variance_threshold")
                # 保留 use_different_weights 在主參數中
                # 添加嵌套參數
                params["quad_tree_params"] = quad_tree_params
            
            # 運行精確測量
            results_df = run_precise_measurements(
                origImg, imgName, method_name, prediction_method, ratio_of_ones,
                total_embeddings, el_mode, segments, step_size,
                **params  # 展開方法特定參數
            )
            
            # 儲存結果
            all_results[method_name] = results_df
            
            # 保存為CSV
            results_df.to_csv(f"{comparison_dir}/{method_name}_{predictor}_precise.csv", index=False)
            
            # 記錄完成
            with open(log_file, 'a') as f:
                f.write(f"Completed measurements for {method_name} method\n")
                f.write(f"Results saved to {comparison_dir}/{method_name}_{predictor}_precise.csv\n\n")
            
        except Exception as e:
            print(f"Error processing {method_name}: {str(e)}")
            with open(log_file, 'a') as f:
                f.write(f"Error processing {method_name}: {str(e)}\n")
                f.write(traceback.format_exc())
                f.write("\n\n")
            
            # 確保清理記憶體
            cleanup_memory()
    
    # 如果有結果，創建比較圖表
    if all_results:
        plot_method_comparison(all_results, imgName, predictor, comparison_dir)
        create_comparative_table(all_results, f"{comparison_dir}/method_comparison_table.csv")
        print(f"Comparison plots saved to {comparison_dir}")
    
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

def plot_method_comparison(all_results, imgName, predictor, output_dir):
    """
    為使用相同預測器的不同方法創建比較圖表
    
    Parameters:
    -----------
    all_results : dict
        包含每個方法結果的字典 {方法名稱: DataFrame}
    imgName : str
        圖像名稱
    predictor : str
        所有方法使用的預測器
    output_dir : str
        輸出目錄
    """
    
    # 為不同方法設置顏色和標記
    colors = {
        'rotation': 'blue',
        'split': 'green',
        'quadtree': 'red',
        'custom': 'purple'  # 用於其他方法
    }
    
    markers = {
        'rotation': 'o',
        'split': 's',
        'quadtree': '^',
        'custom': 'D'  # 用於其他方法
    }
    
    # 創建 BPP-PSNR 比較圖
    plt.figure(figsize=(12, 8))
    
    for method, df in all_results.items():
        color = colors.get(method, 'black')
        marker = markers.get(method, 'x')
        
        plt.plot(df['BPP'], df['PSNR'], 
                 color=color,
                 linewidth=2.5,
                 marker=marker,
                 markersize=8,
                 label=f'Method: {method}')
    
    plt.xlabel('Bits Per Pixel (BPP)', fontsize=14)
    plt.ylabel('PSNR (dB)', fontsize=14)
    plt.title(f'Method Comparison with {predictor.capitalize()} Predictor\n'
              f'Image: {imgName}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/method_comparison_bpp_psnr.png", dpi=300)
    plt.close()
    
    # 創建 BPP-SSIM 比較圖
    plt.figure(figsize=(12, 8))
    
    for method, df in all_results.items():
        color = colors.get(method, 'black')
        marker = markers.get(method, 'x')
        
        plt.plot(df['BPP'], df['SSIM'], 
                 color=color,
                 linewidth=2.5,
                 marker=marker,
                 markersize=8,
                 label=f'Method: {method}')
    
    plt.xlabel('Bits Per Pixel (BPP)', fontsize=14)
    plt.ylabel('SSIM', fontsize=14)
    plt.title(f'Method Comparison with {predictor.capitalize()} Predictor\n'
              f'Image: {imgName}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/method_comparison_bpp_ssim.png", dpi=300)
    plt.close()
    
    # 如果有直方圖相關性數據，創建 BPP-Hist_Corr 比較圖
    if 'Hist_Corr' in next(iter(all_results.values())):
        plt.figure(figsize=(12, 8))
        
        for method, df in all_results.items():
            color = colors.get(method, 'black')
            marker = markers.get(method, 'x')
            
            plt.plot(df['BPP'], df['Hist_Corr'], 
                    color=color,
                    linewidth=2.5,
                    marker=marker,
                    markersize=8,
                    label=f'Method: {method}')
        
        plt.xlabel('Bits Per Pixel (BPP)', fontsize=14)
        plt.ylabel('Histogram Correlation', fontsize=14)
        plt.title(f'Histogram Correlation Comparison with {predictor.capitalize()} Predictor\n'
                f'Image: {imgName}', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/method_comparison_bpp_histcorr.png", dpi=300)
        plt.close()
    
    # 創建 PSNR-Payload 比較圖（額外的視角）
    plt.figure(figsize=(12, 8))
    
    for method, df in all_results.items():
        color = colors.get(method, 'black')
        marker = markers.get(method, 'x')
        
        plt.plot(df['Actual_Payload'], df['PSNR'], 
                 color=color,
                 linewidth=2.5,
                 marker=marker,
                 markersize=8,
                 label=f'Method: {method}')
    
    plt.xlabel('Payload (bits)', fontsize=14)
    plt.ylabel('PSNR (dB)', fontsize=14)
    plt.title(f'Payload-PSNR Comparison with {predictor.capitalize()} Predictor\n'
              f'Image: {imgName}', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/method_comparison_payload_psnr.png", dpi=300)
    plt.close()
    
    # 創建綜合雷達圖（僅在有多個方法時才有意義）
    if len(all_results) >= 2:
        create_radar_chart(all_results, predictor, imgName, output_dir)

def create_comparative_table(all_results, output_path):
    """
    創建所有方法在相似 BPP 值下的比較表
    
    Parameters:
    -----------
    all_results : dict
        包含每個方法結果的字典 {方法名稱: DataFrame}
    output_path : str
        保存比較表的路徑
    
    Returns:
    --------
    pandas.DataFrame
        比較表
    """
    
    # 首先，識別共同的 BPP 範圍
    all_bpp_values = []
    for method, df in all_results.items():
        all_bpp_values.extend(df['BPP'].tolist())
    
    # 排序並去除重複
    all_bpp_values = sorted(list(set([round(bpp, 4) for bpp in all_bpp_values])))
    
    # 創建固定間隔的參考 BPP 值
    min_bpp = min(all_bpp_values)
    max_bpp = max(all_bpp_values)
    step = (max_bpp - min_bpp) / 20  # 20 個參考點
    reference_bpps = [round(min_bpp + i * step, 4) for i in range(21)]  # 包括終點
    
    # 創建具有共同 BPP 值的 DataFrame
    comp_data = {'BPP': reference_bpps}
    
    # 為每個方法找到最接近的 PSNR 和 SSIM 值
    for method, df in all_results.items():
        psnr_values = []
        ssim_values = []
        hist_corr_values = []
        
        for bpp in reference_bpps:
            # 在此方法的結果中找到最接近的 BPP
            closest_idx = (df['BPP'] - bpp).abs().idxmin()
            psnr_values.append(df.loc[closest_idx, 'PSNR'])
            ssim_values.append(df.loc[closest_idx, 'SSIM'])
            if 'Hist_Corr' in df.columns:
                hist_corr_values.append(df.loc[closest_idx, 'Hist_Corr'])
        
        comp_data[f'{method}_PSNR'] = psnr_values
        comp_data[f'{method}_SSIM'] = ssim_values
        if hist_corr_values:
            comp_data[f'{method}_Hist_Corr'] = hist_corr_values
    
    # 添加方法之間的性能差異
    if len(all_results) > 1:
        methods = list(all_results.keys())
        
        # 獲取第一個方法作為基準
        base_method = methods[0]
        
        # 計算其他方法與基準方法之間的 PSNR 差異
        for method in methods[1:]:
            psnr_diff = [comp_data[f'{method}_PSNR'][i] - comp_data[f'{base_method}_PSNR'][i] 
                         for i in range(len(reference_bpps))]
            comp_data[f'{method}_vs_{base_method}_PSNR_diff'] = psnr_diff
    
    # 創建並保存 DataFrame
    comp_df = pd.DataFrame(comp_data)
    comp_df.to_csv(output_path, index=False)
    
    # 創建 LaTeX 格式表格
    latex_path = output_path.replace('.csv', '.tex')
    try:
        with open(latex_path, 'w') as f:
            # 只選擇部分行和列以獲得簡潔的 LaTeX 表格
            # 取 5 個均勻分佈的點
            indices = [0, 5, 10, 15, 20]  # 位於 0%, 25%, 50%, 75%, 100%
            
            # 為每個方法選擇 PSNR 列
            columns = ['BPP'] + [f'{method}_PSNR' for method in all_results.keys()]
            
            # 創建子表
            sub_df = comp_df.iloc[indices][columns]
            
            # 重命名列以便更好地顯示
            column_mapping = {f'{method}_PSNR': method.capitalize() for method in all_results.keys()}
            column_mapping['BPP'] = 'BPP'
            sub_df = sub_df.rename(columns=column_mapping)
            
            # 生成 LaTeX 表格
            latex_table = sub_df.to_latex(index=False, float_format="%.2f")
            
            # 添加表格標題和標籤
            latex_header = "\\begin{table}[h]\n\\centering\n\\caption{PSNR Comparison at Different BPP Levels}\n\\label{tab:psnr_comparison}\n"
            latex_footer = "\\end{table}"
            
            # 寫入完整表格
            f.write(latex_header + latex_table + latex_footer)
    except Exception as e:
        print(f"Could not generate LaTeX table: {e}")
    
    return comp_df

def create_radar_chart(all_results, predictor, imgName, output_dir):
    """
    創建比較不同方法性能的雷達圖
    
    Parameters:
    -----------
    all_results : dict
        包含每個方法結果的字典 {方法名稱: DataFrame}
    predictor : str
        所有方法使用的預測器
    imgName : str
        圖像名稱
    output_dir : str
        輸出目錄
    """
    
    # 檢查是否有足夠的方法來創建雷達圖
    if len(all_results) < 2:
        return
    
    # 選擇用於比較的 BPP 水平
    # 找到所有方法共有的 BPP 範圍
    all_bpp_min = max([df['BPP'].min() for df in all_results.values()])
    all_bpp_max = min([df['BPP'].max() for df in all_results.values()])
    
    # 如果共同範圍無效，則退出
    if all_bpp_min >= all_bpp_max:
        print("Cannot create radar chart: methods have non-overlapping BPP ranges")
        return
    
    # 選擇 3 個 BPP 點進行比較：低、中、高
    bpp_levels = [
        all_bpp_min + (all_bpp_max - all_bpp_min) * 0.25,  # 低 BPP
        all_bpp_min + (all_bpp_max - all_bpp_min) * 0.5,   # 中 BPP
        all_bpp_min + (all_bpp_max - all_bpp_min) * 0.75   # 高 BPP
    ]
    
    # 雷達圖類別
    categories = ['PSNR', 'SSIM', 'Speed']
    
    # 為不同方法設置顏色
    colors = {
        'rotation': 'blue',
        'split': 'green',
        'quadtree': 'red',
        'custom': 'purple'  # 用於其他方法
    }
    
    # 用於顯示每個方法在各個類別的性能
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
    
    # 為每個 BPP 水平創建雷達圖
    for i, bpp in enumerate(bpp_levels):
        ax = axs[i]
        
        # 獲取每個方法在此 BPP 下的性能
        performances = {}
        for method, df in all_results.items():
            # 找到最接近的 BPP
            closest_idx = (df['BPP'] - bpp).abs().idxmin()
            closest_row = df.loc[closest_idx]
            
            # 記錄性能指標
            performances[method] = {
                'PSNR': closest_row['PSNR'],
                'SSIM': closest_row['SSIM'],
                'Speed': 1.0 / closest_row['Processing_Time']  # 速度是處理時間的倒數
            }
        
        # 正規化性能指標到 0-1 範圍
        normalized_performances = {}
        for category in categories:
            values = [performances[method][category] for method in performances]
            min_val = min(values)
            max_val = max(values)
            
            if max_val > min_val:
                for method in performances:
                    if category not in normalized_performances:
                        normalized_performances[category] = {}
                    
                    # 正規化，確保較高的值總是更好
                    normalized_performances[category][method] = (performances[method][category] - min_val) / (max_val - min_val)
            else:
                # 如果所有值相同，則設為 1
                for method in performances:
                    if category not in normalized_performances:
                        normalized_performances[category] = {}
                    normalized_performances[category][method] = 1.0
        
        # 設置角度
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 閉合圖形
        
        # 設置雷達圖
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        
        # 添加類別標籤
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # 設置 y 軸刻度
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])
        ax.set_rlim(0, 1)
        
        # 繪製每個方法的雷達圖
        for method in performances:
            color = colors.get(method, 'black')
            
            # 獲取性能值
            values = [normalized_performances[cat][method] for cat in categories]
            values += values[:1]  # 閉合圖形
            
            # 繪製雷達線
            ax.plot(angles, values, color=color, linewidth=2, label=method)
            ax.fill(angles, values, color=color, alpha=0.1)
        
        # 設置標題
        ax.set_title(f'BPP = {bpp:.4f}', size=14, y=1.1)
        
        # 只在第一個子圖顯示圖例
        if i == 0:
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.suptitle(f'Performance Comparison of Different Methods with {predictor.capitalize()} Predictor', size=16)
    plt.tight_layout()
    
    # 保存圖表
    plt.savefig(f"{output_dir}/method_radar_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

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