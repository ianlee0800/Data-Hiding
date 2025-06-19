import pandas as pd
import cv2
import prettytable as pt
import time
import traceback
from scipy.signal import savgol_filter
from tqdm import tqdm
from datetime import datetime
import cupy as cp

from embedding import (
    pee_process_with_rotation_cuda,
    pee_process_with_split_cuda
)
from quadtree import pee_process_with_quadtree_cuda

# 🔧 更新：使用新的工具模組
from utils import (
    MeasurementConstants,
    PathConstants,
    DataConverter,
    MeasurementPointGenerator,
    calculate_quality_metrics_unified,
    ensure_dir,
    cleanup_memory
)

# 🔧 更新：從 image_processing 導入 get_image_info
from image_processing import (
    save_image,
    PredictionMethod,
    get_image_info
)

# =============================================================================
# 輔助函數：日誌管理
# =============================================================================

class MeasurementLogger:
    """簡化的測量日誌管理器"""
    
    def __init__(self, log_file, img_name, method_name):
        self.log_file = log_file
        self.img_name = img_name
        self.method_name = method_name
        
    def log_header(self, img_info, measurement_mode, total_embeddings, el_mode, use_different_weights):
        """記錄測量開始信息"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Precise measurement run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image: {self.img_name} ({img_info['description']})\n")
            f.write(f"Method: {self.method_name}\n")
            f.write(f"Pixel count for BPP calculation: {img_info['pixel_count']}\n")
            f.write(f"Measurement mode: {measurement_mode}\n")
            f.write(f"Total embeddings: {total_embeddings}\n")
            f.write(f"EL mode: {el_mode}\n")
            f.write(f"Use different weights: {use_different_weights}\n")
            f.write("\n" + "="*80 + "\n\n")
    
    def log_step(self, step_name, details):
        """記錄測量步驟"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{step_name}:\n")
            for key, value in details.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

# =============================================================================
# 第一步：最大容量測量
# =============================================================================

def _find_maximum_capacity(origImg, method, prediction_method, ratio_of_ones, 
                          total_embeddings, el_mode, split_size, block_base, 
                          quad_tree_params, use_different_weights, logger):
    """
    步驟1: 找出最大嵌入容量
    
    Returns:
    --------
    tuple: (final_img_max, max_payload, stages_max, max_run_time)
    """
    print(f"\n{'='*80}")
    print(f"Step 1: Finding maximum payload capacity")
    print(f"{'='*80}")
    
    start_time = time.time()
    final_img_max, max_payload, stages_max = run_embedding_with_target(
        origImg, method, prediction_method, ratio_of_ones, 
        total_embeddings, el_mode, target_payload_size=-1,
        split_size=split_size, block_base=block_base, 
        quad_tree_params=quad_tree_params,
        use_different_weights=use_different_weights
    )
    max_run_time = time.time() - start_time
    
    # 記錄到日誌
    logger.log_step("Step 1: Maximum capacity found", {
        "Maximum payload": f"{max_payload} bits",
        "Time taken": f"{max_run_time:.2f} seconds"
    })
    
    print(f"Maximum payload: {max_payload} bits")
    print(f"Time taken: {max_run_time:.2f} seconds")
    
    return final_img_max, max_payload, stages_max, max_run_time

# =============================================================================
# 第二步：測量點生成
# =============================================================================

def _generate_measurement_points(max_payload, step_size, segments, logger):
    """
    步驟2: 計算測量點
    
    Returns:
    --------
    tuple: (payload_points, measurement_mode)
    """
    print(f"\n{'='*80}")
    print(f"Step 2: Calculating measurement points")
    print(f"{'='*80}")
    
    # 🔧 更新：使用新的工具類生成測量點
    payload_points, measurement_mode, mode_description = MeasurementPointGenerator.generate_points(
        max_payload, step_size, segments
    )
    
    print(f"Measurement mode: {measurement_mode}")
    print(f"Mode description: {mode_description}")
    print(f"Total measurement points: {len(payload_points) + 1} (including max capacity)")
    
    # 記錄到日誌
    logger.log_step("Step 2: Measurement points generated", {
        "Measurement mode": measurement_mode,
        "Mode description": mode_description,
        "Points generated": len(payload_points),
        "Total points": len(payload_points) + 1
    })
    
    return payload_points, measurement_mode

# =============================================================================
# 第三步：批量測量執行
# =============================================================================

def _run_measurement_batch(origImg, payload_points, method, prediction_method, 
                          ratio_of_ones, total_embeddings, el_mode, split_size, 
                          block_base, quad_tree_params, use_different_weights, 
                          img_info, measurement_mode, result_dir, logger):
    """
    步驟3: 為每個目標點運行嵌入算法
    
    Returns:
    --------
    list: 測量結果列表
    """
    print(f"\n{'='*80}")
    print(f"Step 3: Running embedding algorithm for each target point")
    print(f"Processing {len(payload_points)} measurement points...")
    print(f"{'='*80}")
    
    results = []
    
    for i, target in enumerate(tqdm(payload_points, desc="處理測量點")):
        percentage = target / max(payload_points + [target]) * 100
        
        print(f"\nRunning point {i+1}/{len(payload_points)}: {target} bits ({percentage:.1f}% of max)")
        
        start_time = time.time()
        final_img, actual_payload, stages = run_embedding_with_target(
            origImg, method, prediction_method, ratio_of_ones, 
            total_embeddings, el_mode, target_payload_size=target,
            split_size=split_size, block_base=block_base, 
            quad_tree_params=quad_tree_params,
            use_different_weights=use_different_weights
        )
        run_time = time.time() - start_time
        
        # 🔧 更新：使用新的統一品質指標計算函數
        psnr, ssim, hist_corr = calculate_quality_metrics_unified(origImg, final_img, img_info)
        
        # 檢查 PSNR 是否異常
        is_psnr_suspicious = False
        if len(results) > 0:
            last_result = results[-1]
            current_bpp = actual_payload / img_info['pixel_count']
            if (current_bpp > last_result['BPP'] and psnr > last_result['PSNR']):
                is_psnr_suspicious = True
                print(f"  Warning: Suspicious PSNR value detected: {psnr:.2f} > previous {last_result['PSNR']:.2f}")
        
        # 計算BPP
        bpp = actual_payload / img_info['pixel_count']
        
        # 記錄結果
        result = {
            'Target_Percentage': percentage,
            'Target_Payload': target,
            'Actual_Payload': actual_payload,
            'BPP': bpp,
            'PSNR': psnr,
            'SSIM': ssim,
            'Hist_Corr': hist_corr,
            'Processing_Time': run_time,
            'Suspicious': is_psnr_suspicious,
            'Measurement_Mode': measurement_mode,
            'Image_Type': img_info['type_name'],
            'Pixel_Count': img_info['pixel_count']
        }
        results.append(result)
        
        # 保存嵌入圖像
        if img_info['is_color']:
            cv2.imwrite(f"{result_dir}/embedded_{int(percentage)}pct.png", final_img)
        else:
            save_image(final_img, f"{result_dir}/embedded_{int(percentage)}pct.png")
        
        print(f"  Actual: {actual_payload} bits")
        print(f"  BPP: {bpp:.6f}")
        print(f"  PSNR: {psnr:.2f}")
        print(f"  SSIM: {ssim:.4f}")
        print(f"  Time: {run_time:.2f} seconds")
        
        # 🔧 更新：使用新的記憶體清理函數
        cleanup_memory()
    
    # 記錄批量測量完成
    logger.log_step("Step 3: Batch measurement completed", {
        "Total points processed": len(results),
        "Average processing time": f"{sum(r['Processing_Time'] for r in results) / len(results):.2f} seconds"
    })
    
    return results

# =============================================================================
# 第四步：結果數據處理
# =============================================================================

def _process_measurement_results(results, max_capacity_result, img_info):
    """
    步驟4: 數據平滑處理
    
    Returns:
    --------
    pandas.DataFrame: 處理後的結果
    """
    print(f"\n{'='*80}")
    print(f"Step 4: Processing data (preserving max capacity point)")
    print(f"{'='*80}")
    
    # 添加最大容量結果並排序
    all_results = results + [max_capacity_result]
    all_results.sort(key=lambda x: x['BPP'])
    
    # 轉換為DataFrame
    df = pd.DataFrame(all_results)
    
    # 保留原始數據
    original_df = df.copy()
    
    # 標記最大容量點
    max_capacity_idx = df[df['Target_Percentage'] == 100.0].index[0]
    
    # 針對彩色圖像調整平滑參數
    if img_info['is_color']:
        print("Applying color image specific data processing...")
        correction_strength = MeasurementConstants.COLOR_CORRECTION_STRENGTH
    else:
        correction_strength = MeasurementConstants.DEFAULT_CORRECTION_STRENGTH
    
    # 應用數據平滑處理
    df = _apply_data_smoothing(df, max_capacity_idx, correction_strength, original_df)
    
    return df

def _apply_data_smoothing(df, max_capacity_idx, correction_strength, original_df):
    """應用數據平滑處理"""
    metrics_to_smooth = ['PSNR', 'SSIM', 'Hist_Corr']
    corrections_made = False
    corrections_log = []
    
    # 存儲最大容量點的原始指標值
    max_capacity_metrics = {
        metric: df.loc[max_capacity_idx, metric] for metric in metrics_to_smooth
    }
    
    # 針對每個需要平滑的指標
    for metric in metrics_to_smooth:
        # 確保單調性，但排除最大容量點的處理
        for i in range(1, len(df)):
            if i == max_capacity_idx:
                continue
                
            if df.iloc[i]['BPP'] > df.iloc[i-1]['BPP']:
                if df.iloc[i][metric] > df.iloc[i-1][metric]:
                    original_value = df.iloc[i][metric]
                    
                    # 計算預期的降低值
                    if i > 1:
                        prev_rate = (df.iloc[i-2][metric] - df.iloc[i-1][metric]) / \
                                  (df.iloc[i-2]['BPP'] - df.iloc[i-1]['BPP'])
                        prev_rate = min(prev_rate, 0)
                        expected_change = prev_rate * (df.iloc[i]['BPP'] - df.iloc[i-1]['BPP'])
                        
                        if abs(expected_change) < 0.001:
                            expected_change = -0.005 * df.iloc[i-1][metric]
                        
                        corrected_value = df.iloc[i-1][metric] + expected_change
                        df.loc[df.index[i], metric] = df.iloc[i-1][metric] * (1 - correction_strength) + \
                                                    corrected_value * correction_strength
                    else:
                        df.loc[df.index[i], metric] = df.iloc[i-1][metric] * 0.995
                    
                    corrections_made = True
                    corrections_log.append(f"  {metric} at BPP={df.iloc[i]['BPP']:.4f}: {original_value:.4f} -> {df.loc[df.index[i], metric]:.4f}")
        
        # 恢復最大容量點的原始指標值
        df.loc[max_capacity_idx, metric] = max_capacity_metrics[metric]
    
    # 輸出修正日誌
    if corrections_made:
        print("Anomalous data points detected and corrected:")
        for correction in corrections_log:
            print(correction)
        
        # 為異常點處理前後的比較添加列
        for metric in metrics_to_smooth:
            df[f'{metric}_Original'] = original_df[metric]
    else:
        print("No anomalous data points detected.")
        for metric in metrics_to_smooth:
            df[f'{metric}_Original'] = df[metric]
    
    return df

# =============================================================================
# 第五步：結果保存和總結
# =============================================================================

def _save_measurement_results(df, result_dir, imgName, method, method_name, total_time, img_info, logger):
    """
    步驟5: 保存結果和總結
    """
    print(f"\n{'='*80}")
    print(f"Step 5: Results summary")
    print(f"{'='*80}")
    
    # 顯示總結信息
    measurement_mode = df['Measurement_Mode'].iloc[0] if len(df) > 0 else "unknown"
    total_points = len(df)
    avg_time = total_time / total_points if total_points > 0 else 0
    
    summary_info = {
        "Image type": img_info['type_name'].capitalize(),
        "Pixel count used for BPP": img_info['pixel_count'],
        "Measurement mode used": measurement_mode,
        "Total data points generated": total_points,
        "Total processing time": f"{total_time:.2f} seconds",
        "Average time per point": f"{avg_time:.2f} seconds",
        "Results saved to": result_dir
    }
    
    for key, value in summary_info.items():
        print(f"{key}: {value}")
    
    # 保存結果到CSV
    csv_path = f"{result_dir}/precise_measurements.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # 記錄到日誌
    logger.log_step("Step 5: Results saved", summary_info)
    
    return csv_path

# =============================================================================
# 重構後的主函數
# =============================================================================

def run_precise_measurements(origImg, imgName, method, prediction_method, ratio_of_ones, 
                           total_embeddings=5, el_mode=0, segments=15, step_size=None, 
                           use_different_weights=False, split_size=2, block_base=False, 
                           quad_tree_params=None):
    """
    重構版的精確測量函數 - 主函數現在只負責協調各個步驟
    """
    # 記錄總運行開始時間
    total_start_time = time.time()
    
    # 🔧 更新：使用新的圖像信息獲取函數
    img_info = get_image_info(origImg)
    
    # 獲取方法名稱
    if isinstance(prediction_method, str):
        method_name = prediction_method
    else:
        method_name = prediction_method.value
    
    # 🔧 更新：使用新的路徑管理工具
    from utils import get_precise_measurement_directories
    directories = get_precise_measurement_directories(imgName, method_name)
    result_dir = directories['result']
    ensure_dir(f"{result_dir}/dummy.txt")
    
    # 設置日誌
    log_file = f"{result_dir}/precise_measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = MeasurementLogger(log_file, imgName, method_name)
    
    # 確定測量模式
    payload_points, measurement_mode, _ = MeasurementPointGenerator.generate_points(
        1000000, step_size, segments  # 暫時用大數值，後面會用實際最大容量
    )
    
    # 記錄開始信息
    logger.log_header(img_info, measurement_mode, total_embeddings, el_mode, use_different_weights)
    
    print(f"Color image detected: {img_info['pixel_count']} pixel positions" if img_info['is_color'] 
          else f"Grayscale image detected: {img_info['pixel_count']} pixels")
    
    try:
        # 步驟1: 找出最大嵌入容量
        final_img_max, max_payload, stages_max, max_run_time = _find_maximum_capacity(
            origImg, method, prediction_method, ratio_of_ones, 
            total_embeddings, el_mode, split_size, block_base, 
            quad_tree_params, use_different_weights, logger
        )
        
        # 計算最大容量的品質指標
        psnr_max, ssim_max, hist_corr_max = calculate_quality_metrics_unified(origImg, final_img_max, img_info)
        max_bpp = max_payload / img_info['pixel_count']
        
        # 創建最大容量結果
        max_capacity_result = {
            'Target_Percentage': 100.0,
            'Target_Payload': max_payload,
            'Actual_Payload': max_payload,
            'BPP': max_bpp,
            'PSNR': psnr_max,
            'SSIM': ssim_max,
            'Hist_Corr': hist_corr_max,
            'Processing_Time': max_run_time,
            'Suspicious': False,
            'Image_Type': img_info['type_name'],
            'Pixel_Count': img_info['pixel_count']
        }
        
        print(f"Max BPP: {max_bpp:.6f}")
        print(f"Max PSNR: {psnr_max:.2f}")
        print(f"Max SSIM: {ssim_max:.4f}")
        
        # 保存最大容量的嵌入圖像
        if img_info['is_color']:
            cv2.imwrite(f"{result_dir}/embedded_100pct.png", final_img_max)
        else:
            save_image(final_img_max, f"{result_dir}/embedded_100pct.png")
        
        # 步驟2: 重新生成測量點（使用實際最大容量）
        payload_points, measurement_mode = _generate_measurement_points(
            max_payload, step_size, segments, logger
        )
        
        # 步驟3: 批量測量執行
        results = _run_measurement_batch(
            origImg, payload_points, method, prediction_method, 
            ratio_of_ones, total_embeddings, el_mode, split_size, 
            block_base, quad_tree_params, use_different_weights, 
            img_info, measurement_mode, result_dir, logger
        )
        
        # 步驟4: 數據處理和平滑
        df = _process_measurement_results(results, max_capacity_result, img_info)
        
        # 步驟5: 保存結果
        total_time = time.time() - total_start_time
        csv_path = _save_measurement_results(
            df, result_dir, imgName, method, method_name, total_time, img_info, logger
        )
        
        # 確認使用的測量模式
        if step_size is not None and step_size > 0:
            print(f"Confirmed: Used step_size={step_size} bits for {method_name}")
        else:
            print(f"Confirmed: Used segments={segments} for {method_name}")
        
        return df
        
    except Exception as e:
        print(f"Error in precise measurements: {str(e)}")
        logger.log_step("Error occurred", {"Error": str(e)})
        raise e

# =============================================================================
# 簡化測量函數（重構版）
# =============================================================================

def run_simplified_precise_measurements(origImg, imgName, method, prediction_method, ratio_of_ones, 
                                       total_embeddings=5, el_mode=0, segments=15, step_size=None, 
                                       use_different_weights=False, split_size=2, block_base=False, 
                                       quad_tree_params=None):
    """
    簡化版精確測量函數 - 重用主函數的邏輯，但不保存圖像和圖表
    """
    # 🔧 更新：使用新的圖像信息獲取函數
    img_info = get_image_info(origImg)
    
    # 獲取方法名稱
    method_name = prediction_method.value
    
    # 設置簡化的目錄結構
    from utils import get_precise_measurement_directories
    directories = get_precise_measurement_directories(imgName, method_name)
    result_dir = directories['data']  # 使用data目錄而不是result目錄
    ensure_dir(f"{result_dir}/dummy.txt")
    
    # 設置日誌
    log_file = f"{result_dir}/simplified_measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = MeasurementLogger(log_file, imgName, method_name)
    
    total_start_time = time.time()
    
    try:
        # 步驟1: 找出最大嵌入容量（重用函數）
        final_img_max, max_payload, stages_max, max_run_time = _find_maximum_capacity(
            origImg, method, prediction_method, ratio_of_ones, 
            total_embeddings, el_mode, split_size, block_base, 
            quad_tree_params, use_different_weights, logger
        )
        
        # 計算最大容量的品質指標
        psnr_max, ssim_max, hist_corr_max = calculate_quality_metrics_unified(origImg, final_img_max, img_info)
        
        # 創建最大容量結果
        max_capacity_result = {
            'Target_Percentage': 100.0,
            'Target_Payload': max_payload,
            'Actual_Payload': max_payload,
            'BPP': max_payload / img_info['pixel_count'],
            'PSNR': psnr_max,
            'SSIM': ssim_max,
            'Hist_Corr': hist_corr_max,
            'Processing_Time': max_run_time,
            'Measurement_Mode': f"simplified_mode"
        }
        
        print(f"Maximum payload: {max_payload} bits")
        print(f"Max BPP: {max_payload/img_info['pixel_count']:.6f}")
        
        # 清理記憶體
        cleanup_memory()
        
        # 步驟2: 生成測量點
        payload_points, measurement_mode = _generate_measurement_points(
            max_payload, step_size, segments, logger
        )
        
        # 步驟3: 簡化的批量測量（不保存圖像）
        results = _run_simplified_measurement_batch(
            origImg, payload_points, method, prediction_method, 
            ratio_of_ones, total_embeddings, el_mode, split_size, 
            block_base, quad_tree_params, use_different_weights, 
            img_info, measurement_mode, logger
        )
        
        # 添加最大容量結果並轉換為DataFrame
        all_results = results + [max_capacity_result]
        all_results.sort(key=lambda x: x['Target_Percentage'])
        df = pd.DataFrame(all_results)
        
        # 保存結果
        total_time = time.time() - total_start_time
        csv_path = f"{result_dir}/simplified_measurements.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"Simplified measurements completed.")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Results saved to {csv_path}")
        
        return df
        
    except Exception as e:
        print(f"Error in simplified measurements: {str(e)}")
        logger.log_step("Error occurred", {"Error": str(e)})
        return pd.DataFrame()  # 返回空DataFrame而不是拋出異常

def _run_simplified_measurement_batch(origImg, payload_points, method, prediction_method, 
                                    ratio_of_ones, total_embeddings, el_mode, split_size, 
                                    block_base, quad_tree_params, use_different_weights, 
                                    img_info, measurement_mode, logger):
    """簡化版的批量測量（不保存圖像）"""
    results = []
    successful_measurements = 0
    failed_measurements = 0
    
    for i, target in enumerate(tqdm(payload_points, desc=f"處理數據點")):
        percentage = target / max(payload_points) * 100 if payload_points else 0
        
        print(f"\nRunning point {i+1}/{len(payload_points)}: {target} bits ({percentage:.1f}% of max)")
        
        start_time = time.time()
        try:
            final_img, actual_payload, stages = run_embedding_with_target(
                origImg, method, prediction_method, ratio_of_ones, 
                total_embeddings, el_mode, target_payload_size=target,
                split_size=split_size, block_base=block_base, 
                quad_tree_params=quad_tree_params,
                use_different_weights=use_different_weights
            )
            
            # 驗證返回的數據
            if final_img is None or actual_payload is None:
                print(f"  Warning: Invalid return data for target {target} bits")
                failed_measurements += 1
                continue
                
        except Exception as embedding_error:
            print(f"  Error in embedding for target {target} bits: {str(embedding_error)}")
            failed_measurements += 1
            continue
        
        run_time = time.time() - start_time
        
        # 計算質量指標
        try:
            psnr, ssim, hist_corr = calculate_quality_metrics_unified(origImg, final_img, img_info)
        except Exception as metrics_error:
            print(f"  Error calculating quality metrics: {str(metrics_error)}")
            failed_measurements += 1
            continue
        
        # 記錄結果
        result = {
            'Target_Percentage': percentage,
            'Target_Payload': target,
            'Actual_Payload': actual_payload,
            'BPP': actual_payload / img_info['pixel_count'],
            'PSNR': psnr,
            'SSIM': ssim,
            'Hist_Corr': hist_corr,
            'Processing_Time': run_time,
            'Measurement_Mode': measurement_mode
        }
        results.append(result)
        successful_measurements += 1
        
        print(f"  Actual: {actual_payload} bits, PSNR: {psnr:.2f}, Time: {run_time:.2f}s")
        
        # 清理記憶體
        cleanup_memory()
    
    # 記錄批量測量統計
    logger.log_step("Simplified batch measurement completed", {
        "Successful measurements": successful_measurements,
        "Failed measurements": failed_measurements,
        "Success rate": f"{(successful_measurements / (successful_measurements + failed_measurements) * 100):.1f}%" if (successful_measurements + failed_measurements) > 0 else "0%"
    })
    
    return results

def run_embedding_with_target(origImg, method, prediction_method, ratio_of_ones, 
                                   total_embeddings, el_mode, target_payload_size,
                                   split_size=2, block_base=False, quad_tree_params=None,
                                   use_different_weights=False):
    """
    修復版的嵌入測試函數，正確處理彩色圖像的目標容量和BPP計算
    """
    from embedding import (
        pee_process_with_rotation_cuda,
        pee_process_with_split_cuda
    )
    from quadtree import pee_process_with_quadtree_cuda
    
    # 檢測圖像類型
    is_color = len(origImg.shape) == 3 and origImg.shape[2] == 3
    
    if is_color:
        print(f"Processing color image with target: {target_payload_size} bits")
        print(f"Expected capacity: ~3x equivalent grayscale image")
    else:
        print(f"Processing grayscale image with target: {target_payload_size} bits")
    
    # 重置GPU記憶體
    cp.get_default_memory_pool().free_all_blocks()
    
    # 修正權重設置
    if prediction_method in [PredictionMethod.MED, PredictionMethod.GAP, PredictionMethod.RHOMBUS]:
        actual_use_weights = False
        if use_different_weights:
            print(f"Note: Weight optimization disabled for {prediction_method.value} prediction method")
    else:
        actual_use_weights = use_different_weights
    
    try:
        if method == "rotation":
            final_img, actual_payload, stages = pee_process_with_rotation_cuda(
                origImg,
                total_embeddings,
                ratio_of_ones,
                actual_use_weights,
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
                actual_use_weights,
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
                actual_use_weights,
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
        
    except Exception as e:
        print(f"Error in embedding: method={method}, predictor={prediction_method.value}")
        print(f"Error details: {str(e)}")
        raise e

# =============================================================================
# 多預測器測量相關函數（保留原有邏輯，但更新導入）
# =============================================================================

class MultiPredictorConfig:
    """多預測器測量的配置管理"""
    
    DEFAULT_PREDICTOR_RATIOS = {
        "PROPOSED": 0.5,
        "MED": 1.0,
        "GAP": 1.0,
        "RHOMBUS": 1.0
    }
    
    PREDICTION_METHODS = [
        PredictionMethod.PROPOSED,
        PredictionMethod.MED,
        PredictionMethod.GAP,
        PredictionMethod.RHOMBUS
    ]

def _setup_multi_predictor_environment(imgName, filetype, method, quad_tree_params):
    """設置多預測器測量的環境"""
    # 讀取原始圖像
    from image_processing import read_image_auto
    from utils import find_image_path
    
    try:
        img_path = find_image_path(imgName, filetype)
        print(f"Loading image from: {img_path}")
        origImg, is_grayscale_img = read_image_auto(img_path)
        return origImg, is_grayscale_img
    except FileNotFoundError as e:
        raise ValueError(str(e))

def _create_multi_predictor_logger(comparison_dir, step_size, segments, predictor_ratios, 
                                  total_embeddings, el_mode, method, quad_tree_params):
    """創建多預測器測量的日誌文件"""
    log_file = f"{comparison_dir}/multi_predictor_precise_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Multi-predictor precise measurement run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Total embeddings: {total_embeddings}\n")
        f.write(f"EL mode: {el_mode}\n")
        
        # 記錄測量模式
        if step_size is not None and step_size > 0:
            f.write(f"Using step_size: {step_size} bits (segments parameter ignored)\n")
            f.write(f"Measurement mode: step_size\n")
        else:
            f.write(f"Using segments: {segments} (no step_size provided)\n")
            f.write(f"Measurement mode: segments\n")
            
        f.write("Predictor ratio settings:\n")
        for pred, ratio in predictor_ratios.items():
            f.write(f"  {pred}: {ratio}\n")
            
        if method == "quadtree":
            f.write(f"Quadtree params: min_block_size={quad_tree_params['min_block_size']}, variance_threshold={quad_tree_params['variance_threshold']}\n")
        f.write("\n" + "="*80 + "\n\n")
    
    return log_file

def _process_single_predictor(prediction_method, predictor_ratios, imgName, method, 
                             total_embeddings, el_mode, segments, step_size, 
                             use_different_weights, split_size, block_base, 
                             quad_tree_params, origImg, log_file):
    """處理單個預測器的測量"""
    method_name = prediction_method.value.upper()
    is_proposed = method_name.upper() == "PROPOSED"
    
    print(f"\n{'='*80}")
    print(f"Running precise measurements for {method_name.lower()} predictor")
    
    # 顯示使用的測量參數
    if step_size is not None and step_size > 0:
        print(f"Using step_size: {step_size} bits (segments parameter {segments} will be ignored)")
    else:
        print(f"Using segments: {segments} (no step_size provided)")
    print(f"{'='*80}")
    
    # 獲取當前預測器的ratio_of_ones
    current_ratio_of_ones = predictor_ratios.get(method_name, 0.5)
    print(f"Using ratio_of_ones = {current_ratio_of_ones}")
    
    # 記錄到日誌
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"Starting precise measurements for {method_name.lower()} predictor\n")
        f.write(f"Using ratio_of_ones = {current_ratio_of_ones}\n")
        if step_size is not None and step_size > 0:
            f.write(f"Using step_size = {step_size} bits\n")
        else:
            f.write(f"Using segments = {segments}\n")
        f.write("\n")
    
    try:
        predictor_start_time = time.time()
        
        if is_proposed:
            # 對於 proposed 預測器，儲存所有詳細資料
            results_df = run_precise_measurements(
                origImg, imgName, method, prediction_method, 
                current_ratio_of_ones, total_embeddings, 
                el_mode, segments, step_size,
                use_different_weights, split_size, block_base, quad_tree_params
            )
        else:
            # 對於其他預測器，僅儲存數據而不儲存圖像和圖表
            results_df = run_simplified_precise_measurements(
                origImg, imgName, method, prediction_method, 
                current_ratio_of_ones, total_embeddings, 
                el_mode, segments, step_size,
                use_different_weights, split_size, block_base, quad_tree_params
            )
        
        predictor_time = time.time() - predictor_start_time
        
        # 記錄完成信息
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Completed measurements for {method_name.lower()} predictor\n")
            f.write(f"Time taken: {predictor_time:.2f} seconds\n")
            f.write(f"Generated {len(results_df)} data points\n")
            if step_size is not None and step_size > 0:
                f.write(f"Measurement mode used: step_size={step_size}\n")
            else:
                f.write(f"Measurement mode used: segments={segments}\n")
            f.write("\n")
        
        # 清理記憶體
        cleanup_memory()
        
        return results_df, predictor_time
        
    except Exception as e:
        print(f"Error processing {method_name.lower()}: {str(e)}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Error processing {method_name.lower()}: {str(e)}\n")
            f.write(traceback.format_exc())
            f.write("\n\n")
        return None, 0

def _finalize_multi_predictor_results(all_results, comparison_dir, imgName, method, 
                                     total_start_time, step_size, segments, log_file):
    """完成多預測器結果的處理和保存"""
    if not all_results:
        print("No valid results to process")
        return all_results
    
    try:
        # 導入繪圖函數（如果需要的話，可以移到 visualization.py）
        # plot_predictor_comparison(all_results, imgName, method, comparison_dir)
        
        # 記錄運行時間
        total_time = time.time() - total_start_time
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\nComparison completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total processing time: {total_time:.2f} seconds\n")
            if step_size is not None and step_size > 0:
                f.write(f"Final confirmation: Used step_size={step_size} for all predictors\n")
            else:
                f.write(f"Final confirmation: Used segments={segments} for all predictors\n")
            f.write("\n")
        
        print(f"\nComparison completed and saved to {comparison_dir}")
        print(f"Total processing time: {total_time:.2f} seconds")
        
        # 確認使用的測量模式
        if step_size is not None and step_size > 0:
            print(f"Confirmed: All predictors used step_size={step_size} bits")
        else:
            print(f"Confirmed: All predictors used segments={segments}")
        
        # 創建寬格式表格（如果需要的話）
        # create_wide_format_tables(all_results, comparison_dir)
        
        return all_results
        
    except Exception as e:
        print(f"Error generating comparison: {str(e)}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\nError generating comparison: {str(e)}\n")
            f.write(traceback.format_exc())
        return all_results

# =============================================================================
# 重構後的主函數
# =============================================================================

def run_multi_predictor_precise_measurements(imgName, filetype="png", method="quadtree", 
                                           predictor_ratios=None, total_embeddings=5, 
                                           el_mode=0, segments=15, step_size=None, use_different_weights=False,
                                           split_size=2, block_base=False, quad_tree_params=None):
    """
    重構版的多預測器精確測量函數
    """
    
    # 設置默認的預測器ratio字典
    if predictor_ratios is None:
        predictor_ratios = MultiPredictorConfig.DEFAULT_PREDICTOR_RATIOS
    
    # 記錄總運行開始時間
    total_start_time = time.time()
    
    # 步驟1: 設置環境
    print("Step 1: Setting up multi-predictor measurement environment")
    try:
        origImg, is_grayscale_img = _setup_multi_predictor_environment(
            imgName, filetype, method, quad_tree_params
        )
    except ValueError as e:
        print(f"Failed to setup environment: {e}")
        return None
    
    # 步驟2: 創建比較結果目錄和日誌
    print("Step 2: Creating output directories and logging setup")
    from utils import get_precise_measurement_directories
    
    comparison_dir = f"{PathConstants.BASE_OUTPUT_DIR}/plots/{imgName}/precise_comparison"
    ensure_dir(f"{comparison_dir}/dummy.txt")
    
    log_file = _create_multi_predictor_logger(
        comparison_dir, step_size, segments, predictor_ratios, 
        total_embeddings, el_mode, method, quad_tree_params
    )
    
    # 步驟3: 依次運行每種預測方法
    print("Step 3: Running measurements for each predictor")
    all_results = {}
    
    for prediction_method in tqdm(MultiPredictorConfig.PREDICTION_METHODS, desc="處理預測器"):
        method_name = prediction_method.value.upper()
        
        # 處理單個預測器
        results_df, predictor_time = _process_single_predictor(
            prediction_method, predictor_ratios, imgName, method, 
            total_embeddings, el_mode, segments, step_size, 
            use_different_weights, split_size, block_base, 
            quad_tree_params, origImg, log_file
        )
        
        if results_df is not None and len(results_df) > 0:
            # 保存結果
            all_results[method_name.lower()] = results_df
            
            # 保存CSV到比較目錄
            results_df.to_csv(f"{comparison_dir}/{method_name.lower()}_precise.csv", index=False)
            print(f"Saved results for {method_name.lower()} predictor")
        else:
            print(f"Warning: No valid results for {method_name.lower()} predictor")
    
    # 步驟4: 生成比較圖表和最終處理
    print("Step 4: Finalizing results and generating comparison")
    final_results = _finalize_multi_predictor_results(
        all_results, comparison_dir, imgName, method, 
        total_start_time, step_size, segments, log_file
    )
    
    return final_results

# =============================================================================
# 方法比較的輔助函數
# =============================================================================

class MethodComparisonConfig:
    """方法比較的配置管理"""
    
    DEFAULT_METHODS = ["rotation", "split", "quadtree"]
    
    DEFAULT_METHOD_PARAMS = {
        "rotation": {"split_size": 2, "use_different_weights": False},
        "split": {"split_size": 2, "block_base": False, "use_different_weights": False},
        "quadtree": {"min_block_size": 16, "variance_threshold": 300, "use_different_weights": False}
    }
    
    PREDICTOR_MAP = {
        "proposed": PredictionMethod.PROPOSED,
        "med": PredictionMethod.MED,
        "gap": PredictionMethod.GAP,
        "rhombus": PredictionMethod.RHOMBUS
    }

def _validate_method_comparison_params(methods, method_params, predictor):
    """驗證和設置方法比較的參數"""
    # 如果未提供方法，使用默認方法
    if methods is None:
        methods = MethodComparisonConfig.DEFAULT_METHODS
    
    # 如果未提供方法參數，使用默認參數
    if method_params is None:
        method_params = MethodComparisonConfig.DEFAULT_METHOD_PARAMS.copy()
    
    # 驗證預測器
    prediction_method = MethodComparisonConfig.PREDICTOR_MAP.get(predictor.lower())
    if prediction_method is None:
        prediction_method = PredictionMethod.PROPOSED
        print(f"Warning: Unknown predictor '{predictor}', using PROPOSED instead")
    
    return methods, method_params, prediction_method

def _setup_method_comparison_environment(imgName, filetype, predictor):
    """設置方法比較的環境"""
    import cv2
    
    # 讀取原始圖像
    origImg = cv2.imread(f"{PathConstants.BASE_IMAGE_DIR}/{imgName}.{filetype}", cv2.IMREAD_GRAYSCALE)
    if origImg is None:
        raise ValueError(f"Failed to read image: {PathConstants.BASE_IMAGE_DIR}/{imgName}.{filetype}")
    
    # 創建比較輸出目錄
    comparison_dir = f"{PathConstants.BASE_OUTPUT_DIR}/plots/{imgName}/method_comparison_{predictor}"
    ensure_dir(f"{comparison_dir}/dummy.txt")
    
    return origImg, comparison_dir

def _create_method_comparison_logger(comparison_dir, imgName, filetype, predictor, 
                                   methods, total_embeddings, el_mode, step_size, segments):
    """創建方法比較的日誌文件"""
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
    
    return log_file

def _prepare_method_params(method_name, method_params):
    """準備單個方法的參數"""
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
        params["quad_tree_params"] = quad_tree_params
    
    return params

def _process_single_method(method_name, origImg, imgName, prediction_method, ratio_of_ones,
                          total_embeddings, el_mode, segments, step_size, 
                          method_params, comparison_dir, log_file):
    """處理單個方法的測量"""
    print(f"\n{'='*80}")
    print(f"Running precise measurements for {method_name} method")
    print(f"{'='*80}")
    
    try:
        # 準備方法參數
        params = _prepare_method_params(method_name, method_params)
        
        # 運行精確測量
        results_df = run_precise_measurements(
            origImg, imgName, method_name, prediction_method, ratio_of_ones,
            total_embeddings, el_mode, segments, step_size,
            **params  # 展開方法特定參數
        )
        
        # 保存為CSV
        csv_filename = f"{method_name}_{prediction_method.value}_precise.csv"
        results_df.to_csv(f"{comparison_dir}/{csv_filename}", index=False)
        
        # 記錄完成
        with open(log_file, 'a') as f:
            f.write(f"Completed measurements for {method_name} method\n")
            f.write(f"Results saved to {comparison_dir}/{csv_filename}\n\n")
        
        print(f"Completed measurements for {method_name} method")
        return results_df
        
    except Exception as e:
        print(f"Error processing {method_name}: {str(e)}")
        with open(log_file, 'a') as f:
            f.write(f"Error processing {method_name}: {str(e)}\n")
            f.write(traceback.format_exc())
            f.write("\n\n")
        
        # 確保清理記憶體
        cleanup_memory()
        return None

def _finalize_method_comparison_results(all_results, imgName, predictor, comparison_dir):
    """完成方法比較結果的處理"""
    if not all_results:
        print("No valid results for comparison")
        return all_results
    
    try:
        # 創建比較圖表（可以移到 visualization.py）
        # plot_method_comparison(all_results, imgName, predictor, comparison_dir)
        
        # 創建比較表格（可以移到 visualization.py）
        # create_comparative_table(all_results, f"{comparison_dir}/method_comparison_table.csv")
        
        print(f"Method comparison completed.")
        print(f"Results saved to {comparison_dir}")
        
        return all_results
        
    except Exception as e:
        print(f"Error finalizing method comparison: {str(e)}")
        return all_results

# =============================================================================
# 重構後的主函數
# =============================================================================

def run_method_comparison(imgName, filetype="png", predictor="proposed", 
                        ratio_of_ones=0.5, methods=None, method_params=None,
                        total_embeddings=5, el_mode=0, segments=15, step_size=None):
    """
    重構版的方法比較函數
    
    主要改進：
    1. 拆分為多個小函數，職責明確
    2. 統一的參數驗證和錯誤處理
    3. 重用已重構的 run_precise_measurements
    4. 更清晰的步驟劃分
    
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
        每個方法的特定參數
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
    
    print(f"\n{'='*80}")
    print(f"Starting method comparison with {predictor} predictor")
    print(f"{'='*80}")
    
    # 步驟1: 驗證和設置參數
    print("Step 1: Validating parameters and setup")
    methods, method_params, prediction_method = _validate_method_comparison_params(
        methods, method_params, predictor
    )
    
    print(f"Methods to compare: {methods}")
    print(f"Using predictor: {prediction_method.value}")
    
    # 步驟2: 設置環境
    print("Step 2: Setting up comparison environment")
    try:
        origImg, comparison_dir = _setup_method_comparison_environment(
            imgName, filetype, predictor
        )
    except ValueError as e:
        print(f"Failed to setup environment: {e}")
        return None
    
    # 步驟3: 創建日誌
    print("Step 3: Creating comparison log")
    log_file = _create_method_comparison_logger(
        comparison_dir, imgName, filetype, predictor, 
        methods, total_embeddings, el_mode, step_size, segments
    )
    
    # 步驟4: 處理每個方法
    print("Step 4: Running measurements for each method")
    all_results = {}
    
    for method_name in methods:
        results_df = _process_single_method(
            method_name, origImg, imgName, prediction_method, ratio_of_ones,
            total_embeddings, el_mode, segments, step_size, 
            method_params, comparison_dir, log_file
        )
        
        if results_df is not None:
            all_results[method_name] = results_df
        else:
            print(f"Warning: No valid results for {method_name} method")
    
    # 步驟5: 完成比較結果處理
    print("Step 5: Finalizing comparison results")
    final_results = _finalize_method_comparison_results(
        all_results, imgName, predictor, comparison_dir
    )
    
    return final_results


# =============================================================================
# 數據一致性檢查函數（重構版）
# =============================================================================

def ensure_bpp_psnr_consistency(results_df, is_color=False):
    """
    重構版的BPP-PSNR一致性檢查函數
    
    主要改進：
    1. 使用常數而非魔法數字
    2. 更清晰的邏輯結構
    3. 統一的處理策略
    """
    df = results_df.copy().sort_values('BPP')
    
    # 根據圖像類型調整檢查參數
    if is_color:
        tolerance_factor = MeasurementConstants.COLOR_CORRECTION_STRENGTH * 3
        print("Applying color image specific consistency checks...")
    else:
        tolerance_factor = MeasurementConstants.DEFAULT_CORRECTION_STRENGTH * 2
    
    # 確保 PSNR 隨著 BPP 增加而單調下降
    df = _ensure_metric_monotonicity(df, 'PSNR', tolerance_factor, is_color)
    
    # 對其他指標進行類似處理
    for metric in ['SSIM', 'Hist_Corr']:
        df = _ensure_metric_monotonicity(df, metric, tolerance_factor, is_color)
    
    return df.sort_values('Target_Percentage')

def _ensure_metric_monotonicity(df, metric, tolerance_factor, is_color):
    """確保指標的單調性"""
    for i in range(1, len(df)):
        if df.iloc[i][metric] > df.iloc[i-1][metric]:
            if i > 1:
                # 計算預期的變化
                prev_slope = (df.iloc[i-1][metric] - df.iloc[i-2][metric]) / \
                           (df.iloc[i-1]['BPP'] - df.iloc[i-2]['BPP'])
                expected_drop = prev_slope * (df.iloc[i]['BPP'] - df.iloc[i-1]['BPP'])
                
                # 應用修正
                decay_factor = 0.998 if is_color else 0.995
                corrected_value = max(
                    df.iloc[i-1][metric] + expected_drop, 
                    df.iloc[i-1][metric] * decay_factor
                )
                df.loc[df.index[i], metric] = corrected_value
            else:
                # 對於前面的點，使用百分比降低
                decay_factor = 0.998 if is_color else 0.995
                df.loc[df.index[i], metric] = df.iloc[i-1][metric] * decay_factor
    
    return df

# =============================================================================
# 統計函數（重構版）
# =============================================================================

def generate_interval_statistics(original_img, stages, total_payload, segments=15):
    """
    重構版的統計數據生成函數
    
    主要改進：
    1. 使用統一的圖像信息獲取
    2. 重用質量指標計算函數
    3. 更清晰的錯誤處理
    """
    # 確保輸入數據類型正確
    if isinstance(original_img, cp.ndarray):
        original_img = cp.asnumpy(original_img)
    
    # 獲取圖像信息
    img_info = get_image_info(original_img)
    print(f"Generating interval statistics for {img_info['description']}")
    
    # 驗證輸入
    if total_payload <= 0:
        print("Warning: Total payload is zero or negative. No statistics generated.")
        return None, None
    
    # 設置分段參數
    segments = max(2, min(segments, total_payload))
    
    # 生成統計數據
    results = _generate_payload_intervals(
        original_img, stages, total_payload, segments, img_info
    )
    
    if not results:
        return None, None
    
    # 創建DataFrame和表格
    df = pd.DataFrame(results)
    table = _create_statistics_table(results)
    
    return df, table

def _generate_payload_intervals(original_img, stages, total_payload, segments, img_info):
    """生成載荷間隔的統計數據"""
    # 計算每個級距的目標嵌入量
    payload_interval = total_payload / segments
    payload_points = [int(i * payload_interval) for i in range(1, segments + 1)]
    payload_points[-1] = total_payload  # 確保最後一個點是總嵌入量
    
    results = []
    accumulated_payload = 0
    current_stage_index = 0
    current_stage_img = None
    
    for target_payload in payload_points:
        # 模擬嵌入到目標嵌入量的圖像狀態
        current_stage_img = _find_stage_for_payload(
            stages, target_payload, accumulated_payload, current_stage_index
        )
        
        if current_stage_img is None:
            print("Warning: No valid stage image found.")
            continue
        
        # 計算性能指標
        psnr, ssim, hist_corr = calculate_quality_metrics_unified(
            original_img, current_stage_img, img_info
        )
        
        # 計算BPP
        bpp = target_payload / img_info['pixel_count']
        
        # 添加到結果列表
        results.append({
            'Payload': target_payload,
            'BPP': bpp,
            'PSNR': psnr,
            'SSIM': ssim,
            'Hist_Corr': hist_corr
        })
    
    return results

def _find_stage_for_payload(stages, target_payload, accumulated_payload, current_stage_index):
    """為給定的payload找到對應的stage圖像"""
    temp_accumulated = accumulated_payload
    temp_index = current_stage_index
    current_stage_img = None
    
    while temp_accumulated < target_payload and temp_index < len(stages):
        current_stage = stages[temp_index]
        stage_payload = current_stage['payload']
        current_stage_img = current_stage['stage_img']
        
        # 確保是numpy格式
        if isinstance(current_stage_img, cp.ndarray):
            current_stage_img = cp.asnumpy(current_stage_img)
        
        if temp_accumulated + stage_payload <= target_payload:
            # 完整包含當前階段
            temp_accumulated += stage_payload
            temp_index += 1
        else:
            # 部分包含當前階段
            break
    
    # 確保有有效的圖像
    if current_stage_img is None and temp_index > 0:
        prev_stage = stages[temp_index - 1]
        current_stage_img = prev_stage['stage_img']
        if isinstance(current_stage_img, cp.ndarray):
            current_stage_img = cp.asnumpy(current_stage_img)
    
    return current_stage_img

def _create_statistics_table(results):
    """創建統計數據的PrettyTable"""
    table = pt.PrettyTable()
    table.field_names = ["Payload", "BPP", "PSNR", "SSIM", "Hist_Corr"]
    
    for result in results:
        table.add_row([
            result['Payload'],
            f"{result['BPP']:.6f}",
            f"{result['PSNR']:.2f}",
            f"{result['SSIM']:.4f}",
            f"{result['Hist_Corr']:.4f}"
        ])
    
    return table

# =============================================================================
# 簡化的寬格式表格函數
# =============================================================================

def create_wide_format_tables(all_results, output_dir):
    """
    重構版的寬格式表格創建函數
    
    主要改進：
    1. 更清晰的邏輯結構
    2. 統一的數據處理
    3. 更好的錯誤處理
    """
    if not all_results:
        print("No results to create wide format tables")
        return
    
    try:
        # 創建各種指標的表格
        tables = _create_metric_tables(all_results)
        
        # 保存表格
        _save_metric_tables(tables, output_dir)
        
        # 創建LaTeX格式表格
        _save_latex_tables(tables, output_dir)
        
        print(f"Wide format tables saved to {output_dir}")
        
    except Exception as e:
        print(f"Error creating wide format tables: {str(e)}")

def _create_metric_tables(all_results):
    """創建各種指標的表格"""
    # 確定所有百分比值
    percentages = []
    for df in all_results.values():
        percentages.extend(df['Target_Percentage'].tolist())
    
    percentages = sorted(list(set(percentages)))
    
    # 初始化表格
    tables = {
        'psnr': {'Percentage': percentages},
        'ssim': {'Percentage': percentages},
        'hist_corr': {'Percentage': percentages}
    }
    
    # 填充各預測器的數據
    for predictor, df in all_results.items():
        for metric in ['psnr', 'ssim', 'hist_corr']:
            metric_col = metric.upper() if metric != 'hist_corr' else 'Hist_Corr'
            values = []
            
            for percentage in percentages:
                # 找到最接近的百分比行
                closest_idx = (df['Target_Percentage'] - percentage).abs().idxmin()
                values.append(df.loc[closest_idx, metric_col])
            
            tables[metric][predictor] = values
    
    return tables

def _save_metric_tables(tables, output_dir):
    """保存指標表格為CSV"""
    for metric, table_data in tables.items():
        df = pd.DataFrame(table_data)
        csv_path = f"{output_dir}/wide_format_{metric}.csv"
        df.to_csv(csv_path, index=False)

def _save_latex_tables(tables, output_dir):
    """保存LaTeX格式表格"""
    format_specs = {
        'psnr': "%.2f",
        'ssim': "%.4f", 
        'hist_corr': "%.4f"
    }
    
    for metric, table_data in tables.items():
        df = pd.DataFrame(table_data)
        latex_path = f"{output_dir}/latex_table_{metric}.txt"
        
        with open(latex_path, 'w') as f:
            latex_content = df.to_latex(
                index=False, 
                float_format=format_specs[metric]
            )
            f.write(latex_content)