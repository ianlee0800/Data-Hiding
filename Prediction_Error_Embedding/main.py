import warnings
from numba.core.errors import NumbaPerformanceWarning
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import cupy as cp
import numpy as np
import cv2
import time
from image_processing import (
    save_image,
    generate_histogram,
    save_histogram,
    add_grid_lines,
    PredictionMethod,
    plot_interval_statistics,
)
from embedding import (
    pee_process_with_rotation_cuda,
    pee_process_with_split_cuda
)
from utils import (
    create_pee_info_table,
    generate_interval_statistics,
    save_interval_statistics,
    run_multiple_predictors,
    run_precise_measurements,
    run_multi_predictor_precise_measurements,
    run_simplified_precise_measurements
)
# 導入視覺化模組
from visualization import (
    visualize_split, visualize_quadtree, save_comparison_image,
    create_block_size_distribution_chart, visualize_rotation_angles,
    create_metrics_comparison_chart, visualize_embedding_heatmap,
    create_payload_distribution_chart, create_el_distribution_chart,
    create_histogram_animation, visualize_color_histograms, create_color_heatmap,
    visualize_color_metrics_comparison, create_color_channel_comparison
)

from common import calculate_psnr, calculate_ssim, histogram_correlation, cleanup_memory
from quadtree import pee_process_with_quadtree_cuda

# Import the color detection functions
from color import read_image_auto, calculate_color_metrics

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    """
    主函數，負責整個數據隱藏過程的控制和執行
    
    更新功能:
    1. 支持最大嵌入量，並將嵌入數據分成15段進行統計
    2. 新增進度條顯示
    3. 改進記憶體管理
    4. 只為 proposed 預測器儲存詳細資訊
    5. 修復圖表資源管理和 DataFrame 警告
    6. 調整圖像儲存路徑至 "./Prediction_Error_Embedding/outcome/image"
    7. 按照方法類型儲存更多詳細的實驗圖像
    8. 新增彩色圖像處理支持，自動偵測圖像類型
    """
    # ==== 參數設置（直接在代碼中調整） ====
    
    # 基本參數設置
    imgName = "Male"         # 圖像名稱
    filetype = "tiff"           # 圖像檔案類型
    total_embeddings = 5        # 總嵌入次數
    
    # 各預測器的ratio_of_ones設置
    predictor_ratios = {
        "PROPOSED": 0.3,        # proposed預測器的ratio_of_ones
        "MED": 1.0,             # MED預測器的ratio_of_ones
        "GAP": 0.7,             # GAP預測器的ratio_of_ones
        "RHOMBUS": 0.9          # RHOMBUS預測器的ratio_of_ones
    }
    
    el_mode = 0                 # 0: 無限制, 1: 漸增, 2: 漸減
    use_different_weights = False 
    
    # 測量方式
    use_precise_measurement = False  # True: 使用精確測量模式, False: 使用近似模式
    
    # 統計分段數量
    stats_segments = 20
    
    # 預測方法選擇
    # 可選：PROPOSED, MED, GAP, RHOMBUS, ALL (ALL表示運行所有方法並生成比較)
    prediction_method_str = "PROPOSED"
    
    # 方法選擇
    method = "quadtree"         # 可選："rotation", "split", "quadtree"
    
    # 各方法共用參數
    split_size = 2              # 用於 rotation 和 split 方法
    block_base = False          # 用於 split 方法
    
    # quad tree 特定參數
    quad_tree_params = {
        'min_block_size': 16,   # 支援到16x16
        'variance_threshold': 300
    }
    
    # ==== 主程序開始 ====
    
    # 處理預測方法選擇
    prediction_method_map = {
        "PROPOSED": PredictionMethod.PROPOSED,
        "MED": PredictionMethod.MED,
        "GAP": PredictionMethod.GAP,
        "RHOMBUS": PredictionMethod.RHOMBUS
    }
    
    # 如果選擇 ALL，則執行所有預測方法並生成比較
    if prediction_method_str.upper() == "ALL":
        print("Running all prediction methods and generating comparison...")
        
        if use_precise_measurement:
            # 使用精確測量模式，改進版：只為 proposed 儲存詳細資訊
            print("Using precise measurement mode with separate runs per predictor...")
            
            run_multi_predictor_precise_measurements(
                imgName=imgName,
                filetype=filetype,
                method=method,
                predictor_ratios=predictor_ratios,
                total_embeddings=total_embeddings,
                el_mode=el_mode,
                segments=stats_segments,
                use_different_weights=use_different_weights,
                split_size=split_size,
                block_base=block_base,
                quad_tree_params=quad_tree_params
            )
        else:
            # 使用近似模式
            print("Using approximate measurement mode based on stages...")
            
            run_multiple_predictors(
                imgName=imgName,
                filetype=filetype,
                method=method,
                predictor_ratios=predictor_ratios,
                total_embeddings=total_embeddings,
                el_mode=el_mode,
                use_different_weights=use_different_weights,
                split_size=split_size,
                block_base=block_base,
                quad_tree_params=quad_tree_params,
                stats_segments=stats_segments
            )
        
        return
    
    # 否則，執行單一預測方法
    prediction_method = prediction_method_map.get(prediction_method_str.upper())
    if prediction_method is None:
        print(f"Error: Unknown prediction method: {prediction_method_str}")
        print(f"Available options: PROPOSED, MED, GAP, RHOMBUS, ALL")
        return

    # 獲取當前預測器的ratio_of_ones
    ratio_of_ones = predictor_ratios.get(prediction_method_str.upper(), 0.5)
    print(f"Using ratio_of_ones = {ratio_of_ones} for {prediction_method_str} predictor")

    # 創建必要的目錄
    # 更新圖像儲存路徑
    base_dir = "./Prediction_Error_Embedding/outcome"
    image_dir = f"{base_dir}/image/{imgName}/{method}"
    histogram_dir = f"{base_dir}/histogram/{imgName}/{method}"
    plots_dir = f"{base_dir}/plots/{imgName}/{method}"
    
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(histogram_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 特定方法的子目錄
    if method == "rotation":
        os.makedirs(f"{image_dir}/rotated", exist_ok=True)
        os.makedirs(f"{image_dir}/subimages", exist_ok=True)
    elif method == "split":
        os.makedirs(f"{image_dir}/split_visualization", exist_ok=True)
        os.makedirs(f"{image_dir}/subimages", exist_ok=True)
    elif method == "quadtree":
        os.makedirs(f"{image_dir}/quadtree_visualization", exist_ok=True)
        os.makedirs(f"{image_dir}/with_grid", exist_ok=True)
        os.makedirs(f"{image_dir}/rotated_blocks", exist_ok=True)  # New directory for rotated block images
        os.makedirs(f"{plots_dir}/block_distribution", exist_ok=True)
        # 彩色圖像特有目錄
        os.makedirs(f"{image_dir}/channels", exist_ok=True)
        os.makedirs(f"{histogram_dir}/channels", exist_ok=True)
    
    # 確保結果數據目錄存在
    os.makedirs(f"{base_dir}/data/{imgName}", exist_ok=True)
    
    try:
        # 清理 GPU 記憶體
        cleanup_memory()

        # 讀取圖像 - 使用新的自動檢測功能
        img_path = f"./Prediction_Error_Embedding/image/{imgName}.{filetype}"
        if not os.path.exists(img_path):
            img_path = f"./pred_and_QR/image/{imgName}.{filetype}"
            if not os.path.exists(img_path):
                raise ValueError(f"Failed to find image: {imgName}.{filetype}")
        
        print(f"Loading image from: {img_path}")
        # 使用新的函數自動檢測圖像類型
        origImg, is_grayscale_img = read_image_auto(img_path)
        
        # 根據圖像類型儲存原始圖像
        if is_grayscale_img:
            print(f"Processing grayscale image: {imgName}.{filetype}")
            save_image(origImg, f"{image_dir}/original.png")
            # 儲存原始圖像直方圖
            save_histogram(origImg, 
                         f"{histogram_dir}/original_histogram.png", 
                         "Original Image Histogram")
        else:
            print(f"Processing color image: {imgName}.{filetype}")
            cv2.imwrite(f"{image_dir}/original.png", origImg)
            
            # 為彩色圖像創建三通道直方圖
            visualize_color_histograms(origImg, 
                                     f"{histogram_dir}/original_color_histogram.png", 
                                     f"Original Color Image Histogram - {imgName}")
        
        # 如果使用精確測量模式，執行精確測量後返回
        if use_precise_measurement:
            print(f"\nUsing precise measurement mode with {stats_segments} separate runs...")
            
            # 判斷是否為 proposed 預測器
            is_proposed = prediction_method_str.upper() == "PROPOSED"
            
            if is_proposed:
                # 為 proposed 執行完整測量，包括圖像和圖表
                run_precise_measurements(
                    origImg, imgName, method, prediction_method, ratio_of_ones, 
                    total_embeddings, el_mode, stats_segments, use_different_weights,
                    split_size, block_base, quad_tree_params
                )
            else:
                # 為其他預測器執行簡化測量，僅儲存數據
                run_simplified_precise_measurements(
                    origImg, imgName, method, prediction_method, ratio_of_ones, 
                    total_embeddings, el_mode, stats_segments, use_different_weights,
                    split_size, block_base, quad_tree_params
                )
            
            return

        print(f"Starting encoding process... ({'CUDA' if cp.cuda.is_available() else 'CPU'} mode)")
        print(f"Using method: {method}")
        print(f"Prediction method: {prediction_method.value}")
        print(f"Using approximate measurement mode based on stages...")
        
        try:
            # 如果使用 MED、GAP 或 RHOMBUS 方法，強制設置 use_different_weights 為 False
            if prediction_method in [PredictionMethod.MED, PredictionMethod.GAP, PredictionMethod.RHOMBUS]:
                use_different_weights = False
                print(f"Note: Weight optimization disabled for {prediction_method.value} prediction method")
                
            # 判斷是否為 proposed 預測器
            is_proposed = prediction_method == PredictionMethod.PROPOSED
                
            # 執行選定的方法 - 這部分代碼已經設計為同時支持灰階和彩色圖像
            if method == "rotation":
                final_pee_img, total_payload, pee_stages = pee_process_with_rotation_cuda(
                    origImg,
                    total_embeddings,
                    ratio_of_ones,
                    use_different_weights,
                    split_size,
                    el_mode,
                    prediction_method=prediction_method,
                    target_payload_size=-1  # 使用最大嵌入量
                )
            elif method == "split":
                final_pee_img, total_payload, pee_stages = pee_process_with_split_cuda(
                    origImg,
                    total_embeddings,
                    ratio_of_ones,
                    use_different_weights,
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
                    ratio_of_ones,
                    use_different_weights,
                    quad_tree_params['min_block_size'],
                    quad_tree_params['variance_threshold'],
                    el_mode,
                    rotation_mode='random',
                    prediction_method=prediction_method,
                    target_payload_size=-1,
                    imgName=imgName,  # Pass the image name
                    output_dir="./Prediction_Error_Embedding/outcome"  # Pass the output directory
                )

            # 建立並列印 PEE 資訊表格
            total_pixels = origImg.size
            pee_table = create_pee_info_table(pee_stages, use_different_weights, total_pixels, 
                                           split_size, method == "quadtree")
            print(pee_table)

            # 生成並保存統計數據
            print("\nGenerating interval statistics...")
            stats_df, stats_table = generate_interval_statistics(
                origImg, pee_stages, total_payload, segments=stats_segments
            )
            
            if stats_df is not None:
                print("\nInterval Statistics:")
                print(stats_table)
                
                # 保存統計數據
                save_interval_statistics(
                    stats_df, imgName, method, prediction_method.value, 
                    base_dir=f"{base_dir}/plots"  # 更新儲存路徑
                )
                
                # 繪製統計圖表
                if is_proposed:
                    plot_interval_statistics(
                        stats_df, imgName, method, prediction_method.value,
                        output_dir=plots_dir  # 更新儲存路徑
                    )

            # 創建有效載荷分布圖表
            if is_proposed:
                create_payload_distribution_chart(
                    pee_stages, f"{plots_dir}/payload_distribution.png"
                )
                
                # 創建EL分布圖表
                create_el_distribution_chart(
                    pee_stages, f"{plots_dir}/el_distribution.png"
                )
                
                # 創建指標比較圖表
                metrics = {
                    'psnr': [stage['psnr'] for stage in pee_stages],
                    'ssim': [stage['ssim'] for stage in pee_stages],
                    'hist_corr': [stage['hist_corr'] for stage in pee_stages],
                    'bpp': [stage['bpp'] for stage in pee_stages]
                }
                stages = [stage['embedding'] for stage in pee_stages]
                create_metrics_comparison_chart(
                    stages, metrics, f"{plots_dir}/metrics_comparison.png",
                    f"Metrics Comparison Across Stages for {imgName}"
                )
                
                # 為彩色圖像創建通道比較圖表
                if not is_grayscale_img and 'channel_metrics' in pee_stages[0]:
                    visualize_color_metrics_comparison(
                        pee_stages, f"{plots_dir}/channel_metrics_comparison.png",
                        f"Channel Metrics Comparison for {imgName}"
                    )
                
                if is_grayscale_img:
                    # 創建直方圖動畫 (僅適用於灰階圖像)
                    create_histogram_animation(
                        pee_stages, origImg, plots_dir, imgName, method
                    )

            # 儲存每個階段的圖像和相關資訊
            for i, stage in enumerate(pee_stages):
                # 處理圖像格式 (針對彩色或灰階)
                if is_grayscale_img:
                    # 確保數據類型一致 (灰階)
                    stage_img = cp.asnumpy(stage['stage_img']) if isinstance(stage['stage_img'], cp.ndarray) else stage['stage_img']
                    
                    # 共通項目：儲存階段結果圖像
                    stage_img_path = f"{image_dir}/stage_{i}_result.png"
                    save_image(stage_img, stage_img_path)
                else:
                    # 彩色圖像處理
                    if isinstance(stage['stage_img'], cp.ndarray):
                        stage_img = cp.asnumpy(stage['stage_img'])
                    else:
                        stage_img = stage['stage_img']
                    
                    # 儲存階段結果圖像
                    stage_img_path = f"{image_dir}/stage_{i}_result.png"
                    cv2.imwrite(stage_img_path, stage_img)
                
                # 僅對 proposed 預測器儲存更多詳細資訊
                if is_proposed:
                    if is_grayscale_img:
                        # 灰階圖像的直方圖處理
                        hist_path = f"{histogram_dir}/stage_{i}_histogram.png"
                        plt.figure(figsize=(10, 6))
                        plt.bar(range(256), generate_histogram(stage_img), alpha=0.7)
                        plt.title(f"Histogram after PEE Stage {i}")
                        plt.xlabel("Pixel Value")
                        plt.ylabel("Frequency")
                        plt.savefig(hist_path)
                        plt.close()
                    else:
                        # 彩色圖像的直方圖處理
                        visualize_color_histograms(
                            stage_img, 
                            f"{histogram_dir}/stage_{i}_color_histogram.png",
                            f"Color Histogram after PEE Stage {i}"
                        )
                    
                    # 創建嵌入熱圖 (對彩色和灰階都適用，但使用不同的函數)
                    if is_grayscale_img:
                        heatmap_path = f"{image_dir}/stage_{i}_heatmap.png"
                        visualize_embedding_heatmap(origImg, stage_img, heatmap_path)
                    else:
                        heatmap_path = f"{image_dir}/stage_{i}_color_heatmap.png"
                        create_color_heatmap(origImg, stage_img, heatmap_path)
                        
                        # 創建通道對比圖
                        channel_path = f"{image_dir}/stage_{i}_channel_comparison.png"
                        create_color_channel_comparison(origImg, stage_img, channel_path)
                    
                    # 根據方法類型處理特定圖像儲存
                    if method == "rotation" and is_grayscale_img:
                        # Rotation 方法特有項目 (僅適用於灰階)
                        if 'rotated_stage_img' in stage:
                            rotated_img = cp.asnumpy(stage['rotated_stage_img'])
                            # 儲存旋轉後的圖像
                            rotation_angle = stage.get('rotation', i * 90)
                            rot_path = f"{image_dir}/rotated/stage_{i}_rotated_{rotation_angle}.png"
                            save_image(rotated_img, rot_path)
                            
                        # 儲存子圖像（可選，選擇性儲存幾個具代表性的）
                        if 'sub_images' in stage and i == 0:  # 僅存儲第一階段的子圖像示例
                            for j, sub_img_info in enumerate(stage['sub_images'][:4]):  # 只存儲前4個
                                if 'sub_img' in sub_img_info:
                                    sub_img = cp.asnumpy(sub_img_info['sub_img'])
                                    sub_path = f"{image_dir}/subimages/stage_{i}_subimage_{j}.png"
                                    save_image(sub_img, sub_path)
                                    
                    elif method == "split" and is_grayscale_img:
                        # Split 方法特有項目 (僅適用於灰階)
                        # 創建分割示意圖
                        if i == 0:  # 僅第一階段需要
                            split_viz = visualize_split(origImg, split_size, block_base)
                            viz_path = f"{image_dir}/split_visualization/split_visualization.png"
                            save_image(split_viz, viz_path)
                            
                        # 儲存旋轉前後對比（僅對一些具代表性的子圖像）
                        if 'sub_images' in stage and i == 0:  # 僅存儲第一階段的子圖像示例
                            for j, sub_img_info in enumerate(stage['sub_images'][:4]):  # 只存儲前4個
                                if 'original_sub_img' in sub_img_info and 'embedded_sub_img' in sub_img_info:
                                    orig_sub = cp.asnumpy(sub_img_info['original_sub_img'])
                                    emb_sub = cp.asnumpy(sub_img_info['embedded_sub_img'])
                                    comp_path = f"{image_dir}/subimages/stage_{i}_subimage_{j}_comparison.png"
                                    save_comparison_image(orig_sub, emb_sub, comp_path, 
                                                        labels=("Original", "Embedded"))
                                    
                    elif method == "quadtree":
                        # Quadtree 方法特有項目 (彩色和灰階都可)
                        # 創建帶格線的結果圖像
                        if 'block_info' in stage:
                            if is_grayscale_img:
                                # 灰階圖像的網格處理
                                grid_image = add_grid_lines(stage_img.copy(), stage['block_info'])
                                grid_path = f"{image_dir}/with_grid/stage_{i}_result_with_grid.png"
                                save_image(grid_image, grid_path)
                            else:
                                # 彩色圖像需要為每個通道單獨處理網格
                                # 為藍色通道單獨繪製網格
                                if 'channel_metrics' in stage and 'block_info' in stage:
                                    b, g, r = cv2.split(stage_img)
                                    if 'blue' in stage['block_info']:
                                        try:
                                            grid_b = add_grid_lines(b.copy(), stage['block_info']['blue'])
                                            grid_path = f"{image_dir}/with_grid/stage_{i}_blue_channel_grid.png"
                                            save_image(grid_b, grid_path)
                                        except Exception as e:
                                            print(f"Warning: Could not create grid for blue channel: {e}")
                            
                        # 創建 Quadtree 分割視覺化
                        if 'block_info' in stage:
                            if is_grayscale_img:
                                try:
                                    quadtree_viz = visualize_quadtree(stage['block_info'], origImg.shape)
                                    viz_path = f"{image_dir}/quadtree_visualization/stage_{i}_quadtree.png"
                                    save_image(quadtree_viz, viz_path)
                                except Exception as e:
                                    print(f"Warning: Could not create quadtree visualization: {e}")
                            else:
                                # 彩色圖像的quadtree分割視覺化 - 為每個通道創建單獨的視覺化
                                if 'blue' in stage['block_info']:
                                    try:
                                        b, g, r = cv2.split(origImg)
                                        blue_viz = visualize_quadtree(stage['block_info']['blue'], b.shape)
                                        viz_path = f"{image_dir}/quadtree_visualization/stage_{i}_blue_quadtree.png"
                                        save_image(blue_viz, viz_path)
                                    except Exception as e:
                                        print(f"Warning: Could not create blue channel quadtree visualization: {e}")
                            
                        # 創建區塊大小分布統計
                        if 'block_info' in stage:
                            if is_grayscale_img:
                                try:
                                    create_block_size_distribution_chart(
                                        stage['block_info'], 
                                        f"{plots_dir}/block_distribution/stage_{i}_block_distribution.png",
                                        i
                                    )
                                except Exception as e:
                                    print(f"Warning: Could not create block size distribution chart: {e}")
                            else:
                                # 彩色圖像 - 僅為藍色通道創建統計
                                if 'blue' in stage['block_info']:
                                    try:
                                        create_block_size_distribution_chart(
                                            stage['block_info']['blue'], 
                                            f"{plots_dir}/block_distribution/stage_{i}_blue_distribution.png",
                                            i
                                        )
                                    except Exception as e:
                                        print(f"Warning: Could not create blue channel block distribution chart: {e}")
                            
                        # 創建旋轉角度視覺化（如果有）
                        if 'block_rotations' in stage:
                            if is_grayscale_img:
                                try:
                                    visualize_rotation_angles(
                                        stage['block_rotations'],
                                        origImg.shape,
                                        f"{image_dir}/quadtree_visualization/stage_{i}_rotation_angles.png"
                                    )
                                except Exception as e:
                                    print(f"Warning: Could not create rotation angles visualization: {e}")
                                
                # 打印階段統計資訊
                print(f"\nStage {i} metrics:")
                print(f"  Payload: {stage['payload']} bits")
                print(f"  BPP: {stage['bpp']:.4f}")
                print(f"  PSNR: {stage['psnr']:.2f}")
                print(f"  SSIM: {stage['ssim']:.4f}")
                print(f"  Hist Correlation: {stage['hist_corr']:.4f}")
                
                # 彩色圖像時，額外顯示每個通道的指標
                if not is_grayscale_img and 'channel_metrics' in stage:
                    print("\n  Channel metrics:")
                    for channel, metrics in stage['channel_metrics'].items():
                        print(f"    {channel.capitalize()}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, Hist_Corr={metrics['hist_corr']:.4f}")
                
                # Quadtree 方法特有的統計輸出
                if method == "quadtree" and 'block_info' in stage:
                    if is_grayscale_img:
                        print(f"\nBlock statistics for Stage {i}:")
                        for size in sorted([int(s) for s in stage['block_info'].keys()], reverse=True):
                            block_count = len(stage['block_info'][str(size)]['blocks'])
                            if block_count > 0:
                                rotation = stage['block_rotations'][size] if 'block_rotations' in stage else 0
                                print(f"  {size}x{size} blocks: {block_count}, Rotation: {rotation}°")
                        print("")
                    else:
                        # 彩色圖像時的區塊統計
                        if 'blue' in stage['block_info']:
                            try:
                                print(f"\nBlue channel block statistics for Stage {i}:")
                                for size in sorted([int(s) for s in stage['block_info']['blue'].keys()], reverse=True):
                                    block_count = len(stage['block_info']['blue'][str(size)]['blocks'])
                                    if block_count > 0:
                                        print(f"  {size}x{size} blocks: {block_count}")
                                print("")
                            except:
                                print("Warning: Cannot display blue channel block statistics.")

            # 儲存最終嵌入結果圖像
            final_img_path = f"{image_dir}/final_result.png"
            if is_grayscale_img:
                save_image(final_pee_img, final_img_path)
            else:
                cv2.imwrite(final_img_path, final_pee_img)
            
            # 原始與最終結果對比圖
            if is_proposed:
                compare_path = f"{image_dir}/original_vs_final.png"
                if is_grayscale_img:
                    save_comparison_image(origImg, final_pee_img, compare_path, 
                                        labels=("Original", "Embedded"))
                else:
                    # 彩色圖像的比較需要特殊處理
                    # 將兩張彩色圖像水平拼接
                    h1, w1 = origImg.shape[:2]
                    h2, w2 = final_pee_img.shape[:2]
                    
                    max_h = max(h1, h2)
                    combined = np.zeros((max_h, w1 + w2 + 5, 3), dtype=np.uint8)
                    
                    combined[:h1, :w1] = origImg
                    combined[:h2, w1+5:w1+5+w2] = final_pee_img
                    
                    # 添加標籤
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(combined, "Original", (10, 30), font, 0.8, (0, 0, 255), 2)
                    cv2.putText(combined, "Embedded", (w1 + 15, 30), font, 0.8, (0, 0, 255), 2)
                    
                    cv2.imwrite(compare_path, combined)
                
                # 創建最終嵌入熱圖
                if is_grayscale_img:
                    heatmap_path = f"{image_dir}/final_heatmap.png"
                    visualize_embedding_heatmap(origImg, final_pee_img, heatmap_path)
                else:
                    heatmap_path = f"{image_dir}/final_color_heatmap.png"
                    create_color_heatmap(origImg, final_pee_img, heatmap_path)
                    
                    # 儲存最終通道對比
                    create_color_channel_comparison(origImg, final_pee_img, f"{image_dir}/final_channel_comparison.png")
            
            # Quadtree 方法特有：儲存最終帶格線圖像
            if method == "quadtree" and 'block_info' in pee_stages[-1]:
                if is_grayscale_img:
                    final_grid_path = f"{image_dir}/with_grid/final_result_with_grid.png"
                    final_grid_image = add_grid_lines(final_pee_img.copy(), pee_stages[-1]['block_info'])
                    save_image(final_grid_image, final_grid_path)
                else:
                    # 彩色圖像 - 展示藍色通道的網格
                    if 'blue' in pee_stages[-1]['block_info']:
                        try:
                            b, g, r = cv2.split(final_pee_img)
                            final_grid_path = f"{image_dir}/with_grid/final_blue_channel_grid.png"
                            final_blue_grid = add_grid_lines(b.copy(), pee_stages[-1]['block_info']['blue'])
                            save_image(final_blue_grid, final_grid_path)
                        except Exception as e:
                            print(f"Warning: Could not create final blue channel grid: {e}")

            # 計算並儲存 BPP-PSNR 數據
            bpp_psnr_data = []
            accumulated_payload = 0
            for stage in pee_stages:
                accumulated_payload += stage['payload']
                bpp_psnr_data.append({
                    'stage': stage['embedding'],
                    'bpp': accumulated_payload / total_pixels,
                    'psnr': stage['psnr']
                })

            # 僅對 proposed 預測器繪製和儲存圖表
            if is_proposed:
                # 繪製並儲存 BPP-PSNR 曲線
                plt.figure(figsize=(12, 8))
                bpps = [data['bpp'] for data in bpp_psnr_data]
                psnrs = [data['psnr'] for data in bpp_psnr_data]
                
                plt.plot(bpps, psnrs, 'b.-', linewidth=2, markersize=8, 
                        label=f'Method: {method}, Predictor: {prediction_method.value}')
                
                # 為每個點添加標籤
                for i, (bpp, psnr) in enumerate(zip(bpps, psnrs)):
                    plt.annotate(f'Stage {i}\n({bpp:.3f}, {psnr:.2f})',
                                (bpp, psnr), 
                                textcoords="offset points",
                                xytext=(0,10), 
                                ha='center',
                                bbox=dict(boxstyle='round,pad=0.5', 
                                         fc='yellow', 
                                         alpha=0.3),
                                fontsize=8)
                
                plt.xlabel('Accumulated Bits Per Pixel (BPP)', fontsize=12)
                plt.ylabel('PSNR (dB)', fontsize=12)
                title = f'BPP-PSNR Curve for {imgName}\n'
                if method == "quadtree":
                    title += f'Min Block Size: {quad_tree_params["min_block_size"]}'
                elif method == "split":
                    title += f'Split Size: {split_size}x{split_size}, {"Block-based" if block_base else "Quarter-based"}'
                else:
                    title += f'Split Size: {split_size}x{split_size}'
                plt.title(title, fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=10)
                
                plt.margins(0.1)
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.savefig(f"{plots_dir}/bpp_psnr_curve.png", dpi=300, bbox_inches='tight')
                plt.close()

            # 儲存 BPP-PSNR 數據
            np.save(f"{base_dir}/data/{imgName}/{method}_{prediction_method.value}_bpp_psnr_data.npy",
                   {
                       'method': method,
                       'prediction_method': prediction_method.value,
                       'ratio_of_ones': ratio_of_ones,
                       'bpp': bpps,
                       'psnr': psnrs,
                       'split_size': split_size if method != "quadtree" else None,
                       'block_base': block_base if method == "split" else None,
                       'min_block_size': quad_tree_params['min_block_size'] if method == "quadtree" else None,
                       'stages': bpp_psnr_data,
                       'interval_stats': stats_df.to_dict('records') if stats_df is not None else None
                   })

            # 計算並輸出最終結果
            final_bpp = total_payload / total_pixels
            
            # 根據圖像類型計算最終品質指標
            if is_grayscale_img:
                final_psnr = calculate_psnr(origImg, final_pee_img)
                final_ssim = calculate_ssim(origImg, final_pee_img)
                hist_orig = generate_histogram(origImg)
                hist_final = generate_histogram(final_pee_img)
                final_hist_corr = histogram_correlation(hist_orig, hist_final)
            else:
                final_psnr, final_ssim, final_hist_corr = calculate_color_metrics(origImg, final_pee_img)

            print("\nFinal Results:")
            print(f"Image Type: {'Grayscale' if is_grayscale_img else 'Color'}")
            print(f"Method: {method}")
            print(f"Prediction Method: {prediction_method.value}")
            print(f"Ratio of Ones: {ratio_of_ones}")
            print(f"Total Payload: {total_payload}")
            print(f"Final BPP: {final_bpp:.4f}")
            print(f"Final PSNR: {final_psnr:.2f}")
            print(f"Final SSIM: {final_ssim:.4f}")
            print(f"Final Histogram Correlation: {final_hist_corr:.4f}")
            
            # 彩色圖像時，顯示每個通道的指標
            if not is_grayscale_img and 'channel_metrics' in pee_stages[-1]:
                print("\nFinal Channel Metrics:")
                for channel, metrics in pee_stages[-1]['channel_metrics'].items():
                    print(f"  {channel.capitalize()}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, Hist_Corr={metrics['hist_corr']:.4f}")

            # 更新最終結果儲存
            final_results = {
                'image_type': 'grayscale' if is_grayscale_img else 'color',
                'method': method,
                'prediction_method': prediction_method.value,
                'ratio_of_ones': ratio_of_ones,
                'total_payload': total_payload,
                'final_bpp': final_bpp,
                'final_psnr': final_psnr,
                'final_ssim': final_ssim,
                'final_hist_corr': final_hist_corr,
                'split_size': split_size,
                'block_base': block_base if method == "split" else None
            }
            
            if method == "quadtree":
                final_results.update({
                    'min_block_size': quad_tree_params['min_block_size'],
                    'variance_threshold': quad_tree_params['variance_threshold']
                })
            else:
                final_results.update({
                    'split_size': split_size,
                    'block_base': block_base if method == "split" else None
                })
            
            # 儲存最終結果
            np.save(f"{base_dir}/data/{imgName}/{method}_{prediction_method.value}_final_results.npy", final_results)
            print(f"Final results saved to {base_dir}/data/{imgName}/{method}_{prediction_method.value}_final_results.npy")

        except Exception as e:
                print(f"Error occurred in PEE process:")
                print(f"Method: {method}")
                print(f"Prediction method: {prediction_method.value}")
                print(f"Error details: {e}")
                import traceback
                traceback.print_exc()
                return

        print("PEE encoding process completed.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理 GPU 記憶體
        cleanup_memory()

if __name__ == "__main__":
    main()