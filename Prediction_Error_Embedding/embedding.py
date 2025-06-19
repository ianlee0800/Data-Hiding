"""
嵌入功能模組 - 重構版本
支援 rotation 和 split 方法的 PEE 嵌入
"""

from numba.core.errors import NumbaPerformanceWarning
import warnings
import numpy as np
import cupy as cp
import cv2

# 🔧 更新：從新的模組導入所需功能
from image_processing import (
    split_image_flexible,
    merge_image_flexible,
    PredictionMethod
)

# 🔧 更新：從 utils 導入工具函數和數據轉換
from utils import (
    generate_random_binary_array,
    generate_embedding_data,
    DataConverter,
    calculate_psnr,
    calculate_ssim,
    histogram_correlation,
    cleanup_memory
)

# 🔧 更新：從 pee 導入 EL 計算和嵌入核心功能
from pee import (
    compute_improved_adaptive_el,
    multi_pass_embedding,
    brute_force_weight_search_cuda
)

# 🔧 更新：從 image_processing 導入顏色處理功能
from image_processing import (
    split_color_channels,
    combine_color_channels
)

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

def pee_process_with_rotation_cuda(img, total_embeddings, ratio_of_ones, use_different_weights, 
                                 split_size, el_mode, prediction_method=None,
                                 target_payload_size=-1):
    """
    Using rotation PEE method with support for both grayscale and color images
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image (grayscale or color)
    (other parameters remain the same)
        
    Returns:
    --------
    tuple
        (final_pee_img, total_payload, pee_stages)
    """
    # Check if this is a color image by examining image shape
    if len(img.shape) == 3 and img.shape[2] == 3:
        # For color images, redirect to color image processing function
        print("Detected color image, redirecting to color image processing...")
        return pee_process_color_image_rotation_cuda(
            img, total_embeddings, ratio_of_ones, use_different_weights,
            split_size, el_mode, prediction_method, target_payload_size
        )

    # 🔧 更新：使用新的數據轉換工具
    original_img = DataConverter.to_cupy(img)
    height, width = img.shape
    total_pixels = height * width
    pee_stages = []
    total_payload = 0
    current_img = original_img.copy()
    previous_psnr = float('inf')
    previous_ssim = 1.0
    
    # 計算子圖像數量和每個子圖像的最大容量
    sub_images_per_stage = split_size * split_size
    max_capacity_per_subimage = (height * width) // sub_images_per_stage
    
    # 生成嵌入數據
    embedding_data = generate_embedding_data(
        total_embeddings=total_embeddings,
        sub_images_per_stage=sub_images_per_stage,
        max_capacity_per_subimage=max_capacity_per_subimage,
        ratio_of_ones=ratio_of_ones,
        target_payload_size=target_payload_size
    )
    
    # 設定剩餘目標payload
    remaining_target = target_payload_size if target_payload_size > 0 else None

    # 開始逐階段處理
    for embedding in range(total_embeddings):
        print(f"\nStarting embedding {embedding}")
        stage_data = embedding_data['stage_data'][embedding]
        
        if remaining_target is not None:
            print(f"Remaining target payload: {remaining_target}")
            if remaining_target <= 0:
                print("Target payload reached. Stage will only process image without embedding.")
                break
        
        # 設定目標品質參數
        if embedding == 0:
            target_psnr = 40.0
            target_bpp = 0.9
        else:
            target_psnr = max(28.0, previous_psnr - 1)
            target_bpp = max(0.5, (total_payload / total_pixels) * 0.95)
        
        print(f"Target PSNR: {target_psnr:.2f}, Target BPP: {target_bpp:.4f}")
        
        # 初始化階段資訊
        stage_info = {
            'embedding': embedding,
            'sub_images': [],
            'payload': 0,
            'psnr': 0,
            'ssim': 0,
            'hist_corr': 0,
            'bpp': 0,
            'block_params': [],
            'original_img': DataConverter.to_numpy(original_img)  # 新增：保存原始圖像
        }
        
        stage_payload = 0
        embedded_sub_images = []
        
        # 計算當前階段的旋轉角度
        stage_rotation = embedding * 90
        
        # 旋轉當前圖像
        if stage_rotation != 0:
            rotated_img = cp.rot90(current_img, k=stage_rotation // 90)
        else:
            rotated_img = current_img
        
        # 保存旋轉後的圖像 (新增)
        stage_info['rotated_img'] = rotated_img
        
        # 分割圖像
        sub_images = split_image_flexible(rotated_img, split_size, block_base=True)
        
        # 保存累計的預測圖像
        all_sub_preds = []
        
        # 處理每個子圖像
        for i, sub_img in enumerate(sub_images):
            # 檢查是否已達到目標payload
            if remaining_target is not None and remaining_target <= 0:
                embedded_sub_images.append(DataConverter.to_cupy(sub_img))
                continue
            
            sub_img = DataConverter.to_cupy(sub_img)
            sub_data = stage_data['sub_data'][i]
            sub_data = DataConverter.to_cupy(sub_data.astype(np.uint8))
            
            # 計算當前子圖像的嵌入目標
            if remaining_target is not None:
                current_target = min(len(sub_data), remaining_target)
            else:
                current_target = None
            
            # 根據 el_mode 決定 max_el
            if el_mode == 1:  # Increasing
                max_el = 3 + embedding * 2
            elif el_mode == 2:  # Decreasing
                max_el = 11 - embedding * 2
            else:  # No restriction
                max_el = 7
            
            # 🔧 更新：使用從 pee 模組導入的 EL 計算函數
            local_el = compute_improved_adaptive_el(sub_img, window_size=5, max_el=max_el)
            
            # 根據預測方法進行不同的處理
            if prediction_method == PredictionMethod.PROPOSED:
                if use_different_weights or i == 0:
                    weights, (sub_payload, sub_psnr) = brute_force_weight_search_cuda(
                        sub_img, sub_data, local_el, target_bpp, target_psnr, embedding
                    )
            else:
                # MED 和 GAP 方法不需要權重
                weights = None
            
            # 🔧 更新：使用從 pee 模組導入的嵌入函數
            embedded_sub, payload, pred_sub = multi_pass_embedding(
                sub_img,
                sub_data,
                local_el,
                weights,
                embedding,
                prediction_method=prediction_method,
                remaining_target=remaining_target
            )
            
            # 保存預測圖像
            all_sub_preds.append(pred_sub)
            
            # 如果是第一個子圖像，保存作為示例
            if i == 0:
                stage_info['sample_original_sub'] = DataConverter.to_numpy(sub_img)
                stage_info['sample_pred_sub'] = pred_sub
                stage_info['sample_embedded_sub'] = DataConverter.to_numpy(embedded_sub)
            
            # 更新剩餘目標量
            if remaining_target is not None:
                payload = min(payload, current_target)
                remaining_target -= payload
            
            embedded_sub_images.append(embedded_sub)
            stage_payload += payload
            
            # 🔧 更新：使用新的品質指標計算函數
            sub_img_np = DataConverter.to_numpy(sub_img)
            embedded_sub_np = DataConverter.to_numpy(embedded_sub)
            sub_psnr = calculate_psnr(sub_img_np, embedded_sub_np)
            sub_ssim = calculate_ssim(sub_img_np, embedded_sub_np)
            sub_hist_corr = histogram_correlation(
                np.histogram(sub_img_np, bins=256, range=(0, 255))[0],
                np.histogram(embedded_sub_np, bins=256, range=(0, 255))[0]
            )
            
            # 記錄區塊資訊
            block_info = {
                'weights': (weights.tolist() if weights is not None and hasattr(weights, 'tolist') 
                          else "N/A" if prediction_method in [PredictionMethod.MED, PredictionMethod.GAP] 
                          else None),
                'EL': int(DataConverter.to_numpy(local_el).max()),
                'payload': int(payload),
                'psnr': float(sub_psnr),
                'ssim': float(sub_ssim),
                'rotation': stage_rotation,
                'hist_corr': float(sub_hist_corr),
                'prediction_method': prediction_method.value,
                'original_img': sub_img_np,         # 新增
                'pred_img': pred_sub,               # 新增
                'embedded_img': embedded_sub_np     # 新增
            }
            stage_info['sub_images'].append(block_info)
            stage_info['block_params'].append(block_info)
        
        # 合併處理後的子圖像
        stage_img = merge_image_flexible(embedded_sub_images, split_size, block_base=True)
        
        # 嘗試合併預測圖像 (如果可能)
        if all_sub_preds:
            try:
                pred_img_merged = merge_image_flexible([DataConverter.to_cupy(p) for p in all_sub_preds], 
                                                     split_size, block_base=True)
                stage_info['pred_img'] = DataConverter.to_numpy(pred_img_merged)
            except Exception as e:
                print(f"Warning: Could not merge prediction images: {e}")
                # 如果不能合併，使用第一個子圖像的預測作為示例
                if 'sample_pred_sub' in stage_info:
                    stage_info['pred_img'] = stage_info['sample_pred_sub']
        
        # 將圖像旋轉回原始方向
        if stage_rotation != 0:
            stage_img = cp.rot90(stage_img, k=-stage_rotation // 90)
            # 如果有合併的預測圖像，也需要旋轉回來
            if 'pred_img' in stage_info:
                if isinstance(stage_info['pred_img'], cp.ndarray):
                    stage_info['pred_img'] = DataConverter.to_numpy(cp.rot90(DataConverter.to_cupy(stage_info['pred_img']), 
                                                              k=-stage_rotation // 90))
                else:
                    stage_info['pred_img'] = np.rot90(stage_info['pred_img'], 
                                                     k=-stage_rotation // 90)
        
        stage_info['stage_img'] = stage_img
        
        # 🔧 更新：使用新的品質指標計算函數
        stage_img_np = DataConverter.to_numpy(stage_img)
        original_img_np = DataConverter.to_numpy(original_img)
        stage_info['psnr'] = float(calculate_psnr(original_img_np, stage_img_np))
        stage_info['ssim'] = float(calculate_ssim(original_img_np, stage_img_np))
        stage_info['hist_corr'] = float(histogram_correlation(
            np.histogram(original_img_np, bins=256, range=(0, 255))[0],
            np.histogram(stage_img_np, bins=256, range=(0, 255))[0]
        ))
        stage_info['payload'] = stage_payload
        stage_info['bpp'] = float(stage_info['payload'] / total_pixels)
        stage_info['prediction_method'] = prediction_method.value
        
        # 輸出階段摘要
        print(f"\nEmbedding {embedding} summary:")
        print(f"Prediction Method: {prediction_method.value}")
        print(f"Payload: {stage_info['payload']}")
        print(f"BPP: {stage_info['bpp']:.4f}")
        print(f"PSNR: {stage_info['psnr']:.2f}")
        print(f"SSIM: {stage_info['ssim']:.4f}")
        print(f"Hist Corr: {stage_info['hist_corr']:.4f}")
        print(f"Rotation: {stage_rotation}°")
        
        # 更新資訊
        pee_stages.append(stage_info)
        total_payload += stage_payload
        previous_psnr = stage_info['psnr']
        previous_ssim = stage_info['ssim']
        
        current_img = stage_img
        
        # 檢查是否已達到總目標
        if remaining_target is not None and remaining_target <= 0:
            print(f"Reached target payload ({target_payload_size} bits) at stage {embedding}")
            break

    # 返回最終結果
    final_pee_img = DataConverter.to_numpy(current_img)
    return final_pee_img, int(total_payload), pee_stages

def pee_process_color_image_rotation_cuda(img, total_embeddings, ratio_of_ones, use_different_weights, 
                                       split_size, el_mode, prediction_method=None,
                                       target_payload_size=-1):
    """
    Process a color image using rotation PEE method with independent channel processing
    
    🔧 修改版本：每個通道都當作獨立的灰階圖像處理，發揮彩色圖像的真正3倍容量
    """
    
    if prediction_method is None:
        prediction_method = PredictionMethod.PROPOSED
    
    print(f"Processing color image with independent channel rotation method using {prediction_method.value}")
    
    # 🔧 核心改變：分離彩色通道後，每個都當作獨立的灰階圖像處理
    b_channel, g_channel, r_channel = split_color_channels(img)
    channels = [b_channel, g_channel, r_channel]
    channel_names = ['blue', 'green', 'red']
    
    # 總嵌入量追蹤
    total_payload = 0
    color_pee_stages = []
    final_channels = []
    
    # 🔧 關鍵修改：每個通道的目標容量計算
    if target_payload_size > 0:
        # 如果有指定目標容量，平均分配給三個通道
        channel_target = target_payload_size // 3
        print(f"Target payload distributed: {channel_target} bits per channel")
        print(f"Total target: {target_payload_size} bits across all channels")
    else:
        # 如果是最大容量模式，每個通道都用最大容量
        channel_target = -1
        print("Using maximum capacity for each channel independently")
        print("Expected total capacity: ~3x equivalent grayscale image")
    
    # 🔧 獨立處理每個通道 - 這是關鍵改變
    for ch_idx, (channel, ch_name) in enumerate(zip(channels, channel_names)):
        print(f"\n{'='*60}")
        print(f"Processing {ch_name} channel as independent grayscale image")
        print(f"Channel shape: {channel.shape}")
        print(f"{'='*60}")
        
        try:
            # 🔧 核心修改：每個通道都調用完整的灰階處理函數
            final_ch_img, ch_payload, ch_stages = pee_process_with_rotation_cuda(
                channel,                    # 當作灰階圖像處理
                total_embeddings,           # 使用相同的嵌入階段數
                ratio_of_ones,              # 使用相同的數據比例
                use_different_weights,      # 使用相同的權重策略
                split_size,                 # 使用相同的分割大小
                el_mode,                    # 使用相同的EL模式
                prediction_method=prediction_method,
                target_payload_size=channel_target  # 每個通道的目標容量
            )
            
            final_channels.append(final_ch_img)
            total_payload += ch_payload
            
            print(f"{ch_name} channel processed successfully:")
            print(f"  Payload: {ch_payload} bits")
            if len(ch_stages) > 0:
                final_stage = ch_stages[-1]
                print(f"  Final PSNR: {final_stage['psnr']:.2f}")
                print(f"  Final SSIM: {final_stage['ssim']:.4f}")
                print(f"  Final BPP: {final_stage['bpp']:.6f}")
            
            # 🔧 合併階段資訊 - 保持與原有格式的兼容性
            for i, stage in enumerate(ch_stages):
                # 確保有足夠的階段容器
                while len(color_pee_stages) <= i:
                    color_pee_stages.append({
                        'embedding': i,
                        'payload': 0,
                        'channel_payloads': {'blue': 0, 'green': 0, 'red': 0},
                        'bpp': 0,
                        'psnr': 0,
                        'ssim': 0,
                        'hist_corr': 0,
                        'channel_metrics': {
                            'blue': {'psnr': 0, 'ssim': 0, 'hist_corr': 0},
                            'green': {'psnr': 0, 'ssim': 0, 'hist_corr': 0},
                            'red': {'psnr': 0, 'ssim': 0, 'hist_corr': 0}
                        },
                        'prediction_method': prediction_method.value,
                        'split_size': split_size,
                        'original_img': img,  # 保存原始彩色圖像
                        'block_params': []  # 用於兼容現有的表格生成函數
                    })
                
                # 添加通道特定的資訊
                combined_stage = color_pee_stages[i]
                combined_stage['channel_payloads'][ch_name] = stage['payload']
                combined_stage['channel_metrics'][ch_name] = {
                    'psnr': stage['psnr'],
                    'ssim': stage['ssim'],
                    'hist_corr': stage['hist_corr']
                }
                
                # 累加總payload
                combined_stage['payload'] += stage['payload']
                
                # 🔧 保存各種圖像資訊以便後續可視化
                # 保存階段圖像（每個通道處理完後更新）
                if 'channel_imgs' not in combined_stage:
                    combined_stage['channel_imgs'] = {}
                combined_stage['channel_imgs'][ch_name] = DataConverter.to_numpy(stage['stage_img']) if isinstance(stage['stage_img'], cp.ndarray) else stage['stage_img']
                
                # 保存原始圖像（rotation方法特有）
                if 'original_img' not in combined_stage:
                    combined_stage['original_img'] = img
                
                # 保存預測圖像（如果有）
                if 'pred_img' in stage:
                    if 'channel_pred_imgs' not in combined_stage:
                        combined_stage['channel_pred_imgs'] = {}
                    combined_stage['channel_pred_imgs'][ch_name] = stage['pred_img']
                
                # 保存子圖像資訊
                if 'channel_sub_images' not in combined_stage:
                    combined_stage['channel_sub_images'] = {}
                if 'sub_images' in stage:
                    combined_stage['channel_sub_images'][ch_name] = stage['sub_images']
                
                # 🔧 合併區塊參數（為了兼容現有的表格生成函數）
                if 'block_params' in stage:
                    # 如果是第一個通道，初始化block_params結構
                    if ch_idx == 0:
                        combined_stage['block_params'] = []
                        for j, block_param in enumerate(stage['block_params']):
                            combined_stage['block_params'].append({
                                'channel_params': {
                                    'blue': {},
                                    'green': {},
                                    'red': {}
                                },
                                # 使用第一個通道的基本參數
                                'weights': block_param.get('weights', 'N/A'),
                                'EL': block_param.get('EL', 0),
                                'payload': 0,  # 會累加
                                'psnr': 0,     # 會平均
                                'ssim': 0,     # 會平均
                                'hist_corr': 0, # 會平均
                                'rotation': block_param.get('rotation', 0),
                                'prediction_method': prediction_method.value
                            })
                    
                    # 保存通道特定的區塊參數並累加/平均化指標
                    for j, block_param in enumerate(stage['block_params']):
                        if j < len(combined_stage['block_params']):
                            # 保存通道特定的參數
                            combined_stage['block_params'][j]['channel_params'][ch_name] = block_param
                            
                            # 累加payload
                            combined_stage['block_params'][j]['payload'] += block_param.get('payload', 0)
                            
                            # 累加指標（最後會除以3）
                            combined_stage['block_params'][j]['psnr'] += block_param.get('psnr', 0)
                            combined_stage['block_params'][j]['ssim'] += block_param.get('ssim', 0)
                            combined_stage['block_params'][j]['hist_corr'] += block_param.get('hist_corr', 0)
                            
                            # 在最後一個通道時計算平均值
                            if ch_idx == 2:  # 紅色通道（最後一個）
                                combined_stage['block_params'][j]['psnr'] /= 3
                                combined_stage['block_params'][j]['ssim'] /= 3
                                combined_stage['block_params'][j]['hist_corr'] /= 3
            
            # 🔧 更新：使用新的記憶體清理函數
            cleanup_memory()
            
        except Exception as e:
            print(f"Error processing {ch_name} channel: {str(e)}")
            print(f"Using original channel data for {ch_name}")
            # 如果某個通道處理失敗，使用原始通道
            final_channels.append(channel)
            continue
    
    # 🔧 重新組合彩色圖像
    if len(final_channels) == 3:
        final_color_img = combine_color_channels(final_channels[0], final_channels[1], final_channels[2])
        print(f"\nSuccessfully combined all three channels")
    else:
        print(f"Warning: Only {len(final_channels)} channels processed successfully")
        print("Using original image as fallback")
        final_color_img = img
    
    # 🔧 計算合併階段的整體指標
    pixel_count = img.shape[0] * img.shape[1]  # 只計算像素位置數，不包含通道數
    
    for stage in color_pee_stages:
        # 🔧 修正BPP計算：總payload除以像素位置數（不包含通道數）
        stage['bpp'] = stage['payload'] / pixel_count
        
        # 計算平均品質指標
        channel_metrics = stage['channel_metrics']
        stage['psnr'] = sum(channel_metrics[ch]['psnr'] for ch in channel_names) / 3
        stage['ssim'] = sum(channel_metrics[ch]['ssim'] for ch in channel_names) / 3
        stage['hist_corr'] = sum(channel_metrics[ch]['hist_corr'] for ch in channel_names) / 3
        
        # 🔧 合併階段圖像（如果所有通道都有的話）
        if 'channel_imgs' in stage and len(stage['channel_imgs']) == 3:
            stage['stage_img'] = combine_color_channels(
                stage['channel_imgs']['blue'],
                stage['channel_imgs']['green'], 
                stage['channel_imgs']['red']
            )
        
        # 輸出階段摘要
        print(f"\nColor Rotation Stage {stage['embedding']} summary:")
        print(f"  Total Payload: {stage['payload']} bits")
        print(f"  BPP: {stage['bpp']:.6f}")
        print(f"  Average PSNR: {stage['psnr']:.2f}")
        print(f"  Average SSIM: {stage['ssim']:.4f}")
        print(f"  Average Hist Corr: {stage['hist_corr']:.4f}")
        print("  Channel payloads:")
        for ch_name, payload in stage['channel_payloads'].items():
            print(f"    {ch_name.capitalize()}: {payload} bits")
    
    # 🔧 輸出最終結果摘要
    print(f"\n{'='*80}")
    print(f"Final Independent Channel Rotation Processing Results:")
    print(f"Image type: Color ({img.shape[0]}x{img.shape[1]}x{img.shape[2]})")
    print(f"Total Payload: {total_payload} bits")
    print(f"Total BPP: {total_payload / pixel_count:.6f}")
    
    if len(color_pee_stages) > 0:
        final_stage = color_pee_stages[-1]
        print(f"Final Average PSNR: {final_stage['psnr']:.2f}")
        print(f"Final Average SSIM: {final_stage['ssim']:.4f}")
        print(f"Final Average Hist Corr: {final_stage['hist_corr']:.4f}")
    
    print("Final channel payloads:")
    if len(color_pee_stages) > 0:
        final_payloads = color_pee_stages[-1]['channel_payloads']
        total_channel_bpp = 0
        for ch_name, payload in final_payloads.items():
            channel_bpp = payload / pixel_count
            total_channel_bpp += channel_bpp
            print(f"  {ch_name.capitalize()}: {payload} bits (BPP: {channel_bpp:.6f})")
        print(f"  Total BPP (sum of channels): {total_channel_bpp:.6f}")
    
    # 🔧 與等效灰階圖像的容量比較分析
    print(f"\nBPP Analysis:")
    print(f"  Pixel positions: {pixel_count}")
    print(f"  Total payload: {total_payload} bits")
    print(f"  Color image BPP: {total_payload / pixel_count:.6f}")
    print(f"  This represents ~{total_payload / pixel_count:.1f} bits per pixel position across all channels")
    print(f"  Compared to equivalent grayscale: {(total_payload / pixel_count):.2f}x higher capacity potential")
    print(f"{'='*80}")
    
    return final_color_img, int(total_payload), color_pee_stages

def pee_process_with_split_cuda(img, total_embeddings, ratio_of_ones, use_different_weights, 
                              split_size, el_mode, block_base, 
                              prediction_method=None,
                              target_payload_size=-1):
    """
    Using split PEE method with support for both grayscale and color images
    Enhanced with rotation effect tracking for visualization
    """
    # Check if this is a color image by examining image shape
    if len(img.shape) == 3 and img.shape[2] == 3:
        # For color images, redirect to color image processing function
        print("Detected color image, redirecting to color image processing...")
        return pee_process_color_image_split_cuda(
            img, total_embeddings, ratio_of_ones, use_different_weights,
            split_size, el_mode, block_base, prediction_method, target_payload_size
        )

    # 🔧 更新：使用新的數據轉換工具
    original_img = DataConverter.to_cupy(img)
    height, width = original_img.shape
    total_pixels = height * width
    
    # 計算子圖像數量和每個子圖像的最大容量
    sub_images_per_stage = split_size * split_size
    max_capacity_per_subimage = (height * width) // sub_images_per_stage
    
    # 生成嵌入數據
    embedding_data = generate_embedding_data(
        total_embeddings=total_embeddings,
        sub_images_per_stage=sub_images_per_stage,
        max_capacity_per_subimage=max_capacity_per_subimage,
        ratio_of_ones=ratio_of_ones,
        target_payload_size=target_payload_size
    )
    
    # 初始化追蹤變數
    pee_stages = []
    total_payload = 0
    current_img = original_img.copy()
    previous_psnr = float('inf')
    previous_ssim = 1.0
    
    # 設定剩餘目標payload
    remaining_target = target_payload_size if target_payload_size > 0 else None
    
    # 開始逐階段處理
    for embedding in range(total_embeddings):
        print(f"\nStarting embedding {embedding}")
        stage_data = embedding_data['stage_data'][embedding]
        
        # 輸出當前階段的目標資訊
        if remaining_target is not None:
            print(f"Remaining target payload: {remaining_target}")
        
        # 設定目標品質參數
        if embedding == 0:
            target_psnr = 40.0
            target_bpp = 0.9
        else:
            target_psnr = max(28.0, previous_psnr - 1)
            target_bpp = max(0.5, (total_payload / total_pixels) * 0.95)
        
        print(f"Target PSNR: {target_psnr:.2f}, Target BPP: {target_bpp:.4f}")
        
        # 💡 新增：初始化旋轉效果追蹤
        stage_rotations = cp.random.choice([0, 90, 180, 270], 
                                         size=split_size * split_size)
        rotated_embedded_sub_images = []  # 保存旋轉後未旋轉回來的子圖像
        
        # 初始化階段資訊
        stage_info = {
            'embedding': embedding,
            'sub_images': [],
            'payload': 0,
            'psnr': 0,
            'ssim': 0,
            'hist_corr': 0,
            'bpp': 0,
            'block_params': [],
            # 💡 新增：為階段資訊添加旋轉相關欄位
            'split_size': split_size,
            'block_base': block_base,
            'rotations': DataConverter.to_numpy(stage_rotations).tolist(),  # 轉換為可序列化格式
            'original_img': DataConverter.to_numpy(original_img)  # 保存原始圖像
        }
        
        stage_payload = 0
        embedded_sub_images = []
        
        # 使用彈性分割函數切割圖像
        sub_images = split_image_flexible(current_img, split_size, block_base)
        
        # 保存累計的預測圖像
        all_sub_preds = []
        
        # 處理每個子圖像
        for i, sub_img in enumerate(sub_images):
            # 檢查是否已達到目標payload
            if remaining_target is not None and remaining_target <= 0:
                embedded_sub_images.append(DataConverter.to_cupy(sub_img))
                # 💡 對於未嵌入的子圖像，也保存旋轉版本以保持一致性
                rotation = int(stage_rotations[i])
                rotated_sub_img = cp.rot90(DataConverter.to_cupy(sub_img), k=rotation // 90)
                rotated_embedded_sub_images.append(rotated_sub_img)
                continue
            
            # 準備子圖像處理
            sub_img = DataConverter.to_cupy(sub_img)
            rotation = int(stage_rotations[i])
            rotated_sub_img = cp.rot90(sub_img, k=rotation // 90)
            
            # 準備嵌入數據
            sub_data = stage_data['sub_data'][i]
            sub_data = DataConverter.to_cupy(sub_data.astype(np.uint8))
            
            # 計算當前子圖像的嵌入目標
            if remaining_target is not None:
                current_target = min(len(sub_data), remaining_target)
            else:
                current_target = None
            
            # 根據 el_mode 決定 max_el
            if el_mode == 1:
                max_el = 3 + embedding * 2
            elif el_mode == 2:
                max_el = 11 - embedding * 2
            else:
                max_el = 7
            
            # 🔧 更新：使用從 pee 模組導入的 EL 計算函數
            local_el = compute_improved_adaptive_el(
                rotated_sub_img, 
                window_size=5, 
                max_el=max_el
            )
            
            # 根據預測方法進行不同的處理
            if prediction_method == PredictionMethod.PROPOSED:
                # 只有 PROPOSED 方法需要進行權重搜索
                if use_different_weights or i == 0:
                    weights, (sub_payload, sub_psnr) = brute_force_weight_search_cuda(
                        rotated_sub_img, sub_data, local_el, target_bpp, target_psnr, embedding
                    )
            else:
                # MED 和 GAP 方法不需要權重
                weights = None
            
            # 🔧 更新：使用從 pee 模組導入的嵌入函數
            embedded_sub, payload, pred_sub = multi_pass_embedding(
                rotated_sub_img,
                sub_data,
                local_el,
                weights,
                embedding,
                prediction_method=prediction_method,
                remaining_target=current_target
            )
            
            # 保存預測圖像
            all_sub_preds.append(pred_sub)
            
            # 💡 關鍵修改：保存旋轉後的嵌入結果
            rotated_embedded_sub_images.append(embedded_sub)  # 旋轉狀態的結果
            
            # 如果是第一個子圖像，保存作為示例
            if i == 0:
                stage_info['sample_original_sub'] = DataConverter.to_numpy(sub_img)
                stage_info['sample_pred_sub'] = pred_sub
                stage_info['sample_embedded_sub'] = DataConverter.to_numpy(embedded_sub)
                stage_info['sample_rotated_embedded_sub'] = DataConverter.to_numpy(embedded_sub)  # 新增旋轉版本
            
            # 更新剩餘目標量
            if remaining_target is not None:
                payload = min(payload, current_target)
                remaining_target -= payload
            
            # 將嵌入後的圖像旋轉回原始方向（原有邏輯）
            rotated_back_sub = cp.rot90(embedded_sub, k=-rotation // 90)
            embedded_sub_images.append(rotated_back_sub)
            stage_payload += payload
            
            # 🔧 更新：使用新的品質指標計算函數
            sub_img_np = DataConverter.to_numpy(sub_img)
            rotated_back_sub_np = DataConverter.to_numpy(rotated_back_sub)
            sub_psnr = calculate_psnr(sub_img_np, rotated_back_sub_np)
            sub_ssim = calculate_ssim(sub_img_np, rotated_back_sub_np)
            sub_hist_corr = histogram_correlation(
                np.histogram(sub_img_np, bins=256, range=(0, 255))[0],
                np.histogram(rotated_back_sub_np, bins=256, range=(0, 255))[0]
            )
            
            # 記錄區塊資訊
            block_info = {
                'weights': (weights.tolist() if weights is not None and hasattr(weights, 'tolist') 
                            else "N/A" if prediction_method in [PredictionMethod.MED, PredictionMethod.GAP] 
                            else None),
                'EL': int(DataConverter.to_numpy(local_el).max()),
                'payload': int(payload),
                'psnr': float(sub_psnr),
                'ssim': float(sub_ssim),
                'rotation': rotation,
                'hist_corr': float(sub_hist_corr),
                'prediction_method': prediction_method.value,
                'original_img': sub_img_np,         # 新增
                'pred_img': pred_sub,               # 新增
                'embedded_img': rotated_back_sub_np,     # 新增
                'rotated_embedded_img': DataConverter.to_numpy(embedded_sub)  # 💡 新增：旋轉版本
            }
            stage_info['sub_images'].append(block_info)
            stage_info['block_params'].append(block_info)
        
        # 💡 新增：保存旋轉效果到階段資訊
        stage_info['rotated_sub_images'] = rotated_embedded_sub_images
        
        # 合併處理後的子圖像
        stage_img = merge_image_flexible(embedded_sub_images, split_size, block_base)
        
        # 💡 新增：合併旋轉後的子圖像（用於視覺化）
        if rotated_embedded_sub_images:
            try:
                rotated_stage_img = merge_image_flexible(rotated_embedded_sub_images, split_size, block_base)
                stage_info['rotated_stage_img'] = rotated_stage_img
            except Exception as e:
                print(f"Warning: Could not merge rotated sub-images: {e}")
        
        # 嘗試合併預測圖像 (如果可能)
        if all_sub_preds:
            try:
                pred_img_merged = merge_image_flexible([DataConverter.to_cupy(p) for p in all_sub_preds], 
                                                     split_size, block_base)
                stage_info['pred_img'] = DataConverter.to_numpy(pred_img_merged)
            except Exception as e:
                print(f"Warning: Could not merge prediction images: {e}")
                # 如果不能合併，使用第一個子圖像的預測作為示例
                if 'sample_pred_sub' in stage_info:
                    stage_info['pred_img'] = stage_info['sample_pred_sub']
        
        stage_info['stage_img'] = stage_img
        
        # 🔧 更新：使用新的品質指標計算函數
        stage_img_np = DataConverter.to_numpy(stage_img)
        original_img_np = DataConverter.to_numpy(original_img)
        stage_info['psnr'] = float(calculate_psnr(original_img_np, stage_img_np))
        stage_info['ssim'] = float(calculate_ssim(original_img_np, stage_img_np))
        stage_info['hist_corr'] = float(histogram_correlation(
            np.histogram(original_img_np, bins=256, range=(0, 255))[0],
            np.histogram(stage_img_np, bins=256, range=(0, 255))[0]
        ))
        stage_info['payload'] = stage_payload
        stage_info['bpp'] = float(stage_info['payload'] / total_pixels)
        stage_info['prediction_method'] = prediction_method.value
        
        # 更新資訊
        pee_stages.append(stage_info)
        total_payload += stage_payload
        previous_psnr = stage_info['psnr']
        previous_ssim = stage_info['ssim']
        
        # 輸出階段摘要
        print(f"\nEmbedding {embedding} summary:")
        print(f"Prediction Method: {prediction_method.value}")
        print(f"Payload: {stage_info['payload']}")
        print(f"BPP: {stage_info['bpp']:.4f}")
        print(f"PSNR: {stage_info['psnr']:.2f}")
        print(f"SSIM: {stage_info['ssim']:.4f}")
        print(f"Hist Corr: {stage_info['hist_corr']:.4f}")
        print(f"Split info: {split_size}x{split_size}, {'Block-based' if block_base else 'Quarter-based'}")
        
        current_img = stage_img

    # 返回最終結果
    final_pee_img = DataConverter.to_numpy(current_img)
    return final_pee_img, int(total_payload), pee_stages

def pee_process_color_image_split_cuda(img, total_embeddings, ratio_of_ones, use_different_weights, 
                                     split_size, el_mode, block_base, 
                                     prediction_method=None,
                                     target_payload_size=-1):
    """
    Process a color image using split PEE method with independent channel processing
    
    🔧 修改版本：每個通道都當作獨立的灰階圖像處理，發揮彩色圖像的真正3倍容量
    """
    if prediction_method is None:
        prediction_method = PredictionMethod.PROPOSED
    
    print(f"Processing color image with independent channel method using {prediction_method.value}")
    
    # 🔧 核心改變：分離彩色通道後，每個都當作獨立的灰階圖像處理
    b_channel, g_channel, r_channel = split_color_channels(img)
    channels = [b_channel, g_channel, r_channel]
    channel_names = ['blue', 'green', 'red']
    
    # 總嵌入量追蹤
    total_payload = 0
    color_pee_stages = []
    final_channels = []
    
    # 🔧 關鍵修改：每個通道的目標容量計算
    if target_payload_size > 0:
        # 如果有指定目標容量，平均分配給三個通道
        channel_target = target_payload_size // 3
        print(f"Target payload distributed: {channel_target} bits per channel")
        print(f"Total target: {target_payload_size} bits across all channels")
    else:
        # 如果是最大容量模式，每個通道都用最大容量
        channel_target = -1
        print("Using maximum capacity for each channel independently")
        print("Expected total capacity: ~3x equivalent grayscale image")
    
    # 🔧 獨立處理每個通道 - 這是關鍵改變
    for ch_idx, (channel, ch_name) in enumerate(zip(channels, channel_names)):
        print(f"\n{'='*60}")
        print(f"Processing {ch_name} channel as independent grayscale image")
        print(f"Channel shape: {channel.shape}")
        print(f"{'='*60}")
        
        try:
            # 🔧 核心修改：每個通道都調用完整的灰階處理函數
            final_ch_img, ch_payload, ch_stages = pee_process_with_split_cuda(
                channel,                    # 當作灰階圖像處理
                total_embeddings,           # 使用相同的嵌入階段數
                ratio_of_ones,              # 使用相同的數據比例
                use_different_weights,      # 使用相同的權重策略
                split_size,                 # 使用相同的分割大小
                el_mode,                    # 使用相同的EL模式
                block_base,                 # 使用相同的分割方式
                prediction_method=prediction_method,
                target_payload_size=channel_target  # 每個通道的目標容量
            )
            
            final_channels.append(final_ch_img)
            total_payload += ch_payload
            
            print(f"{ch_name} channel processed successfully:")
            print(f"  Payload: {ch_payload} bits")
            if len(ch_stages) > 0:
                final_stage = ch_stages[-1]
                print(f"  Final PSNR: {final_stage['psnr']:.2f}")
                print(f"  Final SSIM: {final_stage['ssim']:.4f}")
                print(f"  Final BPP: {final_stage['bpp']:.6f}")
            
            # 🔧 合併階段資訊 - 保持與原有格式的兼容性
            for i, stage in enumerate(ch_stages):
                # 確保有足夠的階段容器
                while len(color_pee_stages) <= i:
                    color_pee_stages.append({
                        'embedding': i,
                        'payload': 0,
                        'channel_payloads': {'blue': 0, 'green': 0, 'red': 0},
                        'bpp': 0,
                        'psnr': 0,
                        'ssim': 0,
                        'hist_corr': 0,
                        'channel_metrics': {
                            'blue': {'psnr': 0, 'ssim': 0, 'hist_corr': 0},
                            'green': {'psnr': 0, 'ssim': 0, 'hist_corr': 0},
                            'red': {'psnr': 0, 'ssim': 0, 'hist_corr': 0}
                        },
                        'prediction_method': prediction_method.value,
                        'split_size': split_size,
                        'block_base': block_base,
                        'original_img': img,  # 保存原始彩色圖像
                        'block_params': []  # 用於兼容現有的表格生成函數
                    })
                
                # 添加通道特定的資訊
                combined_stage = color_pee_stages[i]
                combined_stage['channel_payloads'][ch_name] = stage['payload']
                combined_stage['channel_metrics'][ch_name] = {
                    'psnr': stage['psnr'],
                    'ssim': stage['ssim'],
                    'hist_corr': stage['hist_corr']
                }
                
                # 累加總payload
                combined_stage['payload'] += stage['payload']
                
                # 🔧 保存各種圖像資訊以便後續可視化
                # 保存階段圖像（每個通道處理完後更新）
                if 'channel_imgs' not in combined_stage:
                    combined_stage['channel_imgs'] = {}
                combined_stage['channel_imgs'][ch_name] = DataConverter.to_numpy(stage['stage_img']) if isinstance(stage['stage_img'], cp.ndarray) else stage['stage_img']
                
                # 保存子圖像資訊
                if 'channel_sub_images' not in combined_stage:
                    combined_stage['channel_sub_images'] = {}
                if 'sub_images' in stage:
                    combined_stage['channel_sub_images'][ch_name] = stage['sub_images']
                
                # 保存旋轉子圖像資訊（split方法的特色）
                if 'rotated_sub_images' in stage:
                    if 'channel_rotated_sub_images' not in combined_stage:
                        combined_stage['channel_rotated_sub_images'] = {}
                    combined_stage['channel_rotated_sub_images'][ch_name] = stage['rotated_sub_images']
                
                # 保存旋轉角度資訊
                if 'rotations' in stage and 'rotations' not in combined_stage:
                    combined_stage['rotations'] = stage['rotations']  # 所有通道應該使用相同的旋轉
                
                # 🔧 合併區塊參數（為了兼容現有的表格生成函數）
                if 'block_params' in stage:
                    # 如果是第一個通道，初始化block_params結構
                    if ch_idx == 0:
                        combined_stage['block_params'] = []
                        for j, block_param in enumerate(stage['block_params']):
                            combined_stage['block_params'].append({
                                'channel_params': {
                                    'blue': {},
                                    'green': {},
                                    'red': {}
                                },
                                # 使用第一個通道的基本參數
                                'weights': block_param.get('weights', 'N/A'),
                                'EL': block_param.get('EL', 0),
                                'payload': 0,  # 會累加
                                'psnr': 0,     # 會平均
                                'ssim': 0,     # 會平均
                                'hist_corr': 0, # 會平均
                                'rotation': block_param.get('rotation', 0),
                                'prediction_method': prediction_method.value
                            })
                    
                    # 保存通道特定的區塊參數並累加/平均化指標
                    for j, block_param in enumerate(stage['block_params']):
                        if j < len(combined_stage['block_params']):
                            # 保存通道特定的參數
                            combined_stage['block_params'][j]['channel_params'][ch_name] = block_param
                            
                            # 累加payload
                            combined_stage['block_params'][j]['payload'] += block_param.get('payload', 0)
                            
                            # 累加指標（最後會除以3）
                            combined_stage['block_params'][j]['psnr'] += block_param.get('psnr', 0)
                            combined_stage['block_params'][j]['ssim'] += block_param.get('ssim', 0)
                            combined_stage['block_params'][j]['hist_corr'] += block_param.get('hist_corr', 0)
                            
                            # 在最後一個通道時計算平均值
                            if ch_idx == 2:  # 紅色通道（最後一個）
                                combined_stage['block_params'][j]['psnr'] /= 3
                                combined_stage['block_params'][j]['ssim'] /= 3
                                combined_stage['block_params'][j]['hist_corr'] /= 3
            
            # 🔧 更新：使用新的記憶體清理函數
            cleanup_memory()
            
        except Exception as e:
            print(f"Error processing {ch_name} channel: {str(e)}")
            print(f"Using original channel data for {ch_name}")
            # 如果某個通道處理失敗，使用原始通道
            final_channels.append(channel)
            continue
    
    # 🔧 重新組合彩色圖像
    if len(final_channels) == 3:
        final_color_img = combine_color_channels(final_channels[0], final_channels[1], final_channels[2])
        print(f"\nSuccessfully combined all three channels")
    else:
        print(f"Warning: Only {len(final_channels)} channels processed successfully")
        print("Using original image as fallback")
        final_color_img = img
    
    # 🔧 計算合併階段的整體指標
    pixel_count = img.shape[0] * img.shape[1]  # 只計算像素位置數，不包含通道數
    
    for stage in color_pee_stages:
        # 🔧 修正BPP計算：總payload除以像素位置數（不包含通道數）
        stage['bpp'] = stage['payload'] / pixel_count
        
        # 計算平均品質指標
        channel_metrics = stage['channel_metrics']
        stage['psnr'] = sum(channel_metrics[ch]['psnr'] for ch in channel_names) / 3
        stage['ssim'] = sum(channel_metrics[ch]['ssim'] for ch in channel_names) / 3
        stage['hist_corr'] = sum(channel_metrics[ch]['hist_corr'] for ch in channel_names) / 3
        
        # 🔧 合併階段圖像（如果所有通道都有的話）
        if 'channel_imgs' in stage and len(stage['channel_imgs']) == 3:
            stage['stage_img'] = combine_color_channels(
                stage['channel_imgs']['blue'],
                stage['channel_imgs']['green'], 
                stage['channel_imgs']['red']
            )
        
        # 輸出階段摘要
        print(f"\nColor Stage {stage['embedding']} summary:")
        print(f"  Total Payload: {stage['payload']} bits")
        print(f"  BPP: {stage['bpp']:.6f}")
        print(f"  Average PSNR: {stage['psnr']:.2f}")
        print(f"  Average SSIM: {stage['ssim']:.4f}")
        print(f"  Average Hist Corr: {stage['hist_corr']:.4f}")
        print("  Channel payloads:")
        for ch_name, payload in stage['channel_payloads'].items():
            print(f"    {ch_name.capitalize()}: {payload} bits")
    
    # 🔧 輸出最終結果摘要
    print(f"\n{'='*80}")
    print(f"Final Independent Channel Processing Results:")
    print(f"Image type: Color ({img.shape[0]}x{img.shape[1]}x{img.shape[2]})")
    print(f"Total Payload: {total_payload} bits")
    print(f"Total BPP: {total_payload / pixel_count:.6f}")
    
    if len(color_pee_stages) > 0:
        final_stage = color_pee_stages[-1]
        print(f"Final Average PSNR: {final_stage['psnr']:.2f}")
        print(f"Final Average SSIM: {final_stage['ssim']:.4f}")
        print(f"Final Average Hist Corr: {final_stage['hist_corr']:.4f}")
    
    print("Final channel payloads:")
    if len(color_pee_stages) > 0:
        final_payloads = color_pee_stages[-1]['channel_payloads']
        total_channel_bpp = 0
        for ch_name, payload in final_payloads.items():
            channel_bpp = payload / pixel_count
            total_channel_bpp += channel_bpp
            print(f"  {ch_name.capitalize()}: {payload} bits (BPP: {channel_bpp:.6f})")
        print(f"  Total BPP (sum of channels): {total_channel_bpp:.6f}")
    
    # 🔧 與等效灰階圖像的容量比較分析
    grayscale_equivalent_bpp = total_payload / pixel_count  # 使用相同的像素計數基準
    
    print(f"\nBPP Analysis:")
    print(f"  Pixel positions: {pixel_count}")
    print(f"  Total payload: {total_payload} bits")
    print(f"  Color image BPP: {total_payload / pixel_count:.6f}")
    print(f"  This represents ~{total_payload / pixel_count:.1f} bits per pixel position across all channels")
    print(f"  Compared to equivalent grayscale: {(total_payload / pixel_count):.2f}x higher capacity potential")
    print(f"{'='*80}")
    
    return final_color_img, int(total_payload), color_pee_stages