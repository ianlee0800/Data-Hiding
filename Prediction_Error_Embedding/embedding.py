from numba.core.errors import NumbaPerformanceWarning
import warnings
import numpy as np
import cupy as cp
from image_processing import (
    split_image_flexible,
    merge_image_flexible
)
from utils import (
    generate_random_binary_array,
    generate_embedding_data
)

from common import *
from pee import *

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

    # 初始化處理
    original_img = cp.asarray(img)
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
            'original_img': cp.asnumpy(original_img)  # 新增：保存原始圖像
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
                # print(f"Target reached. Copying remaining sub-images without embedding.")
                embedded_sub_images.append(cp.asarray(sub_img))
                continue
            
            sub_img = cp.asarray(sub_img)
            sub_data = stage_data['sub_data'][i]
            sub_data = cp.asarray(sub_data, dtype=cp.uint8)
            
            # 計算當前子圖像的嵌入目標
            if remaining_target is not None:
                current_target = min(len(sub_data), remaining_target)
                # print(f"Sub-image {i} target: {current_target} bits")
            else:
                current_target = None
            
            # 根據 el_mode 決定 max_el
            if el_mode == 1:  # Increasing
                max_el = 3 + embedding * 2
            elif el_mode == 2:  # Decreasing
                max_el = 11 - embedding * 2
            else:  # No restriction
                max_el = 7
            
            # 計算自適應嵌入層級
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
            
            # 執行數據嵌入 - 注意返回值增加了預測圖像
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
                stage_info['sample_original_sub'] = cp.asnumpy(sub_img)
                stage_info['sample_pred_sub'] = pred_sub
                stage_info['sample_embedded_sub'] = cp.asnumpy(embedded_sub)
            
            # 更新剩餘目標量
            if remaining_target is not None:
                payload = min(payload, current_target)
                remaining_target -= payload
                # print(f"Sub-image {i} embedded {payload} bits, remaining: {remaining_target}")
            
            embedded_sub_images.append(embedded_sub)
            stage_payload += payload
            
            # 計算品質指標
            sub_img_np = cp.asnumpy(sub_img)
            embedded_sub_np = cp.asnumpy(embedded_sub)
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
                'EL': int(to_numpy(local_el).max()),
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
                pred_img_merged = merge_image_flexible([cp.asarray(p) for p in all_sub_preds], 
                                                     split_size, block_base=True)
                stage_info['pred_img'] = cp.asnumpy(pred_img_merged)
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
                    stage_info['pred_img'] = cp.asnumpy(cp.rot90(cp.asarray(stage_info['pred_img']), 
                                                              k=-stage_rotation // 90))
                else:
                    stage_info['pred_img'] = np.rot90(stage_info['pred_img'], 
                                                     k=-stage_rotation // 90)
        
        stage_info['stage_img'] = stage_img
        
        # 計算階段整體品質指標
        stage_img_np = cp.asnumpy(stage_img)
        original_img_np = cp.asnumpy(original_img)
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
    final_pee_img = cp.asnumpy(current_img)
    return final_pee_img, int(total_payload), pee_stages

def pee_process_color_image_rotation_cuda(img, total_embeddings, ratio_of_ones, use_different_weights, 
                                       split_size, el_mode, prediction_method=None,
                                       target_payload_size=-1):
    """
    Process a color image using rotation PEE method - 完全重寫版本，確保與灰階版本功能一致
    
    This function splits a color image into its RGB channels, processes each channel
    independently using the existing rotation PEE method, and then recombines the
    channels into a final color image.
    
    Parameters:
    -----------
    Same as pee_process_with_rotation_cuda, but img is now a color image
        
    Returns:
    --------
    tuple
        (final_color_img, total_payload, color_pee_stages)
    """
    import os
    import cv2
    import numpy as np
    import cupy as cp
    from color import split_color_channels, combine_color_channels
    from common import cleanup_memory, calculate_psnr, calculate_ssim, histogram_correlation
    
    if prediction_method is None:
        # Import PredictionMethod if not provided to maintain compatibility
        from image_processing import PredictionMethod
        prediction_method = PredictionMethod.PROPOSED
    
    print(f"Processing color image with {prediction_method.value} prediction method")
    
    # Split color image into channels
    b_channel, g_channel, r_channel = split_color_channels(img)
    
    # Track total payload across all channels
    total_payload = 0
    
    color_pee_stages = []
    
    # Calculate target payload per channel
    channel_target = target_payload_size // 3 if target_payload_size > 0 else -1
    
    # Process each channel separately
    print("\nProcessing blue channel...")
    final_b_img, b_payload, b_stages = pee_process_with_rotation_cuda(
        b_channel, total_embeddings, ratio_of_ones, use_different_weights,
        split_size, el_mode, prediction_method=prediction_method,
        target_payload_size=channel_target
    )
    total_payload += b_payload
    
    # Clean GPU memory between channel processing
    cleanup_memory()
    
    print("\nProcessing green channel...")
    final_g_img, g_payload, g_stages = pee_process_with_rotation_cuda(
        g_channel, total_embeddings, ratio_of_ones, use_different_weights,
        split_size, el_mode, prediction_method=prediction_method,
        target_payload_size=channel_target
    )
    total_payload += g_payload
    
    # Clean GPU memory between channel processing
    cleanup_memory()
    
    print("\nProcessing red channel...")
    final_r_img, r_payload, r_stages = pee_process_with_rotation_cuda(
        r_channel, total_embeddings, ratio_of_ones, use_different_weights,
        split_size, el_mode, prediction_method=prediction_method,
        target_payload_size=channel_target
    )
    total_payload += r_payload
    
    # Combine channels back into a color image
    final_color_img = combine_color_channels(final_b_img, final_g_img, final_r_img)
    
    # Create combined stages information for all 3 channels - 完全重寫以確保一致性
    for i in range(min(len(b_stages), len(g_stages), len(r_stages))):
        # Get stage info from each channel
        b_stage = b_stages[i]
        g_stage = g_stages[i]
        r_stage = r_stages[i]
        
        # Initialize combined stage info with all necessary fields
        combined_stage = {
            'embedding': b_stage['embedding'],  # All should be the same
            'payload': b_stage['payload'] + g_stage['payload'] + r_stage['payload'],
            'channel_payloads': {
                'blue': b_stage['payload'],
                'green': g_stage['payload'],
                'red': r_stage['payload']
            },
            'bpp': (b_stage['bpp'] + g_stage['bpp'] + r_stage['bpp']) / 3,  # Average BPP
            'channel_metrics': {
                'blue': {'psnr': b_stage['psnr'], 'ssim': b_stage['ssim'], 'hist_corr': b_stage['hist_corr']},
                'green': {'psnr': g_stage['psnr'], 'ssim': g_stage['ssim'], 'hist_corr': g_stage['hist_corr']},
                'red': {'psnr': r_stage['psnr'], 'ssim': r_stage['ssim'], 'hist_corr': r_stage['hist_corr']}
            },
            'prediction_method': prediction_method.value,
            'rotation': b_stage.get('rotation', i * 90)  # All channels should have the same rotation
        }
        
        # 添加原始圖像資訊
        combined_stage['original_img'] = img
        
        # Combine stage images if available
        if 'stage_img' in b_stage and 'stage_img' in g_stage and 'stage_img' in r_stage:
            b_stage_img = cp.asnumpy(b_stage['stage_img']) if isinstance(b_stage['stage_img'], cp.ndarray) else b_stage['stage_img']
            g_stage_img = cp.asnumpy(g_stage['stage_img']) if isinstance(g_stage['stage_img'], cp.ndarray) else g_stage['stage_img']
            r_stage_img = cp.asnumpy(r_stage['stage_img']) if isinstance(r_stage['stage_img'], cp.ndarray) else r_stage['stage_img']
            
            combined_stage['stage_img'] = combine_color_channels(b_stage_img, g_stage_img, r_stage_img)
        
        # 組合旋轉圖像（如果有）
        if 'rotated_img' in b_stage and 'rotated_img' in g_stage and 'rotated_img' in r_stage:
            try:
                b_rotated = cp.asnumpy(b_stage['rotated_img']) if isinstance(b_stage['rotated_img'], cp.ndarray) else b_stage['rotated_img']
                g_rotated = cp.asnumpy(g_stage['rotated_img']) if isinstance(g_stage['rotated_img'], cp.ndarray) else g_stage['rotated_img']
                r_rotated = cp.asnumpy(r_stage['rotated_img']) if isinstance(r_stage['rotated_img'], cp.ndarray) else r_stage['rotated_img']
                
                combined_stage['rotated_img'] = combine_color_channels(b_rotated, g_rotated, r_rotated)
            except Exception as e:
                print(f"Warning: Could not combine rotated images: {e}")
        
        # 組合預測圖像（如果有）
        if 'pred_img' in b_stage and 'pred_img' in g_stage and 'pred_img' in r_stage:
            try:
                b_pred = b_stage['pred_img'] if isinstance(b_stage['pred_img'], np.ndarray) else np.array(b_stage['pred_img'])
                g_pred = g_stage['pred_img'] if isinstance(g_stage['pred_img'], np.ndarray) else np.array(g_stage['pred_img'])
                r_pred = r_stage['pred_img'] if isinstance(r_stage['pred_img'], np.ndarray) else np.array(r_stage['pred_img'])
                
                combined_stage['pred_img'] = combine_color_channels(b_pred, g_pred, r_pred)
            except Exception as e:
                print(f"Warning: Could not combine prediction images: {e}")
        
        # 組合子圖像樣本（如果有）
        if all(key in b_stage for key in ['sample_original_sub', 'sample_pred_sub', 'sample_embedded_sub']):
            combined_stage['sample_original_sub'] = {
                'blue': b_stage['sample_original_sub'],
                'green': g_stage['sample_original_sub'],
                'red': r_stage['sample_original_sub']
            }
            combined_stage['sample_pred_sub'] = {
                'blue': b_stage['sample_pred_sub'],
                'green': g_stage['sample_pred_sub'],
                'red': r_stage['sample_pred_sub']
            }
            combined_stage['sample_embedded_sub'] = {
                'blue': b_stage['sample_embedded_sub'],
                'green': g_stage['sample_embedded_sub'],
                'red': r_stage['sample_embedded_sub']
            }
        
        # Calculate combined metrics directly from the combined image
        if 'stage_img' in combined_stage:
            psnr = (b_stage['psnr'] + g_stage['psnr'] + r_stage['psnr']) / 3
            ssim = (b_stage['ssim'] + g_stage['ssim'] + r_stage['ssim']) / 3
            hist_corr = (b_stage['hist_corr'] + g_stage['hist_corr'] + r_stage['hist_corr']) / 3
            
            combined_stage['psnr'] = psnr
            combined_stage['ssim'] = ssim
            combined_stage['hist_corr'] = hist_corr
            
        # Save combined sub_images information if available (nested by channel)
        if 'sub_images' in b_stage and 'sub_images' in g_stage and 'sub_images' in r_stage:
            combined_stage['sub_images'] = {
                'blue': b_stage['sub_images'],
                'green': g_stage['sub_images'],
                'red': r_stage['sub_images']
            }
            
        # Block params need special handling to ensure compatibility with create_pee_info_table
        if 'block_params' in b_stage and 'block_params' in g_stage and 'block_params' in r_stage:
            combined_stage['block_params'] = []
            
            # We'll take as many block params as available in all three channels
            for j in range(min(len(b_stage['block_params']), len(g_stage['block_params']), len(r_stage['block_params']))):
                # Get block parameters from each channel
                b_block = b_stage['block_params'][j]
                g_block = g_stage['block_params'][j]
                r_block = r_stage['block_params'][j]
                
                # Create a combined block that has both the nested channel data and
                # flattened keys that create_pee_info_table expects
                combined_block = {
                    'channel_params': {
                        'blue': b_block,
                        'green': g_block,
                        'red': r_block
                    },
                    # Include the keys that create_pee_info_table expects at the top level
                    'weights': b_block.get('weights', 'N/A'),  # Take weights from blue channel
                    'EL': b_block.get('EL', 0),                # Take EL from blue channel
                    'payload': (b_block.get('payload', 0) + 
                               g_block.get('payload', 0) + 
                               r_block.get('payload', 0)),
                    'psnr': (b_block.get('psnr', 0) + 
                            g_block.get('psnr', 0) + 
                            r_block.get('psnr', 0)) / 3,
                    'ssim': (b_block.get('ssim', 0) + 
                            g_block.get('ssim', 0) + 
                            r_block.get('ssim', 0)) / 3,
                    'hist_corr': (b_block.get('hist_corr', 0) + 
                                 g_block.get('hist_corr', 0) + 
                                 r_block.get('hist_corr', 0)) / 3,
                    'rotation': b_block.get('rotation', 0),     # All channels have same rotation
                    'prediction_method': prediction_method.value
                }
                combined_stage['block_params'].append(combined_block)
            
        # Add stage to combined stages
        color_pee_stages.append(combined_stage)
        
        # 輸出階段摘要
        print(f"\nColor Embedding {i} summary:")
        print(f"Prediction Method: {prediction_method.value}")
        print(f"Total Payload: {combined_stage['payload']}")
        print(f"BPP: {combined_stage['bpp']:.4f}")
        print(f"PSNR: {combined_stage['psnr']:.2f}")
        print(f"SSIM: {combined_stage['ssim']:.4f}")
        print(f"Hist Corr: {combined_stage['hist_corr']:.4f}")
        print(f"Rotation: {combined_stage['rotation']}°")
        print("Channel payloads:")
        for channel, payload in combined_stage['channel_payloads'].items():
            print(f"  {channel.capitalize()}: {payload}")
    
    return final_color_img, total_payload, color_pee_stages

def pee_process_with_split_cuda(img, total_embeddings, ratio_of_ones, use_different_weights, 
                              split_size, el_mode, block_base, 
                              prediction_method=None,
                              target_payload_size=-1):
    """
    Using split PEE method with support for both grayscale and color images
    Enhanced with rotation effect tracking for visualization
    
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
        return pee_process_color_image_split_cuda(
            img, total_embeddings, ratio_of_ones, use_different_weights,
            split_size, el_mode, block_base, prediction_method, target_payload_size
        )

    # 將輸入圖像轉換為 CUDA 陣列
    original_img = cp.asarray(img)
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
            'rotations': cp.asnumpy(stage_rotations).tolist(),  # 轉換為可序列化格式
            'original_img': cp.asnumpy(original_img)  # 保存原始圖像
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
                # print(f"Target reached. Copying remaining sub-images without embedding.")
                embedded_sub_images.append(cp.asarray(sub_img))
                # 💡 對於未嵌入的子圖像，也保存旋轉版本以保持一致性
                rotation = int(stage_rotations[i])
                rotated_sub_img = cp.rot90(cp.asarray(sub_img), k=rotation // 90)
                rotated_embedded_sub_images.append(rotated_sub_img)
                continue
            
            # 準備子圖像處理
            sub_img = cp.asarray(sub_img)
            rotation = int(stage_rotations[i])
            rotated_sub_img = cp.rot90(sub_img, k=rotation // 90)
            
            # 準備嵌入數據
            sub_data = stage_data['sub_data'][i]
            sub_data = cp.asarray(sub_data, dtype=cp.uint8)
            
            # 計算當前子圖像的嵌入目標
            if remaining_target is not None:
                current_target = min(len(sub_data), remaining_target)
                # print(f"Sub-image {i} target: {current_target} bits")
            else:
                current_target = None
            
            # 根據 el_mode 決定 max_el
            if el_mode == 1:
                max_el = 3 + embedding * 2
            elif el_mode == 2:
                max_el = 11 - embedding * 2
            else:
                max_el = 7
            
            # 計算自適應嵌入層級
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
            
            # 執行數據嵌入 - 注意返回值增加了預測圖像
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
                stage_info['sample_original_sub'] = cp.asnumpy(sub_img)
                stage_info['sample_pred_sub'] = pred_sub
                stage_info['sample_embedded_sub'] = cp.asnumpy(embedded_sub)
                stage_info['sample_rotated_embedded_sub'] = cp.asnumpy(embedded_sub)  # 新增旋轉版本
            
            # 更新剩餘目標量
            if remaining_target is not None:
                payload = min(payload, current_target)
                remaining_target -= payload
                # print(f"Sub-image {i} embedded {payload} bits, remaining: {remaining_target}")
            
            # 將嵌入後的圖像旋轉回原始方向（原有邏輯）
            rotated_back_sub = cp.rot90(embedded_sub, k=-rotation // 90)
            embedded_sub_images.append(rotated_back_sub)
            stage_payload += payload
            
            # 計算品質指標
            sub_img_np = cp.asnumpy(sub_img)
            rotated_back_sub_np = cp.asnumpy(rotated_back_sub)
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
                'EL': int(to_numpy(local_el).max()),
                'payload': int(payload),
                'psnr': float(sub_psnr),
                'ssim': float(sub_ssim),
                'rotation': rotation,
                'hist_corr': float(sub_hist_corr),
                'prediction_method': prediction_method.value,
                'original_img': sub_img_np,         # 新增
                'pred_img': pred_sub,               # 新增
                'embedded_img': rotated_back_sub_np,     # 新增
                'rotated_embedded_img': cp.asnumpy(embedded_sub)  # 💡 新增：旋轉版本
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
                pred_img_merged = merge_image_flexible([cp.asarray(p) for p in all_sub_preds], 
                                                     split_size, block_base)
                stage_info['pred_img'] = cp.asnumpy(pred_img_merged)
            except Exception as e:
                print(f"Warning: Could not merge prediction images: {e}")
                # 如果不能合併，使用第一個子圖像的預測作為示例
                if 'sample_pred_sub' in stage_info:
                    stage_info['pred_img'] = stage_info['sample_pred_sub']
        
        stage_info['stage_img'] = stage_img
        
        # 計算階段整體品質指標
        stage_img_np = cp.asnumpy(stage_img)
        original_img_np = cp.asnumpy(original_img)
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
    final_pee_img = cp.asnumpy(current_img)
    return final_pee_img, int(total_payload), pee_stages

def pee_process_color_image_split_cuda(img, total_embeddings, ratio_of_ones, use_different_weights, 
                                     split_size, el_mode, block_base, 
                                     prediction_method=None,
                                     target_payload_size=-1):
    """
    Process a color image using split PEE method with proper rotation synchronization and enhanced tracking.
    
    This function implements the split PEE method for color images by:
    1. Splitting the color image into sub-images
    2. Applying synchronized rotation to all channels of each sub-image
    3. Processing each channel independently with PEE
    4. Rotating back and recombining the results
    5. Tracking rotation effects for visualization
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input color image (BGR format)
    total_embeddings : int
        Total number of embedding stages
    ratio_of_ones : float
        Ratio of ones in embedded data
    use_different_weights : bool
        Whether to use different weights for each sub-image
    split_size : int
        Size of split (e.g., 2 for 2x2 split)
    el_mode : int
        EL mode (0: no restriction, 1: increasing, 2: decreasing)
    block_base : bool
        True for block-based split, False for quarter-based split
    prediction_method : PredictionMethod, optional
        Prediction method to use
    target_payload_size : int, optional
        Target payload size (-1 for maximum capacity)
        
    Returns:
    --------
    tuple
        (final_color_img, total_payload, color_pee_stages)
    """
    import os
    import cv2
    import numpy as np
    import cupy as cp
    from color import split_color_channels, combine_color_channels
    from common import cleanup_memory, calculate_psnr, calculate_ssim, histogram_correlation
    from image_processing import split_image_flexible, merge_image_flexible, PredictionMethod
    from pee import multi_pass_embedding, brute_force_weight_search_cuda
    from utils import generate_embedding_data
    
    if prediction_method is None:
        prediction_method = PredictionMethod.PROPOSED
    
    print(f"Processing color image with split method using {prediction_method.value} prediction")
    
    # 初始化處理
    original_img = img.copy()
    height, width = img.shape[:2]
    total_pixels = height * width
    pee_stages = []
    total_payload = 0
    current_img = original_img.copy()
    previous_psnr = float('inf')
    previous_ssim = 1.0
    
    # 分離彩色通道
    b_channel, g_channel, r_channel = split_color_channels(current_img)
    current_channels = [b_channel, g_channel, r_channel]
    channel_names = ['blue', 'green', 'red']
    
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
        print(f"\nStarting embedding {embedding} for color image")
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
        
        # 💡 新增：統一的旋轉角度（確保三通道同步）
        unified_rotations = cp.random.choice([0, 90, 180, 270], 
                                           size=split_size * split_size)
        print(f"Unified rotation angles for this stage: {cp.asnumpy(unified_rotations)}")
        
        # 💡 新增：追蹤各通道的旋轉效果
        channel_rotated_sub_images = {
            'blue': [],
            'green': [],
            'red': []
        }
        
        # 初始化階段資訊
        stage_info = {
            'embedding': embedding,
            'payload': 0,
            'psnr': 0,
            'ssim': 0,
            'hist_corr': 0,
            'bpp': 0,
            'block_params': [],
            'channel_payloads': {'blue': 0, 'green': 0, 'red': 0},
            'channel_metrics': {
                'blue': {'psnr': 0, 'ssim': 0, 'hist_corr': 0},
                'green': {'psnr': 0, 'ssim': 0, 'hist_corr': 0},
                'red': {'psnr': 0, 'ssim': 0, 'hist_corr': 0}
            },
            'prediction_method': prediction_method.value,
            # 💡 新增：為階段資訊添加彩色相關欄位
            'split_size': split_size,
            'block_base': block_base,
            'rotations': cp.asnumpy(unified_rotations).tolist(),
            'original_img': original_img  # 保存原始彩色圖像
        }
        
        stage_payload = 0
        
        # 處理每個通道
        embedded_channels = []
        channel_stages = []
        
        for ch_idx, (channel, ch_name) in enumerate(zip(current_channels, channel_names)):
            print(f"\nProcessing {ch_name} channel...")
            
            # 確保通道是cupy數組
            channel = cp.asarray(channel)
            
            # 使用彈性分割函數切割通道
            sub_images = split_image_flexible(channel, split_size, block_base)
            embedded_sub_images = []
            channel_payload = 0
            
            # 💡 新增：該通道的旋轉子圖像列表
            channel_rotated_subs = []  # 該通道的旋轉子圖像
            
            # 處理每個子圖像
            for i, sub_img in enumerate(sub_images):
                # 檢查是否已達到目標payload
                if remaining_target is not None and remaining_target <= 0:
                    embedded_sub_images.append(cp.asarray(sub_img))
                    # 💡 對於未嵌入的子圖像，也保存旋轉版本以保持一致性
                    rotation = int(unified_rotations[i])
                    rotated_sub_img = cp.rot90(cp.asarray(sub_img), k=rotation // 90)
                    channel_rotated_subs.append(rotated_sub_img)
                    continue
                
                # 準備子圖像處理
                sub_img = cp.asarray(sub_img)
                
                # ===== 關鍵改進：使用統一的旋轉角度 =====
                rotation = int(unified_rotations[i])  # 所有通道在相同位置使用相同旋轉
                rotated_sub_img = cp.rot90(sub_img, k=rotation // 90)
                
                # 準備嵌入數據 - 每個通道分配總數據的1/3
                sub_data = stage_data['sub_data'][i]
                channel_data_size = len(sub_data) // 3
                channel_start = ch_idx * channel_data_size
                channel_end = channel_start + channel_data_size
                
                # 為最後一個通道分配剩餘的數據
                if ch_idx == 2:  # red channel
                    channel_end = len(sub_data)
                
                channel_sub_data = sub_data[channel_start:channel_end]
                channel_sub_data = cp.asarray(channel_sub_data, dtype=cp.uint8)
                
                # 計算當前子圖像的嵌入目標
                if remaining_target is not None:
                    current_target = min(len(channel_sub_data), remaining_target // 3)
                else:
                    current_target = None
                
                # 根據 el_mode 決定 max_el
                if el_mode == 1:  # Increasing
                    max_el = 3 + embedding * 2
                elif el_mode == 2:  # Decreasing
                    max_el = 11 - embedding * 2
                else:  # No restriction
                    max_el = 7
                
                # 計算自適應嵌入層級
                local_el = compute_improved_adaptive_el(
                    rotated_sub_img, 
                    window_size=5, 
                    max_el=max_el
                )
                
                # 根據預測方法進行不同的處理
                if prediction_method == PredictionMethod.PROPOSED:
                    # 如果是第一個通道或使用不同權重，計算權重
                    if ch_idx == 0 or use_different_weights:
                        weights, (sub_payload, sub_psnr) = brute_force_weight_search_cuda(
                            rotated_sub_img, channel_sub_data, local_el, 
                            target_bpp, target_psnr, embedding
                        )
                    # 否則使用與藍色通道相同的權重（如果已計算過）
                    elif 'weights' in locals():
                        pass  # 使用之前計算的權重
                    else:
                        # 如果沒有之前的權重，重新計算
                        weights, (sub_payload, sub_psnr) = brute_force_weight_search_cuda(
                            rotated_sub_img, channel_sub_data, local_el, 
                            target_bpp, target_psnr, embedding
                        )
                else:
                    # MED 和 GAP 方法不需要權重
                    weights = None
                
                # 執行數據嵌入
                embedded_sub, payload, pred_sub = multi_pass_embedding(
                    rotated_sub_img,
                    channel_sub_data,
                    local_el,
                    weights,
                    embedding,
                    prediction_method=prediction_method,
                    remaining_target=[current_target] if current_target else None
                )
                
                # 💡 關鍵修改：保存該通道的旋轉結果
                channel_rotated_subs.append(embedded_sub)
                
                # 更新剩餘目標量
                if remaining_target is not None:
                    actual_payload = min(payload, current_target) if current_target else payload
                    remaining_target -= actual_payload
                    payload = actual_payload
                
                # ===== 關鍵改進：使用相同的旋轉角度旋轉回來 =====
                rotated_back_sub = cp.rot90(embedded_sub, k=-rotation // 90)
                embedded_sub_images.append(rotated_back_sub)
                channel_payload += payload
                
                # 計算品質指標
                sub_img_np = cp.asnumpy(sub_img)
                rotated_back_sub_np = cp.asnumpy(rotated_back_sub)
                sub_psnr = calculate_psnr(sub_img_np, rotated_back_sub_np)
                sub_ssim = calculate_ssim(sub_img_np, rotated_back_sub_np)
                sub_hist_corr = histogram_correlation(
                    np.histogram(sub_img_np, bins=256, range=(0, 255))[0],
                    np.histogram(rotated_back_sub_np, bins=256, range=(0, 255))[0]
                )
                
                # 記錄區塊資訊（只在藍色通道記錄，避免重複）
                if ch_idx == 0:  # 只在藍色通道記錄統一資訊
                    block_info = {
                        'channel': ch_name,
                        'weights': (weights.tolist() if weights is not None and hasattr(weights, 'tolist') 
                                  else "N/A" if prediction_method in [PredictionMethod.MED, PredictionMethod.GAP] 
                                  else None),
                        'EL': int(cp.asnumpy(local_el).max()),
                        'payload': int(payload),
                        'psnr': float(sub_psnr),
                        'ssim': float(sub_ssim),
                        'rotation': rotation,
                        'hist_corr': float(sub_hist_corr),
                        'prediction_method': prediction_method.value,
                        'original_img': sub_img_np,
                        'pred_img': pred_sub,
                        'embedded_img': rotated_back_sub_np,
                        'rotated_embedded_img': cp.asnumpy(embedded_sub)  # 💡 新增：旋轉版本
                    }
                    stage_info['block_params'].append(block_info)
            
            # 💡 新增：保存該通道的旋轉子圖像
            channel_rotated_sub_images[ch_name] = channel_rotated_subs
            
            # 合併處理後的子圖像
            channel_img = merge_image_flexible(embedded_sub_images, split_size, block_base)
            embedded_channels.append(channel_img)
            
            # 計算通道品質指標
            channel_img_np = cp.asnumpy(channel_img)
            original_channel_np = cp.asnumpy(current_channels[ch_idx])
            channel_psnr = calculate_psnr(original_channel_np, channel_img_np)
            channel_ssim = calculate_ssim(original_channel_np, channel_img_np)
            channel_hist_corr = histogram_correlation(
                np.histogram(original_channel_np, bins=256, range=(0, 255))[0],
                np.histogram(channel_img_np, bins=256, range=(0, 255))[0]
            )
            
            # 更新通道資訊
            stage_info['channel_payloads'][ch_name] = channel_payload
            stage_info['channel_metrics'][ch_name] = {
                'psnr': float(channel_psnr),
                'ssim': float(channel_ssim),
                'hist_corr': float(channel_hist_corr)
            }
            
            stage_payload += channel_payload
            
            print(f"{ch_name.capitalize()} channel processed:")
            print(f"  Payload: {channel_payload}")
            print(f"  PSNR: {channel_psnr:.2f}")
            print(f"  SSIM: {channel_ssim:.4f}")
        
        # 💡 新增：保存彩色旋轉效果到階段資訊
        stage_info['channel_rotated_sub_images'] = channel_rotated_sub_images
        
        # 💡 新增：嘗試合併各通道的旋轉效果圖像
        try:
            rotated_merged_channels = []
            for ch_name in channel_names:
                if channel_rotated_sub_images[ch_name]:
                    rotated_channel_img = merge_image_flexible(
                        channel_rotated_sub_images[ch_name], 
                        split_size, 
                        block_base
                    )
                    rotated_merged_channels.append(cp.asnumpy(rotated_channel_img))
            
            if len(rotated_merged_channels) == 3:
                # 合併三個通道形成彩色旋轉效果圖像
                rotated_color_img = combine_color_channels(
                    rotated_merged_channels[0],  # blue
                    rotated_merged_channels[1],  # green  
                    rotated_merged_channels[2]   # red
                )
                stage_info['rotated_stage_img'] = rotated_color_img
                
                # 也分別保存各通道的旋轉效果
                stage_info['rotated_channel_imgs'] = {
                    'blue': rotated_merged_channels[0],
                    'green': rotated_merged_channels[1],
                    'red': rotated_merged_channels[2]
                }
                
        except Exception as e:
            print(f"Warning: Could not merge rotated channel images: {e}")
        
        # 重新組合彩色圖像
        stage_img = combine_color_channels(
            cp.asnumpy(embedded_channels[0]),  # blue
            cp.asnumpy(embedded_channels[1]),  # green
            cp.asnumpy(embedded_channels[2])   # red
        )
        
        stage_info['stage_img'] = stage_img
        
        # 計算整體品質指標
        overall_psnr = sum(stage_info['channel_metrics'][ch]['psnr'] for ch in channel_names) / 3
        overall_ssim = sum(stage_info['channel_metrics'][ch]['ssim'] for ch in channel_names) / 3
        overall_hist_corr = sum(stage_info['channel_metrics'][ch]['hist_corr'] for ch in channel_names) / 3
        
        stage_info['psnr'] = float(overall_psnr)
        stage_info['ssim'] = float(overall_ssim) 
        stage_info['hist_corr'] = float(overall_hist_corr)
        stage_info['payload'] = stage_payload
        stage_info['bpp'] = float(stage_info['payload'] / total_pixels)
        
        # 輸出階段摘要
        print(f"\nColor Embedding {embedding} summary:")
        print(f"Prediction Method: {prediction_method.value}")
        print(f"Total Payload: {stage_info['payload']}")
        print(f"BPP: {stage_info['bpp']:.4f}")
        print(f"Overall PSNR: {stage_info['psnr']:.2f}")
        print(f"Overall SSIM: {stage_info['ssim']:.4f}")
        print(f"Overall Hist Corr: {stage_info['hist_corr']:.4f}")
        print(f"Split info: {split_size}x{split_size}, {'Block-based' if block_base else 'Quarter-based'}")
        print("Channel payloads:")
        for ch_name, payload in stage_info['channel_payloads'].items():
            print(f"  {ch_name.capitalize()}: {payload}")
        
        # 更新資訊
        pee_stages.append(stage_info)
        total_payload += stage_payload
        previous_psnr = stage_info['psnr']
        previous_ssim = stage_info['ssim']
        
        # 更新當前圖像和通道
        current_img = stage_img
        current_channels = [
            cp.asnumpy(embedded_channels[0]),
            cp.asnumpy(embedded_channels[1]), 
            cp.asnumpy(embedded_channels[2])
        ]
        
        # 檢查是否已達到總目標
        if remaining_target is not None and remaining_target <= 0:
            print(f"Reached target payload ({target_payload_size} bits) at stage {embedding}")
            break
        
        # 清理記憶體
        cleanup_memory()

    # 返回最終結果
    final_color_img = current_img
    return final_color_img, int(total_payload), pee_stages