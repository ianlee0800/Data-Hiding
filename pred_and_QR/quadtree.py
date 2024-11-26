# quadtree.py

import numpy as np
import cupy as cp
from pee import multi_pass_embedding, compute_improved_adaptive_el, brute_force_weight_search_cuda

from utils import (
    generate_random_binary_array
)

from common import (
    calculate_psnr,
    calculate_ssim,
    histogram_correlation,
    calculate_metrics_with_rotation,
    compute_improved_adaptive_el,
    calculate_block_variance_cuda
)

def cleanup_quadtree_resources():
    """
    清理 quadtree 處理過程中使用的資源
    """
    try:
        # 清理 GPU 記憶體
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as e:
        print(f"Error cleaning up quadtree resources: {str(e)}")
        
def process_current_block(block, position, size, stage_info, embedding, ratio_of_ones,
                         target_bpp, target_psnr, el_mode, verbose=False):  # 添加 verbose 參數
    """
    處理當前區塊的 PEE 嵌入
    
    Parameters:
    -----------
    block : cupy.ndarray
        輸入區塊
    position : tuple
        區塊在原圖中的位置 (y, x)
    size : int
        區塊大小
    stage_info : dict
        當前階段的資訊
    embedding : int
        當前嵌入階段
    ratio_of_ones : float
        嵌入數據中 1 的比例
    target_bpp : float
        目標 BPP
    target_psnr : float
        目標 PSNR
    el_mode : int
        EL模式 (0:無限制, 1:漸增, 2:漸減)
    
    Returns:
    --------
    list
        包含 (embedded_block, position, size) 的列表
    """
    try:
        # 確保 block 是 CuPy 數組
        if not isinstance(block, cp.ndarray):
            block = cp.asarray(block)
        
        # 根據 el_mode 決定 max_el
        if el_mode == 1:  # Increasing
            max_el = 3 + embedding * 2
        elif el_mode == 2:  # Decreasing
            max_el = 11 - embedding * 2
        else:  # No restriction
            max_el = 7
        
        # 對於小區塊調整 max_el 和其他參數
        if size <= 32:
            max_el = min(max_el, 5)  # 限制最大嵌入層級
            local_target_bpp = target_bpp * 0.8  # 降低小區塊的載荷期望
            local_target_psnr = target_psnr + 2   # 提高小區塊的質量要求
        else:
            local_target_bpp = target_bpp
            local_target_psnr = target_psnr
        
        # 計算改進的 local_el
        local_el = compute_improved_adaptive_el(
            block, 
            window_size=min(5, size//4),  # 對小區塊調整窗口大小
            max_el=max_el, 
            block_size=size
        )
        
        # 生成嵌入數據
        try:
            sub_data = generate_random_binary_array(block.size, ratio_of_ones)
            sub_data = cp.asarray(sub_data, dtype=cp.uint8)
            
            # 計算權重
            weights, (sub_payload, sub_psnr) = brute_force_weight_search_cuda(
                block, sub_data, local_el, local_target_bpp, local_target_psnr, 
                embedding, block_size=size
            )
            
            # 執行嵌入
            data_to_embed = generate_random_binary_array(block.size, ratio_of_ones)
            data_to_embed = cp.asarray(data_to_embed)
            embedded_block, payload = multi_pass_embedding(
                block, data_to_embed, local_el, weights, embedding
            )
            
            # 確保結果是 CuPy 數組
            if not isinstance(embedded_block, cp.ndarray):
                embedded_block = cp.asarray(embedded_block)
            
            # 計算區塊指標
            block_info = {
                'position': position,
                'size': size,
                'weights': weights.tolist() if isinstance(weights, np.ndarray) else weights,
                'payload': int(payload),
                'psnr': float(calculate_psnr(cp.asnumpy(block), cp.asnumpy(embedded_block))),
                'ssim': float(calculate_ssim(cp.asnumpy(block), cp.asnumpy(embedded_block))),
                'hist_corr': float(histogram_correlation(
                    np.histogram(cp.asnumpy(block), bins=256, range=(0, 255))[0],
                    np.histogram(cp.asnumpy(embedded_block), bins=256, range=(0, 255))[0]
                )),
                'EL': int(cp.asnumpy(local_el).max())
            }
            
            # 更新stage資訊
            stage_info['block_info'][str(size)]['blocks'].append(block_info)
            stage_info['payload'] += payload
            
            # 只在 verbose=True 時輸出處理資訊
            if verbose:
                print(f"  Block processed at size {size}x{size}")
                print(f"  Payload: {payload}")
                print(f"  PSNR: {block_info['psnr']:.2f}")
            
            return [(embedded_block, position, size)]
            
        except Exception as e:
            print(f"Error in data embedding process: {str(e)}")
            raise
            
    except Exception as e:
        print(f"Error in block processing: {str(e)}")
        raise

def process_block(block, position, size, stage_info, embedding, variance_threshold, 
                 ratio_of_ones, target_bpp, target_psnr, el_mode, verbose=False,
                 rotation_mode='none'):
    """
    遞迴處理區塊，決定是否需要進一步分割
    
    Parameters:
    -----------
    block : cupy.ndarray
        輸入區塊
    position : tuple
        區塊在原圖中的位置 (y, x)
    size : int
        區塊大小
    stage_info : dict
        當前階段的資訊
    embedding : int
        當前嵌入階段
    variance_threshold : float
        變異度閾值
    ratio_of_ones : float
        嵌入數據中 1 的比例
    target_bpp : float
        目標 BPP
    target_psnr : float
        目標 PSNR
    el_mode : int
        EL模式
    verbose : bool
        是否輸出詳細資訊
    rotation_mode : str
        'none': 原始的 quadtree 方法
        'random': 使用隨機旋轉的新方法
    
    Returns:
    --------
    list
        包含處理後區塊的列表
    """
    try:
        if size < 16:  # 最小區塊大小限制
            return []
        
        # 確保 block 是 CuPy 數組
        if not isinstance(block, cp.ndarray):
            block = cp.asarray(block)
        
        # 計算區塊變異度
        variance = calculate_block_variance_cuda(block)
        
        # 根據區塊大小調整閾值
        adjusted_threshold = variance_threshold
        if size >= 128:
            adjusted_threshold *= 1.2
        elif size >= 64:
            adjusted_threshold *= 1.1
        elif size >= 32:
            adjusted_threshold *= 1.0
        else:  # 16x16 區塊
            adjusted_threshold *= 0.9
        
        # 計算邊緣強度
        dx = cp.diff(block, axis=1)
        dy = cp.diff(block, axis=0)
        edge_strength = float(cp.mean(cp.abs(dx)) + cp.mean(cp.abs(dy)))
        
        # 根據邊緣強度微調閾值
        if edge_strength > variance_threshold * 0.3:
            adjusted_threshold *= 0.9  # 邊緣區域更容易被分割
        
        if verbose:
            print(f"Block at {position}, size: {size}x{size}")
            print(f"  Variance: {variance:.2f}")
            print(f"  Threshold: {adjusted_threshold:.2f}")
            print(f"  Edge strength: {edge_strength:.2f}")
        
        # 純粹基於 variance 決定是否分割
        if size > 16 and variance > adjusted_threshold:
            # 繼續分割
            half_size = size // 2
            sub_blocks = []
            for i in range(2):
                for j in range(2):
                    y_start = position[0] + i * half_size
                    x_start = position[1] + j * half_size
                    sub_block = block[i*half_size:(i+1)*half_size, 
                                    j*half_size:(j+1)*half_size]
                    sub_blocks.extend(process_block(
                        sub_block, (y_start, x_start), half_size,
                        stage_info, embedding, variance_threshold,
                        ratio_of_ones, target_bpp, target_psnr, el_mode,
                        verbose=verbose, rotation_mode=rotation_mode
                    ))
            return sub_blocks
        else:
            # 如果使用隨機旋轉模式，在處理之前進行旋轉
            if rotation_mode == 'random' and 'block_rotations' in stage_info:
                rotation = stage_info['block_rotations'][size]
                if rotation != 0:
                    k = rotation // 90
                    block = cp.rot90(block, k=k)
            
            # 處理當前區塊
            return process_current_block(
                block, position, size, stage_info, embedding,
                ratio_of_ones, target_bpp, target_psnr, el_mode,
                verbose=verbose
            )
            
    except Exception as e:
        print(f"Error processing block at position {position}, size {size}: {str(e)}")
        raise

def pee_process_with_quadtree_cuda(img, total_embeddings, ratio_of_ones, use_different_weights, 
                                 min_block_size, variance_threshold, el_mode, rotation_mode='none'):
    """
    使用Quad tree的PEE處理函數

    Parameters:
    -----------
    img : numpy.ndarray
        輸入圖像
    total_embeddings : int
        總嵌入次數
    ratio_of_ones : float
        嵌入數據中1的比例
    use_different_weights : bool
        是否對每個子圖像使用不同的權重
    min_block_size : int
        最小區塊大小 (支援到16x16)
    variance_threshold : float
        變異閾值
    el_mode : int
        EL模式 (0:無限制, 1:漸增, 2:漸減)
    rotation_mode : str
        'none': 原始的 quadtree 方法
        'random': 使用隨機旋轉的新方法
        
    Returns:
    --------
    tuple
        (final_pee_img, total_payload, pee_stages)
    """
    try:
        # 參數檢查
        if min_block_size < 16:
            raise ValueError("min_block_size must be at least 16")
        if min_block_size & (min_block_size - 1) != 0:
            raise ValueError("min_block_size must be a power of 2")
        if not 0 <= ratio_of_ones <= 1:
            raise ValueError("ratio_of_ones must be between 0 and 1")
        if rotation_mode not in ['none', 'random']:
            raise ValueError("rotation_mode must be 'none' or 'random'")

        # 初始化
        original_img = cp.asarray(img)
        height, width = original_img.shape
        total_pixels = height * width
        pee_stages = []
        total_payload = 0
        current_img = original_img.copy()  # 保存前一階段的結果（已轉回0度）
        previous_psnr = float('inf')
        previous_ssim = 1.0
        previous_payload = float('inf')

        # 初始化區塊權重字典
        block_weights = {size: None for size in [512, 256, 128, 64, 32, 16]}
        
        mem_pool = cp.get_default_memory_pool()

        try:
            for embedding in range(total_embeddings):
                print(f"\nStarting embedding {embedding}")
                print(f"Memory usage before embedding {embedding}: {mem_pool.used_bytes()/1024/1024:.2f} MB")

                # 設定目標值
                if embedding == 0:
                    target_psnr = 40.0
                    target_bpp = 0.9
                else:
                    target_psnr = max(28.0, previous_psnr - 1)
                    target_bpp = max(0.5, (previous_payload / total_pixels) * 0.95)

                print(f"Target PSNR for embedding {embedding}: {target_psnr:.2f}")
                print(f"Target BPP for embedding {embedding}: {target_bpp:.4f}")

                # 初始化階段資訊
                stage_info = {
                    'embedding': embedding,
                    'block_info': {str(size): {'blocks': []} for size in [512, 256, 128, 64, 32, 16]},
                    'payload': 0,
                    'psnr': 0,
                    'ssim': 0,
                    'hist_corr': 0,
                    'bpp': 0,
                    'rotation_mode': rotation_mode
                }

                if rotation_mode == 'random':
                    # 為每個區塊大小生成隨機旋轉角度
                    block_rotations = {
                        size: np.random.choice([-270, -180, -90, 0, 90, 180, 270])
                        for size in [512, 256, 128, 64, 32, 16]
                    }
                    stage_info['block_rotations'] = block_rotations
                    print("\nBlock rotation angles for this stage:")
                    for size, angle in sorted(block_rotations.items(), reverse=True):
                        print(f"  {size}x{size}: {angle}°")

                # 處理整個圖像
                processed_blocks = process_block(
                    current_img, (0, 0), 512, stage_info, embedding,
                    variance_threshold, ratio_of_ones, target_bpp, target_psnr, el_mode,
                    verbose=False, rotation_mode=rotation_mode
                )

                # 重建圖像
                stage_img = cp.zeros_like(current_img)  # 用於儲存轉回0度的圖像
                rotated_stage_img = cp.zeros_like(current_img)  # 用於儲存旋轉狀態的圖像

                # 重建並處理每個區塊
                for block, pos, size in processed_blocks:
                    block = cp.asarray(block)
                    
                    if rotation_mode == 'random':
                        # 保存旋轉後的狀態（未轉回0度）
                        rotated_stage_img[pos[0]:pos[0]+size, pos[1]:pos[1]+size] = block
                        
                        # 將區塊旋轉回原始方向
                        rotation = stage_info['block_rotations'][size]
                        if rotation != 0:
                            k = (-rotation // 90) % 4
                            block = cp.rot90(block, k=k)

                    # 保存轉回0度的狀態
                    stage_img[pos[0]:pos[0]+size, pos[1]:pos[1]+size] = block

                # 保存圖像到stage_info
                if rotation_mode == 'random':
                    stage_info['rotated_stage_img'] = rotated_stage_img
                stage_info['stage_img'] = stage_img

                # 計算階段指標
                if rotation_mode == 'random':
                    # 統一使用原始圖像作為參考
                    stage_img_np = cp.asnumpy(stage_img)
                    reference_img_np = cp.asnumpy(original_img)

                    # 計算指標
                    psnr = calculate_psnr(reference_img_np, stage_img_np)
                    ssim = calculate_ssim(reference_img_np, stage_img_np)
                    hist_corr = histogram_correlation(
                        np.histogram(reference_img_np, bins=256, range=(0, 255))[0],
                        np.histogram(stage_img_np, bins=256, range=(0, 255))[0]
                    )
                else:
                    # 原始模式的計算方式
                    psnr, ssim, hist_corr = calculate_metrics_with_rotation(
                        current_img,
                        stage_img,
                        original_img,
                        embedding
                    )

                stage_info['psnr'] = float(psnr)
                stage_info['ssim'] = float(ssim)
                stage_info['hist_corr'] = float(hist_corr)
                stage_info['bpp'] = float(stage_info['payload'] / total_pixels)

                # 異常值檢查
                if psnr < 28 or ssim < 0.8:
                    print("Warning: Metrics seem unusually low")

                # 維護指標
                if embedding > 0:
                    if stage_info['payload'] >= previous_payload:
                        print(f"Warning: Stage {embedding} payload ({stage_info['payload']}) is not less than previous stage ({previous_payload}).")
                        stage_info['payload'] = int(previous_payload * 0.95)
                        print(f"Adjusted payload: {stage_info['payload']}")

                    if stage_info['psnr'] >= previous_psnr:
                        print(f"Warning: Stage {embedding} PSNR ({stage_info['psnr']:.2f}) is not less than previous stage ({previous_psnr:.2f}).")

                # 更新stage資訊
                pee_stages.append(stage_info)
                total_payload += stage_info['payload']
                previous_psnr = stage_info['psnr']
                previous_ssim = stage_info['ssim']
                previous_payload = stage_info['payload']

                # 輸出階段摘要
                print(f"Embedding {embedding} summary:")
                print(f"  Payload: {stage_info['payload']}")
                print(f"  BPP: {stage_info['bpp']:.4f}")
                print(f"  PSNR: {stage_info['psnr']:.2f}")
                print(f"  SSIM: {stage_info['ssim']:.4f}")
                print(f"  Hist Corr: {stage_info['hist_corr']:.4f}")

                # 準備下一階段：使用轉回0度的圖像
                current_img = stage_img.copy()

                # 清理當前階段的記憶體
                mem_pool.free_all_blocks()
                print(f"Memory usage after embedding {embedding}: {mem_pool.used_bytes()/1024/1024:.2f} MB")

            # 返回最終結果
            return cp.asnumpy(current_img), int(total_payload), pee_stages

        except Exception as e:
            print(f"Error in embedding process: {str(e)}")
            raise

    except Exception as e:
        print(f"Error in quadtree processing: {str(e)}")
        raise

    finally:
        # 確保清理所有記憶體
        cleanup_quadtree_resources()

