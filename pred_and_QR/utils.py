import numpy as np
import struct
import itertools
import math
import cupy as cp
import matplotlib.pyplot as plt
from common import *
from pee import *
from prettytable import PrettyTable

def generate_random_binary_array(size, ratio_of_ones=0.5):
    """生成指定大小的随机二进制数组，可调整1的比例"""
    return np.random.choice([0, 1], size=size, p=[1-ratio_of_ones, ratio_of_ones])

@cuda.jit
def evaluate_weights_kernel(img, data, EL, weight_combinations, results, target_bpp, target_psnr, stage):
    idx = cuda.grid(1)
    if idx < weight_combinations.shape[0]:
        w1, w2, w3, w4 = weight_combinations[idx]
        
        height, width = img.shape
        payload = 0
        mse = 0.0
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Prediction
                ul = img[y-1, x-1]
                up = img[y-1, x]
                ur = img[y-1, x+1]
                left = img[y, x-1]
                p = (w1*up + w2*ul + w3*ur + w4*left) / (w1 + w2 + w3 + w4)
                pred_val = round(p)
                
                # Embedding
                diff = int(img[y, x]) - int(pred_val)
                if abs(diff) < EL[y, x] and payload < data.shape[0]:  # Use EL[y, x] instead of just EL
                    bit = data[payload]
                    payload += 1
                    if stage == 0:
                        # More aggressive embedding for stage 0
                        embedding_strength = min(3, EL[y, x] - abs(diff))
                    else:
                        embedding_strength = 1
                    
                    if diff >= 0:
                        embedded_val = min(255, int(img[y, x]) + bit * embedding_strength)
                    else:
                        embedded_val = max(0, int(img[y, x]) - (1 - bit) * embedding_strength)
                    mse += (embedded_val - img[y, x]) ** 2
                else:
                    mse += 0  # No change to pixel
        
        if mse > 0:
            psnr = 10 * math.log10((255 * 255) / (mse / (height * width)))
        else:
            psnr = 100.0  # High value for perfect embedding
        
        bpp = payload / (height * width)
        
        # Adaptive fitness criteria
        bpp_fitness = min(1.0, bpp / target_bpp)
        psnr_fitness = max(0, 1 - abs(psnr - target_psnr) / target_psnr)
        
        if stage == 0:
            fitness = bpp_fitness * 0.7 + psnr_fitness * 0.3
        else:
            fitness = bpp_fitness * 0.5 + psnr_fitness * 0.5
        
        results[idx, 0] = payload
        results[idx, 1] = psnr
        results[idx, 2] = fitness

def brute_force_weight_search_cuda(img, data, local_el, target_bpp, target_psnr, stage, block_size=None):
    """
    使用暴力搜索找到最佳的權重組合
    
    Parameters:
    -----------
    img : numpy.ndarray or cupy.ndarray
        輸入圖像
    data : numpy.ndarray or cupy.ndarray
        要嵌入的數據
    local_el : numpy.ndarray or cupy.ndarray
        局部EL值
    target_bpp : float
        目標BPP
    target_psnr : float
        目標PSNR
    stage : int
        當前嵌入階段
    block_size : int, optional
        區塊大小（僅在quad tree模式下使用）
    
    Returns:
    --------
    tuple
        (最佳權重, (payload, psnr))
    """
    img = cp.asarray(img)
    data = cp.asarray(data)
    
    weight_combinations = cp.array(list(itertools.product(range(1, 16), repeat=4)), dtype=cp.int32)
    results = cp.zeros((len(weight_combinations), 3), dtype=cp.float32)
    
    # 根據區塊大小調整參數
    if block_size is not None:
        # 對較大的區塊增加權重範圍
        if block_size >= 256:
            weight_combinations = cp.array(list(itertools.product(range(1, 20), repeat=4)), dtype=cp.int32)
        # 對較小的區塊減小權重範圍
        elif block_size <= 64:
            weight_combinations = cp.array(list(itertools.product(range(1, 12), repeat=4)), dtype=cp.int32)
    
    threads_per_block = 256
    blocks_per_grid = (len(weight_combinations) + threads_per_block - 1) // threads_per_block
    
    # 調用評估kernel
    evaluate_weights_kernel[blocks_per_grid, threads_per_block](
        img, data, local_el, weight_combinations, results, 
        target_bpp, target_psnr, stage
    )
    
    # 根據區塊大小調整適應度計算
    if block_size is not None:
        # 對較大的區塊更重視PSNR
        if block_size >= 256:
            results[:, 2] = results[:, 2] * 0.4 + (results[:, 1] / target_psnr) * 0.6
        # 對較小的區塊更重視payload
        elif block_size <= 64:
            results[:, 2] = results[:, 2] * 0.7 + (results[:, 1] / target_psnr) * 0.3
    
    best_idx = cp.argmax(results[:, 2])
    best_weights = weight_combinations[best_idx]
    best_payload, best_psnr, best_fitness = results[best_idx]
    
    return cp.asnumpy(best_weights), (float(best_payload), float(best_psnr))

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
    創建PEE資訊表格
    
    Parameters:
    -----------
    pee_stages : list
        包含所有PEE階段資訊的列表
    use_different_weights : bool
        是否對每個子圖像使用不同的權重
    total_pixels : int
        圖像總像素數
    split_size : int
        分割大小
    quad_tree : bool, optional
        是否使用quad tree模式
    
    Returns:
    --------
    PrettyTable
        格式化的表格
    """
    table = PrettyTable()
    
    if quad_tree:
        # Quad tree模式的表格欄位
        table.field_names = [
            "Embedding", "Block Size", "Block Position", "Payload", "BPP",
            "PSNR", "SSIM", "Hist Corr", "Weights", "EL", "Note"
        ]
    else:
        # 原有模式的表格欄位
        table.field_names = [
            "Embedding", "Sub-image", "Payload", "BPP", "PSNR", "SSIM",
            "Hist Corr", "Weights", "EL", "Rotation", "Note"
        ]
    
    for stage in pee_stages:
        # 添加整體stage資訊
        if quad_tree:
            # Quad tree模式的整體資訊
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
            # 原有模式的整體資訊
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
        
        table.add_row(["-" * 5] * len(table.field_names))
        
        if quad_tree:
            # 處理quad tree模式的區塊資訊
            for size in sorted(stage['block_info'].keys(), key=int, reverse=True):
                blocks = stage['block_info'][size]['blocks']
                for block in blocks:
                    block_pixels = block['size'] * block['size']
                    table.add_row([
                        stage['embedding'],
                        f"{block['size']}x{block['size']}",
                        f"({block['position'][0]}, {block['position'][1]})",
                        block['payload'],
                        f"{block['payload'] / block_pixels:.4f}",
                        f"{block['psnr']:.2f}",
                        f"{block['ssim']:.4f}",
                        f"{block['hist_corr']:.4f}",
                        ", ".join([f"{w:.2f}" for w in block['weights']]),
                        block.get('EL', '-'),
                        "Different weights" if use_different_weights else ""
                    ])
        else:
            # 處理原有模式的區塊資訊
            total_blocks = split_size * split_size
            sub_image_pixels = total_pixels // total_blocks
            
            for i, block in enumerate(stage['block_params']):
                table.add_row([
                    stage['embedding'] if i == 0 else "",
                    i,
                    block['payload'],
                    f"{block['payload'] / sub_image_pixels:.4f}",
                    f"{block['psnr']:.2f}",
                    f"{block['ssim']:.4f}",
                    f"{block['hist_corr']:.4f}",
                    ", ".join([f"{w:.2f}" for w in block['weights']]),
                    block['EL'],
                    f"{block['rotation']}°",
                    "Different weights" if use_different_weights else ""
                ])
        
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

@cuda.jit
def calculate_variance_kernel(block, variance_result):
    """
    CUDA kernel for calculating variance of image blocks
    """
    x, y = cuda.grid(2)
    if x < block.shape[1] and y < block.shape[0]:
        # 使用shared memory來優化性能
        tid = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x
        block_size = block.shape[0] * block.shape[1]
        
        # 計算區域平均值
        local_sum = 0
        local_sum_sq = 0
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                pixel_value = block[i, j]
                local_sum += pixel_value
                local_sum_sq += pixel_value * pixel_value
        
        mean = local_sum / block_size
        variance = (local_sum_sq / block_size) - (mean * mean)
        
        # 只需要一個線程寫入結果
        if x == 0 and y == 0:
            variance_result[0] = variance

def calculate_block_variance_cuda(block):
    """
    Calculate variance of a block using CUDA
    """
    threads_per_block = (16, 16)
    blocks_per_grid_x = (block.shape[1] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (block.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    variance_result = cuda.device_array(1, dtype=np.float32)
    
    calculate_variance_kernel[blocks_per_grid, threads_per_block](block, variance_result)
    
    return variance_result[0]