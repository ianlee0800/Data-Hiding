import numpy as np
import struct
import itertools
from common import *
from pee import *
from prettytable import PrettyTable

def generate_random_binary_array(size, ratio_of_ones=0.5):
    """生成指定大小的随机二进制数组，可调整1的比例"""
    return np.random.choice([0, 1], size=size, p=[1-ratio_of_ones, ratio_of_ones])

@cuda.jit
def evaluate_weights_kernel(img, data, EL, weight_combinations, results, target_bpp, target_psnr):
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
                if abs(diff) < EL and payload < data.shape[0]:
                    bit = data[payload]
                    payload += 1
                    if diff >= 0:
                        embedded_val = min(255, int(img[y, x]) + bit)
                    else:
                        embedded_val = max(0, int(img[y, x]) - (1 - bit))
                    mse += (embedded_val - img[y, x]) ** 2
                else:
                    mse += 0  # No change to the pixel
        
        if mse > 0:
            psnr = 10 * math.log10((255 * 255) / (mse / (height * width)))
        else:
            psnr = 100.0  # A high value for perfect embedding
        
        bpp = payload / (height * width)
        
        # Fitness calculation
        bpp_fitness = min(1.0, bpp / target_bpp)
        psnr_fitness = max(0, 1 - abs(psnr - target_psnr) / target_psnr)
        fitness = bpp_fitness * 0.7 + psnr_fitness * 0.3  # Prioritize BPP
        
        results[idx, 0] = payload
        results[idx, 1] = psnr
        results[idx, 2] = fitness

def brute_force_weight_search_cuda(img, data, EL, target_bpp, target_psnr, weight_range=range(1, 16)):
    img = cp.asarray(img)
    data = cp.asarray(data)
    
    weight_combinations = cp.array(list(itertools.product(weight_range, repeat=4)), dtype=cp.int32)
    
    results = cp.zeros((len(weight_combinations), 3), dtype=cp.float32)

    threads_per_block = 256
    blocks_per_grid = (len(weight_combinations) + threads_per_block - 1) // threads_per_block

    evaluate_weights_kernel[blocks_per_grid, threads_per_block](img, data, EL, weight_combinations, results, target_bpp, target_psnr)

    best_idx = cp.argmax(results[:, 2])  # Use fitness for selection
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

# 更新打印函数以包含新的信息
def create_pee_info_table(pee_stages, use_different_weights, total_pixels):
    table = PrettyTable()
    table.field_names = ["Embedding", "Sub-image", "Payload", "BPP", "PSNR", "SSIM", "Hist Corr", "Weights", "EL", "Rotation", "Note"]
    
    for stage in pee_stages:
        # 添加整體 stage 信息
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
        table.add_row(["-" * 5] * 11)  # 添加分隔行
        
        # 添加每個子圖像的信息
        for i, block in enumerate(stage['block_params']):
            sub_image_pixels = total_pixels // 4
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
        table.add_row(["-" * 5] * 11)  # 添加分隔行
    
    return table

