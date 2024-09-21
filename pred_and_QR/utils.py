import numpy as np
import struct
import cupy as cp
import random
import math
from numba import cuda
from image_processing import improved_predict_image_cuda
from common import calculate_psnr, improved_predict_image_cpu, pee_embedding_adaptive_cpu
from deap import base, creator, tools, algorithms

@cuda.jit
def pee_kernel(img, pred_img, data, embedded, payload, base_EL):
    x, y = cuda.grid(2)
    if 1 <= x < 511 and 1 <= y < 511:
        # 計算局部標準差
        local_sum = 0.0
        local_sum_sq = 0.0
        count = 0
        for i in range(max(0, x-1), min(512, x+2)):
            for j in range(max(0, y-1), min(512, y+2)):
                pixel = img[i, j]
                local_sum += pixel
                local_sum_sq += pixel * pixel
                count += 1
        local_mean = local_sum / count
        local_var = (local_sum_sq / count) - (local_mean * local_mean)
        local_std = math.sqrt(max(0, local_var))

        # 自適應 EL
        adaptive_EL = max(3, min(base_EL, int(base_EL * (1 - local_std / 255))))

        diff = int(img[x, y]) - int(pred_img[x, y])
        if abs(diff) < adaptive_EL and payload[0] < len(data):
            bit = data[payload[0]]
            cuda.atomic.add(payload, 0, 1)
            if diff >= 0:
                embedded[x, y] = img[x, y] + bit
            else:
                embedded[x, y] = img[x, y] - bit
        else:
            embedded[x, y] = img[x, y]

from image_processing import improved_predict_image_cuda

def pee_embedding_adaptive_cuda(img, data, weight, base_EL):
    if not isinstance(img, cp.ndarray):
        img = cp.asarray(img)
    if not isinstance(data, cp.ndarray):
        data = cp.asarray(data)

    # 使用 image_processing.py 中的函數進行預測
    pred_img = improved_predict_image_cuda(img, weight)

    embedded_img = cp.zeros_like(img)
    d_payload = cp.zeros(1, dtype=int)

    threads_per_block = (16, 16)
    blocks_per_grid = (32, 32)  # 針對 512x512 圖像優化

    pee_kernel[blocks_per_grid, threads_per_block](img, pred_img, data, embedded_img, d_payload, base_EL)

    payload = int(d_payload[0])
    embedded_data = data[:payload]

    return embedded_img, payload, embedded_data
    
def choose_el(img, rotation, current_payload):
    target_payload = 480000
    if current_payload < target_payload:
        if rotation == 0:
            return 7  # 第一次旋轉使用較大的 EL
        elif rotation == 1:
            return 5
        else:
            return 3
    else:
        return 3  # 如果已達到目標，使用較小的 EL 以維持圖像質量

def generate_random_binary_array(size, ratio_of_ones=0.5):
    """生成指定大小的随机二进制数组，可调整1的比例"""
    return np.random.choice([0, 1], size=size, p=[1-ratio_of_ones, ratio_of_ones])

# 全局變量來追蹤是否已經創建了類
_created_classes = False

def ensure_deap_classes():
    global _created_classes
    if not _created_classes:
        if 'FitnessMulti' not in creator.__dict__:
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
        if 'Individual' not in creator.__dict__:
            creator.create("Individual", list, fitness=creator.FitnessMulti)
        _created_classes = True

def find_best_weights_ga(img, data, EL, population_size=50, generations=20, max_weight=15):
    ensure_deap_classes()
    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.uniform, 0.1, max_weight)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_weights(individual):
        try:
            weights = cp.asarray(individual, dtype=cp.float32)
            img_gpu = cp.asarray(img)
            data_gpu = cp.asarray(data)

            pred_img = improved_predict_image_cuda(img_gpu, weights)
            embedded, payload, _ = pee_embedding_adaptive_cuda(img_gpu, data_gpu, pred_img, EL)
            
            psnr = calculate_psnr(cp.asnumpy(img_gpu), cp.asnumpy(embedded))
            return payload, psnr
        except cp.cuda.runtime.CUDARuntimeError as e:
            print(f"CUDA error in evaluate_weights: {e}")
            return 0, 0

    toolbox.register("evaluate", evaluate_weights)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    population = toolbox.population(n=population_size)
    
    try:
        algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, 
                                  lambda_=population_size, cxpb=0.7, mutpb=0.2, 
                                  ngen=generations, verbose=False)
    except Exception as e:
        print(f"Error in genetic algorithm: {e}")
        return [1.0, 1.0, 1.0, 1.0], (0, 0)

    best_ind = tools.selBest(population, 1)[0]
    return best_ind, best_ind.fitness.values

def encode_pee_info(total_rotations, pee_stages):
    print(f"Debug: Encoding PEE info")
    print(f"Total rotations: {total_rotations}")
    
    encoded = struct.pack('B', total_rotations)
    for stage in pee_stages:
        block_params = stage['block_params']
        encoded += struct.pack('H', len(block_params))
        for block in block_params:
            weights = block['weights']
            EL = block['EL']
            payload = block['payload']
            
            # 確保 payload 是兩個 unsigned short（16 bits）
            payload_high = payload >> 16
            payload_low = payload & 0xFFFF
            
            # 打包為 19 bytes的數據
            encoded += struct.pack('ffffBHH', *weights, EL, payload_high, payload_low)
    
    print(f"Debug: Encoded data length: {len(encoded)} bytes")
    return encoded

def decode_pee_info(encoded_data):
    if not isinstance(encoded_data, (bytes, bytearray)):
        raise TypeError(f"Expected bytes or bytearray, got {type(encoded_data)}")
    
    if len(encoded_data) < 1:
        raise ValueError("Encoded data is too short")

    offset = 0
    total_rotations = struct.unpack('B', encoded_data[offset:offset+1])[0]
    offset += 1

    pee_stages = []
    try:
        for _ in range(total_rotations):
            if offset + 2 > len(encoded_data):
                raise ValueError("Encoded data is incomplete (block count)")
            
            num_blocks = struct.unpack('H', encoded_data[offset:offset+2])[0]
            offset += 2
            
            block_params = []
            for _ in range(num_blocks):
                if offset + 19 > len(encoded_data):
                    raise ValueError("Encoded data is incomplete (block data)")
                
                # 解包為 19 bytes的數據
                data = struct.unpack('ffffBHH', encoded_data[offset:offset+19])
                weights = list(data[:4])
                EL = data[4]
                payload = (data[5] << 16) | data[6]
                block_params.append({
                    'weights': weights,
                    'EL': EL,
                    'payload': payload
                })
                offset += 19
            
            pee_stages.append({'block_params': block_params})
    except struct.error as e:
        raise ValueError(f"Error decoding data: {e}")

    print(f"Debug: Decoded data length: {offset} bytes")
    print(f"Debug: Decoded rotations: {total_rotations}")
    print(f"Debug: Number of stages: {len(pee_stages)}")
    return total_rotations, pee_stages

def find_max(array1D):
    """找出一維陣列中最大值"""
    if not array1D:
        return None
    if isinstance(array1D[0], (list, np.ndarray)):
        return max(range(len(array1D)), key=lambda i: max(array1D[i]) if len(array1D[i]) > 0 else float('-inf'))
    else:
        return max(range(len(array1D)), key=lambda i: array1D[i])

