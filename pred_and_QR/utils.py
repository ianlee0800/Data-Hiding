import numpy as np
import struct
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
import time
import cupy as cp
import random
import math
from numba import cuda
from image_processing import improved_predict_image_cuda
from common import calculate_psnr
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
    remaining_rotations = 5 - rotation
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

def evaluate_weights(weights, img, data, EL):
    pred_img = improved_predict_image_cuda(img, weights)
    embedded_img, payload, _ = pee_embedding_adaptive_cuda(img, data, weights, EL)
    psnr = calculate_psnr(cp.asnumpy(img), cp.asnumpy(embedded_img))
    return payload, psnr

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

    toolbox.register("attr_int", random.randint, 1, max_weight)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_wrapper(individual):
        return evaluate_weights(individual, img, data, EL)

    toolbox.register("evaluate", eval_wrapper)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=max_weight, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    population = toolbox.population(n=population_size)
    algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, 
                              lambda_=population_size, cxpb=0.7, mutpb=0.2, 
                              ngen=generations, verbose=False)

    best_ind = tools.selBest(population, 1)[0]
    return best_ind, best_ind.fitness.values

def encode_pee_info(total_rotations, weights, payloads, ELs):
    print(f"Debug: Encoding PEE info")
    print(f"Total rotations: {total_rotations}")
    print(f"Weights: {weights}")
    print(f"Payloads: {payloads}")
    print(f"ELs: {ELs}")
    
    if not isinstance(total_rotations, int):
        raise TypeError(f"Expected total_rotations to be an integer, got {type(total_rotations)}")
    if not isinstance(weights, list) or len(weights) != total_rotations:
        raise ValueError(f"Expected weights to be a list of {total_rotations} elements, got {weights}")
    if not isinstance(payloads, list) or len(payloads) != total_rotations:
        raise ValueError(f"Expected payloads to be a list of {total_rotations} elements, got {payloads}")
    if not isinstance(ELs, list) or len(ELs) != total_rotations:
        raise ValueError(f"Expected ELs to be a list of {total_rotations} elements, got {ELs}")
    
    encoded = struct.pack('B', total_rotations)
    for rotation in range(total_rotations):
        if not isinstance(weights[rotation], (list, tuple)) or len(weights[rotation]) != 4:
            raise ValueError(f"Expected weights[{rotation}] to be a list or tuple of 4 integers, got {weights[rotation]}")
        if not isinstance(ELs[rotation], int):
            raise ValueError(f"Expected ELs[{rotation}] to be an integer, got {ELs[rotation]}")
        if not isinstance(payloads[rotation], int):
            raise ValueError(f"Expected payloads[{rotation}] to be an integer, got {payloads[rotation]}")
        
        try:
            encoded += struct.pack('BBBBB', *weights[rotation], ELs[rotation])
            encoded += struct.pack('I', payloads[rotation])  # 使用 'I' 代替 'H' 來存儲更大的整數
        except struct.error as e:
            print(f"Error packing data for rotation {rotation}: {e}")
            print(f"Weights: {weights[rotation]}, EL: {ELs[rotation]}, Payload: {payloads[rotation]}")
            raise
    
    return encoded

def decode_pee_info(encoded_data):
    if not isinstance(encoded_data, (bytes, bytearray)):
        raise TypeError(f"Expected bytes or bytearray, got {type(encoded_data)}")
    
    if len(encoded_data) < 1:
        raise ValueError("Encoded data is too short")

    total_rotations = struct.unpack('B', encoded_data[:1])[0]
    weights = []
    payloads = []
    ELs = []
    offset = 1
    try:
        for _ in range(total_rotations):
            if offset + 9 > len(encoded_data):  # 改為 9，因為我們現在用 4 bytes 存儲 payload
                raise ValueError("Encoded data is incomplete")
            weight_and_el = struct.unpack('BBBBB', encoded_data[offset:offset+5])
            weights.append(list(weight_and_el[:4]))
            ELs.append(weight_and_el[4])
            offset += 5
            payloads.append(struct.unpack('I', encoded_data[offset:offset+4])[0])  # 使用 'I' 來解包
            offset += 4
    except struct.error as e:
        raise ValueError(f"Error decoding data: {e}")

    return total_rotations, weights, payloads, ELs

def find_max(array1D):
    """找出一維陣列中最大值"""
    if not array1D:
        return None
    if isinstance(array1D[0], (list, np.ndarray)):
        return max(range(len(array1D)), key=lambda i: max(array1D[i]) if len(array1D[i]) > 0 else float('-inf'))
    else:
        return max(range(len(array1D)), key=lambda i: array1D[i])

