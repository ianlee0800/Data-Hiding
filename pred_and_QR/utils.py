import numpy as np
import struct
import cupy as cp
import itertools
import time
from common import *
from deap import base, creator, tools, algorithms
from pee import *
import random
from prettytable import PrettyTable

def generate_random_binary_array(size, ratio_of_ones=0.5):
    """生成指定大小的随机二进制数组，可调整1的比例"""
    return np.random.choice([0, 1], size=size, p=[1-ratio_of_ones, ratio_of_ones])

def custom_selNSGA2(individuals, k):
    try:
        return tools.selNSGA2(individuals, k)
    except IndexError:
        # 如果出現索引錯誤，使用一個更簡單的選擇方法
        return sorted(individuals, key=lambda ind: ind.fitness.values[0], reverse=True)[:k]

# 確保這些只被創建一次
if 'FitnessMax' not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
if 'Individual' not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

def brute_force_weight_search(img, data, EL, weight_range=range(1, 16)):
    best_weights = None
    best_fitness = float('-inf')
    start_time = time.time()
    total_combinations = 0

    for weights in itertools.product(weight_range, repeat=4):
        total_combinations += 1
        pred_img = improved_predict_image_cuda(img, weights)
        embedded, payload, _ = pee_embedding_adaptive_cuda(img, data, pred_img, EL)
        psnr = calculate_psnr(to_numpy(img), to_numpy(embedded))
        fitness = (payload, psnr)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_weights = weights

        if total_combinations % 1000 == 0:  # 每1000次組合輸出一次進度
            elapsed_time = time.time() - start_time
            print(f"Processed {total_combinations} combinations. Elapsed time: {elapsed_time:.2f} seconds")

    total_time = time.time() - start_time
    print(f"Total combinations: {total_combinations}")
    print(f"Total search time: {total_time:.2f} seconds")
    return best_weights, best_fitness

# CUDA kernel for weight evaluation
@cuda.jit
def evaluate_weights_kernel(img, data, EL, weight_combinations, results):
    idx = cuda.grid(1)
    if idx < weight_combinations.shape[0]:
        w1, w2, w3, w4 = weight_combinations[idx]
        weights = cuda.local.array(4, dtype=np.int32)
        weights[0], weights[1], weights[2], weights[3] = w1, w2, w3, w4

        # Simplified prediction and embedding (you'll need to implement these)
        pred_img = improved_predict_image_cuda(img, weights)
        embedded, payload = pee_embedding_adaptive_cuda(img, data, pred_img, EL)
        
        psnr = calculate_psnr(img, embedded)
        results[idx, 0] = payload
        results[idx, 1] = psnr

def brute_force_weight_search_cuda(img, data, EL, weight_range=range(1, 16)):
    # Generate all weight combinations
    weight_combinations = np.array(list(itertools.product(weight_range, repeat=4)), dtype=np.int32)
    
    # Move data to GPU
    d_img = cuda.to_device(img)
    d_data = cuda.to_device(data)
    d_weight_combinations = cuda.to_device(weight_combinations)
    d_results = cuda.device_array((len(weight_combinations), 2), dtype=np.float32)

    # Set up grid and block dimensions
    threads_per_block = 256
    blocks_per_grid = (len(weight_combinations) + threads_per_block - 1) // threads_per_block

    # Launch kernel
    evaluate_weights_kernel[blocks_per_grid, threads_per_block](d_img, d_data, EL, d_weight_combinations, d_results)

    # Retrieve results
    results = d_results.copy_to_host()

    # Find best result
    best_idx = np.argmax(results[:, 0])  # Assuming we want to maximize payload
    best_weights = weight_combinations[best_idx]
    best_payload, best_psnr = results[best_idx]

    return best_weights, (best_payload, best_psnr)

def find_best_weights_ga(img, data, EL, toolbox, population_size=50, generations=20):
    
    def evaluate(individual, img=img, data=data, EL=EL):
        weights = np.array(individual)
        pred_img = improved_predict_image_cpu(img, weights)
        embedded, payload, _ = pee_embedding_adaptive(img, data, pred_img, EL)
        psnr = calculate_psnr(img, embedded)
        return payload, psnr

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, 
                                       ngen=generations, stats=stats, 
                                       halloffame=hof, verbose=True)

    best_ind = tools.selBest(pop, 1)[0]
    return np.array(best_ind), best_ind.fitness.values

def find_best_weights_ga_cuda(img, data, EL, population_size=50, generations=20):
    def evaluate(individual):
        weights = cp.array([int(w) for w in individual], dtype=cp.int32)
        pred_img = improved_predict_image_cuda(img, weights)
        embedded, payload, _ = pee_embedding_adaptive_cuda(img, data, pred_img, EL)
        psnr = calculate_psnr(cp.asnumpy(img), cp.asnumpy(embedded))
        return int(payload), float(psnr)

    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 1, 15)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=15, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, 
                                       ngen=generations, stats=stats, 
                                       halloffame=hof, verbose=True)

    best_ind = tools.selBest(pop, 1)[0]
    return cp.array([int(w) for w in best_ind], dtype=cp.int32), best_ind.fitness.values

def encode_pee_info(pee_info):
    encoded = struct.pack('I', pee_info['total_rotations'])
    for stage in pee_info['stages']:
        encoded += struct.pack('I', stage['rotation'])
        encoded += struct.pack('I', len(stage['block_params']))
        for block in stage['block_params']:
            encoded += struct.pack('4I', *[int(w) for w in block['weights']])
            encoded += struct.pack('II', block['EL'], block['payload'])
    return encoded

def decode_pee_info(encoded_data):
    if len(encoded_data) < 2:
        raise ValueError("Encoded data is too short")

    offset = 0
    total_rotations = struct.unpack('B', encoded_data[offset:offset+1])[0]
    offset += 1

    stages = []
    for _ in range(total_rotations):
        if offset + 2 > len(encoded_data):
            raise ValueError("Encoded data is incomplete (rotation or block count)")
        
        rotation = struct.unpack('B', encoded_data[offset:offset+1])[0]
        offset += 1
        num_blocks = struct.unpack('B', encoded_data[offset:offset+1])[0]
        offset += 1
        
        block_params = []
        for _ in range(num_blocks):
            if offset + 21 > len(encoded_data):
                raise ValueError("Encoded data is incomplete (block data)")
            
            data = struct.unpack('ffffBI', encoded_data[offset:offset+21])
            block_params.append({
                'weights': data[:4],
                'EL': data[4],
                'payload': data[5]
            })
            offset += 21
        
        stages.append({
            'rotation': rotation,
            'block_params': block_params
        })

    return {
        'total_rotations': total_rotations,
        'stages': stages
    }

def create_pee_info_table(pee_stages):
    table = PrettyTable()
    table.field_names = ["Rotation", "Payload", "BPP", "PSNR", "SSIM", "Hist Corr"]
    for stage in pee_stages:
        table.add_row([
            stage['rotation'],
            stage['payload'],
            f"{stage['bpp']:.4f}",
            f"{stage['psnr']:.2f}",
            f"{stage['ssim']:.4f}",
            f"{stage['hist_corr']:.4f}"
        ])
    return table
