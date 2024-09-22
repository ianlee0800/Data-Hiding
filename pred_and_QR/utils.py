import numpy as np
import struct
import cupy as cp
from numba import cuda
import random
from image_processing import improved_predict_image_cuda
from common import *
from deap import base, creator, tools, algorithms
from pee import *

def generate_random_binary_array(size, ratio_of_ones=0.5):
    """生成指定大小的随机二进制数组，可调整1的比例"""
    return np.random.choice([0, 1], size=size, p=[1-ratio_of_ones, ratio_of_ones])

@cuda.jit
def evaluate_population_kernel(population, img, data, EL, results):
    idx = cuda.grid(1)
    if idx < population.shape[0]:
        weights = population[idx]
        # 注意：这里我们无法直接在 CUDA kernel 中调用其他函数，
        # 所以我们需要在 CPU 上执行这些操作
        results[idx, 0] = -1  # 标记为需要在 CPU 上计算
        results[idx, 1] = -1

def custom_selNSGA2(individuals, k):
    try:
        return tools.selNSGA2(individuals, k)
    except IndexError:
        # 如果出現索引錯誤，使用一個更簡單的選擇方法
        return sorted(individuals, key=lambda ind: ind.fitness.values[0], reverse=True)[:k]

def gpu_evaluate_population(population, img, data, EL):
    pop_size = len(population)
    d_population = cuda.to_device(np.array([ind[:] for ind in population]))
    d_img = cuda.to_device(img)
    d_data = cuda.to_device(data)
    d_results = cuda.device_array((pop_size, 2), dtype=np.float32)

    threads_per_block = 256
    blocks = (pop_size + threads_per_block - 1) // threads_per_block

    evaluate_population_kernel[blocks, threads_per_block](d_population, d_img, d_data, EL, d_results)

    results = d_results.copy_to_host()

    # 在 CPU 上完成計算
    for i, result in enumerate(results):
        if result[0] == -1:
            weights = cp.asarray(population[i])
            pred_img = improved_predict_image_cuda(cp.asarray(img), weights)
            embedded, payload, _ = pee_embedding_adaptive_cuda(cp.asarray(img), cp.asarray(data), pred_img, EL)
            psnr = calculate_psnr(img, cp.asnumpy(embedded))
            results[i] = [max(0, payload), max(0, psnr)]  # 確保值非負

    return [(float(r[0]), float(r[1])) for r in results]

def safe_stats(x):
    if not x:
        return [0.0, 0.0]
    valid_fitness = [ind.fitness.values for ind in x if hasattr(ind, 'fitness') and ind.fitness.valid]
    if not valid_fitness:
        return [0.0, 0.0]
    return np.mean(valid_fitness, axis=0).tolist()

def find_best_weights_ga(img, data, EL, toolbox, population_size=50, generations=20, early_stop=5):
    creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 1, 15)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        weights = np.array(individual, dtype=np.float32)
        pred_img = improved_predict_image_cuda(cp.asarray(img, dtype=cp.uint8), cp.asarray(weights))
        embedded, payload, _ = pee_embedding_adaptive_cuda(cp.asarray(img, dtype=cp.uint8), 
                                                          cp.asarray(data, dtype=cp.uint8), 
                                                          pred_img, EL)
        psnr = calculate_psnr(img, cp.asnumpy(embedded))
        return payload, psnr

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
    return np.array(best_ind), (-best_ind.fitness.values[1], best_ind.fitness.values[0])  # 返回 (PSNR, payload)

def encode_pee_info(pee_info):
    encoded = struct.pack('B', pee_info['total_rotations'])
    for stage in pee_info['stages']:
        encoded += struct.pack('B', stage['rotation'])
        encoded += struct.pack('B', len(stage['block_params']))
        for block in stage['block_params']:
            encoded += struct.pack('ffffBI', *block['weights'], block['EL'], block['payload'])
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



