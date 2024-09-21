import numpy as np
import struct
import cupy as cp
from numba import cuda
from image_processing import improved_predict_image_cuda
from common import *
from deap import base, creator, tools, algorithms
from pee import *

def generate_random_binary_array(size, ratio_of_ones=0.5):
    """生成指定大小的随机二进制数组，可调整1的比例"""
    return np.random.choice([0, 1], size=size, p=[1-ratio_of_ones, ratio_of_ones])

# 確保只創建一次
if 'FitnessMax' not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
if 'Individual' not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

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

def find_best_weights_ga(img, data, EL, population_size=50, generations=20, early_stop=5):
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 1, 15)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        return individual.fitness.values if individual.fitness.valid else (0.0, 0.0)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=15, indpb=0.2)
    
    # 使用自定義的選擇函數來避免錯誤
    def safe_select(individuals, k):
        try:
            return tools.selNSGA2(individuals, k)
        except Exception as e:
            print(f"NSGA-II selection failed: {e}")
            # 如果 NSGA-II 失敗，使用簡單的錦標賽選擇
            return tools.selTournament(individuals, k, tournsize=3)
    
    toolbox.register("select", safe_select)

    population = toolbox.population(n=population_size)
    
    hof = tools.ParetoFront()
    
    last_best = 0
    no_improvement = 0

    for gen in range(generations):
        try:
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
            
            fits = gpu_evaluate_population(offspring, img, data, EL)
            for ind, fit in zip(offspring, fits):
                ind.fitness.values = fit

            population = toolbox.select(offspring + population, k=population_size)
            
            hof.update(population)
            
            current_best = max([ind.fitness.values[0] for ind in population if ind.fitness.valid], default=0)
            if current_best > last_best:
                last_best = current_best
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= early_stop:
                print(f"Early stopping at generation {gen}")
                break

        except Exception as e:
            print(f"Error in generation {gen}: {str(e)}")

    best_ind = max(population, key=lambda x: x.fitness.values[0] if x.fitness.valid else float('-inf'))
    return np.array(best_ind), best_ind.fitness.values if best_ind.fitness.valid else (0.0, 0.0)

def encode_pee_info(total_rotations, pee_stages):
    encoded = struct.pack('B', total_rotations)
    for stage in pee_stages:
        block_params = stage['block_params']
        encoded += struct.pack('I', len(block_params))
        for block in block_params:
            weights = ensure_type(block['weights'], DataType.FLOAT32)
            EL = ensure_type(block['EL'], DataType.UINT8)
            payload = ensure_type(block['payload'], DataType.INT32)
            encoded += struct.pack('ffffBI', *weights, EL, payload)
    return encoded

def decode_pee_info(encoded_data):
    if len(encoded_data) < 5:  # 1 byte for total_rotations + 4 bytes for first block count
        raise ValueError("Encoded data is too short")

    offset = 0
    total_rotations = struct.unpack('B', encoded_data[offset:offset+1])[0]
    offset += 1

    pee_stages = []
    for _ in range(total_rotations):
        if offset + 4 > len(encoded_data):
            raise ValueError("Encoded data is incomplete (block count)")
        
        num_blocks = struct.unpack('I', encoded_data[offset:offset+4])[0]
        offset += 4
        
        block_params = []
        for _ in range(num_blocks):
            if offset + 21 > len(encoded_data):
                raise ValueError("Encoded data is incomplete (block data)")
            
            data = struct.unpack('ffffBI', encoded_data[offset:offset+21])
            weights = ensure_type(data[:4], DataType.FLOAT32)
            EL = ensure_type(data[4], DataType.UINT8)
            payload = ensure_type(data[5], DataType.INT32)
            block_params.append({
                'weights': weights,
                'EL': EL,
                'payload': payload
            })
            offset += 21
        
        pee_stages.append({'block_params': block_params})

    return total_rotations, pee_stages



