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

def find_best_weights_ga_cuda(img, data, EL, population_size=50, generations=50, target_bpp=0.7, target_psnr=30.0, stage=0, timeout=120):
    start_time = time.time()
    last_improvement = 0
    best_fitness = float('-inf')

    def evaluate(individual):
        if time.time() - start_time > timeout:
            raise TimeoutError("GA optimization timed out")

        weights = cp.array([int(w) for w in individual], dtype=cp.int32)
        pred_img = improved_predict_image_cuda(img, weights)
        embedded, payload, _ = pee_embedding_adaptive_cuda(img, data, pred_img, EL, stage=stage, target_psnr=target_psnr)
        psnr = calculate_psnr(cp.asnumpy(img), cp.asnumpy(embedded))
        ssim = calculate_ssim(cp.asnumpy(img), cp.asnumpy(embedded))
        bpp = payload / img.size

        # Heavily emphasize BPP
        bpp_fitness = min(3.0, (bpp / target_bpp) ** 3)  # Allow BPP fitness to go up to 3.0
        psnr_fitness = max(0, 1.0 - min(1.0, abs(psnr - target_psnr) / target_psnr))
        ssim_fitness = ssim
        
        # Adjust weights to strongly prioritize BPP, especially in early stages
        stage_factor = max(0, 1 - stage * 0.2)  # Decreases with each stage
        bpp_weight = 0.8 + 0.1 * stage_factor
        psnr_weight = 0.15 - 0.05 * stage_factor
        ssim_weight = 0.05 - 0.05 * stage_factor
        
        fitness = bpp_weight * bpp_fitness + psnr_weight * psnr_fitness + ssim_weight * ssim_fitness
        return (fitness,)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
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
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    try:
        for gen in range(generations):
            pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, 
                                               ngen=1, stats=stats, 
                                               halloffame=hof, verbose=False)
            
            current_best = hof[0].fitness.values[0]
            
            if current_best > best_fitness:
                best_fitness = current_best
                last_improvement = gen
            elif gen - last_improvement > 10:  # Early stopping if no improvement for 10 generations
                break
            
            if time.time() - start_time > timeout:
                break

    except TimeoutError:
        print("GA optimization timed out")
    except Exception as e:
        print(f"An error occurred during GA optimization: {e}")

    best_ind = tools.selBest(pop, 1)[0]
    return cp.array([int(w) for w in best_ind], dtype=cp.int32), best_ind.fitness.values

# Clean up DEAP globals
del creator.FitnessMax
del creator.Individual

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

