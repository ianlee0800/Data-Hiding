import numpy as np
import cupy as cp
import struct
from image_processing import (
    generate_histogram,
    merge_image,
    split_image,
    save_histogram
)
from utils import (
    find_best_weights_ga,
    find_best_weights_ga_cuda,
    generate_random_binary_array
)
from common import *
from pee import *
import random
from deap import base, creator, tools

def pee_process_with_rotation(img, total_rotations, ratio_of_ones):
    height, width = img.shape
    pee_stages = []
    total_payload = 0
    current_img = img.copy()
    
    # 选择实现（CPU 版本）
    _, embed_func, predict_func = choose_pee_implementation(use_cuda=False)

    # 创建 toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 1, 15)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    for rotation in range(total_rotations):
        print(f"Starting rotation {rotation}")
        
        # 将图像分割成四个子图像
        sub_images = split_image(current_img)
        embedded_sub_images = []
        sub_payloads = []
        sub_psnrs = []
        sub_ssims = []
        
        for i, sub_img in enumerate(sub_images):
            # 生成随机数据
            sub_data = np.random.binomial(1, ratio_of_ones, sub_img.size).astype(np.uint8)
            
            # 随机选择 EL
            EL = np.random.choice([1, 3, 5, 7])
            
            # 找到最佳权重
            weights, fitness = find_best_weights_ga(sub_img, sub_data, EL, toolbox)
            
            # 生成预测图像
            pred_sub_img = predict_func(sub_img, weights)
            
            # 执行 PEE 嵌入
            embedded_sub, payload, _ = embed_func(sub_img, sub_data, pred_sub_img, EL)
            
            embedded_sub_images.append(embedded_sub)
            sub_payloads.append(payload)
            
            # 计算 PSNR 和 SSIM
            sub_psnr = calculate_psnr(sub_img, embedded_sub)
            sub_ssim = calculate_ssim(sub_img, embedded_sub)
            sub_psnrs.append(sub_psnr)
            sub_ssims.append(sub_ssim)
        
        # 合并子图像
        embedded_img = merge_image(embedded_sub_images)
    
        
        # 计算整体 PSNR 和 SSIM
        psnr = calculate_psnr(current_img, embedded_img)
        ssim = calculate_ssim(current_img, embedded_img)
        
        pee_stages.append({
            'rotation': rotation,
            'payload': sum(sub_payloads),
            'psnr': psnr,
            'ssim': ssim,
            'sub_payloads': sub_payloads,
            'sub_psnrs': sub_psnrs,
            'sub_ssims': sub_ssims
        })
        
        total_payload += sum(sub_payloads)
        current_img = np.rot90(embedded_img)

    return current_img, total_payload, pee_stages

def pee_process_with_rotation_cuda(img, total_rotations, ratio_of_ones, target_payload=480000):
    original_img = cp.asarray(img)
    current_img = original_img.copy()
    height, width = img.shape
    total_pixels = height * width
    pee_stages = []
    total_payload = 0
    rotation_images = []
    rotation_histograms = []

    # 计算原始图像的直方图
    original_hist = cp.histogram(original_img, bins=256, range=(0, 255))[0]

    for rotation in range(total_rotations):
        print(f"Starting rotation {rotation}")
        
        sub_images = split_image(current_img)
        
        stage_info = {
            'rotation': rotation,
            'sub_images': [],
            'payload': 0,
            'psnr': 0,
            'ssim': 0,
            'hist_corr': 0,
            'bpp': 0,
            'block_params': []
        }
        
        embedded_sub_images = []
        for i, sub_img in enumerate(sub_images):
            sub_img = cp.asarray(sub_img)
            # 使用 generate_random_binary_array 函数生成随机数据
            sub_data = generate_random_binary_array(sub_img.size, ratio_of_ones)
            sub_data = cp.asarray(sub_data, dtype=cp.uint8)
            
            # 计算当前的 PSNR
            current_psnr = calculate_psnr(cp.asnumpy(original_img), cp.asnumpy(current_img))
            
            # 使用 choose_el_for_rotation 函数
            EL = choose_el_for_rotation(current_psnr, total_payload, total_pixels, rotation, total_rotations, target_payload)
            
            weights, fitness = find_best_weights_ga_cuda(sub_img, sub_data, EL)
            pred_sub_img = improved_predict_image_cuda(sub_img, weights)
            embedded_sub, payload, _ = pee_embedding_adaptive_cuda(sub_img, sub_data, pred_sub_img, EL)
            
            embedded_sub_images.append(embedded_sub)
            stage_info['payload'] += payload
            
            sub_psnr = calculate_psnr(cp.asnumpy(sub_img), cp.asnumpy(embedded_sub))
            sub_ssim = calculate_ssim(cp.asnumpy(sub_img), cp.asnumpy(embedded_sub))
            sub_hist_corr = histogram_correlation(
                cp.asnumpy(cp.histogram(sub_img, bins=256, range=(0, 255))[0]),
                cp.asnumpy(cp.histogram(embedded_sub, bins=256, range=(0, 255))[0])
            )
            
            block_info = {
                'weights': cp.asnumpy(weights).tolist(),
                'EL': int(EL),
                'payload': int(payload),
                'psnr': float(sub_psnr),
                'ssim': float(sub_ssim),
                'hist_corr': float(sub_hist_corr)
            }
            stage_info['sub_images'].append(block_info)
            stage_info['block_params'].append(block_info)

        embedded_img = merge_image(embedded_sub_images)
        
        # 將嵌入後的圖像旋轉回原始方向
        rotated_back_img = cp.rot90(embedded_img, k=-rotation)
        
        # 計算 PSNR、SSIM 和直方圖相關性
        stage_info['psnr'] = float(calculate_psnr(cp.asnumpy(original_img), cp.asnumpy(rotated_back_img)))
        stage_info['ssim'] = float(calculate_ssim(cp.asnumpy(original_img), cp.asnumpy(rotated_back_img)))
        stage_info['hist_corr'] = float(histogram_correlation(
            cp.asnumpy(original_hist),
            cp.asnumpy(cp.histogram(rotated_back_img, bins=256, range=(0, 255))[0])
        ))
        stage_info['bpp'] = float(stage_info['payload'] / total_pixels)
        
        pee_stages.append(stage_info)
        total_payload += stage_info['payload']

        if np.array_equal(cp.asnumpy(current_img), cp.asnumpy(embedded_img)):
            print(f"Warning: No change in image after rotation {rotation}")
        
        current_img = cp.rot90(embedded_img)
        rotation_images.append(cp.asnumpy(embedded_img))
        rotation_histograms.append(cp.asnumpy(cp.histogram(rotated_back_img, bins=256, range=(0, 255))[0]))

    print("\nPEE process summary:")
    for i, stage in enumerate(pee_stages):
        print(f"Rotation {i}:")
        print(f"  Total payload: {stage['payload']}")
        print(f"  PSNR: {stage['psnr']:.2f}")
        print(f"  SSIM: {stage['ssim']:.4f}")
        print(f"  BPP: {stage['bpp']:.4f}")

    final_img = cp.asnumpy(current_img)
    return final_img, int(total_payload), pee_stages, rotation_images, rotation_histograms, total_rotations

def histogram_data_hiding(img, encoded_pee_info, ratio_of_ones=0.5):
    h_img, w_img = img.shape
    markedImg = img.copy()
    total_payload = 0
    rounds = 0
    payloads = []

    print(f"Initial image shape: {markedImg.shape}")
    print(f"Initial image dtype: {markedImg.dtype}")
    print(f"Initial max pixel value: {np.max(markedImg)}")
    print(f"Initial min pixel value: {np.min(markedImg)}")

    try:
        while np.max(markedImg) < 255:
            rounds += 1
            hist, min_val, max_val, hist_len = generate_histogram(markedImg)
            
            print(f"\nRound {rounds}:")
            print(f"Histogram shape: {hist.shape}")
            print(f"Histogram dtype: {hist.dtype}")
            print(f"Histogram min: {min_val}, max: {max_val}, length: {hist_len}")
            
            peak = find_max(hist)
            
            print(f"Histogram peak: {peak}, value: {hist[peak]}")
            print(f"Histogram sample: {hist[max(0, peak-5):min(256, peak+6)]}")
            
            max_payload = hist[peak]
            
            if max_payload == 0:
                print("No more available peak values. Stopping embedding.")
                break
            
            if rounds == 1:
                embedding_data = ''.join(format(byte, '08b') for byte in encoded_pee_info)
                if len(embedding_data) < max_payload:
                    random_bits = generate_random_binary_array(max_payload - len(embedding_data), ratio_of_ones)
                    embedding_data += ''.join(map(str, random_bits))
            else:
                embedding_data = ''.join(map(str, generate_random_binary_array(max_payload, ratio_of_ones)))
            
            embedding_data = embedding_data[:max_payload]
            actual_payload = len(embedding_data)
            
            embedded_count = 0
            for y in range(h_img):
                for x in range(w_img):
                    if embedded_count < actual_payload and markedImg[y, x] == peak:
                        markedImg[y, x] += int(embedding_data[embedded_count])
                        embedded_count += 1
                    elif markedImg[y, x] > peak:
                        markedImg[y, x] += 1
            
            total_payload += actual_payload
            payloads.append(actual_payload)
            
            print(f"Embedded {actual_payload} bits")
            print(f"Current max pixel value: {np.max(markedImg)}")
            print(f"Current min pixel value: {np.min(markedImg)}")

        print(f"Final max pixel value: {np.max(markedImg)}")
        print(f"Final min pixel value: {np.min(markedImg)}")
        print(f"Total rounds: {rounds}")
        print(f"Total payload: {total_payload}")

        return markedImg, total_payload, payloads, rounds
    except Exception as e:
        print(f"Error in histogram_data_hiding: {e}")
        import traceback
        traceback.print_exc()
        return markedImg, total_payload, payloads, rounds

def choose_pee_implementation(use_cuda=True):
    if use_cuda and cp.cuda.is_available():
        return pee_process_with_rotation_cuda, pee_embedding_adaptive_cuda, improved_predict_image_cuda
    else:
        return pee_process_with_rotation, pee_embedding_adaptive, improved_predict_image_cpu

# 測試函數
def test_pee_process():
    # 創建測試數據
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    total_rotations = 2
    ratio_of_ones = 0.5

    final_img, total_payload, pee_stages = pee_process_with_rotation_cuda(img, total_rotations, ratio_of_ones)

    print(f"Total payload: {total_payload}")
    print(f"Number of PEE stages: {len(pee_stages)}")
    for stage in pee_stages:
        print(f"Rotation {stage['rotation']}:")
        print(f"  Payload: {stage['payload']}")
        print(f"  PSNR: {stage['psnr']:.2f}")
        print(f"  SSIM: {stage['ssim']:.4f}")
        print(f"  Sub-image payloads: {stage['sub_payloads']}")
        print(f"  Sub-image PSNRs: {[f'{psnr:.2f}' for psnr in stage['sub_psnrs']]}")
        print(f"  Sub-image SSIMs: {[f'{ssim:.4f}' for ssim in stage['sub_ssims']]}")

if __name__ == "__main__":
    test_pee_process()