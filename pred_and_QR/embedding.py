import numpy as np
import cupy as cp
from image_processing import (
    generate_histogram,
    merge_image,
    split_image
)
from utils import (
    decode_pee_info,
    find_best_weights_ga,
    find_best_weights_ga_cuda
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

def pee_process_with_rotation_cuda(img, total_rotations, ratio_of_ones):
    height, width = img.shape
    pee_stages = []
    total_payload = 0
    current_img = cp.asarray(img)
    original_hist = cp.histogram(current_img, bins=256, range=(0, 256))[0]

    # 获取必要的函数
    _, embed_func, predict_func = choose_pee_implementation(use_cuda=True)

    for rotation in range(total_rotations):
        print(f"\nProcessing rotation {rotation}")
        
        sub_images = split_image(current_img)
        embedded_sub_images = []
        stage_info = {
            'rotation': rotation,
            'sub_images': [],
            'payload': 0,
            'psnr': 0,
            'ssim': 0,
            'hist_corr': 0,
            'bpp': 0
        }
        
        for i, sub_img in enumerate(sub_images):
            sub_data = cp.random.binomial(1, ratio_of_ones, sub_img.size).astype(cp.uint8)
            EL = int(cp.random.choice([1, 3, 5, 7], size=1)[0])  # 修改这一行
            
            weights, fitness = find_best_weights_ga_cuda(sub_img, sub_data, EL)
            
            pred_sub_img = predict_func(sub_img, weights)
            embedded_sub, payload, _ = embed_func(sub_img, sub_data, pred_sub_img, EL)
            
            embedded_sub_images.append(embedded_sub)
            
            sub_psnr = calculate_psnr(to_cpu(sub_img), to_cpu(embedded_sub))
            sub_ssim = calculate_ssim(to_cpu(sub_img), to_cpu(embedded_sub))
            sub_hist_corr = histogram_correlation(
                cp.histogram(sub_img, bins=256, range=(0, 256))[0],
                cp.histogram(embedded_sub, bins=256, range=(0, 256))[0]
            )
            
            sub_info = {
                'weights': weights.get().tolist(),
                'EL': EL,
                'payload': int(payload),
                'psnr': float(sub_psnr),
                'ssim': float(sub_ssim),
                'hist_corr': float(sub_hist_corr)
            }
            stage_info['sub_images'].append(sub_info)
            stage_info['payload'] += payload
        
        embedded_img = merge_image(embedded_sub_images)
        
        stage_info['psnr'] = float(calculate_psnr(to_cpu(current_img), to_cpu(embedded_img)))
        stage_info['ssim'] = float(calculate_ssim(to_cpu(current_img), to_cpu(embedded_img)))
        stage_info['hist_corr'] = float(histogram_correlation(
            original_hist,
            cp.histogram(embedded_img, bins=256, range=(0, 256))[0]
        ))
        stage_info['bpp'] = float(stage_info['payload'] / (height * width))
        
        pee_stages.append(stage_info)
        total_payload += stage_info['payload']
        
        current_img = cp.rot90(embedded_img)

    final_img = to_cpu(current_img)
    return final_img, int(total_payload), pee_stages
    
def histogram_data_hiding(img, flag, encoded_pee_info):
    """Histogram Shifting Embedding"""
    # 確保輸入是 NumPy 陣列
    if isinstance(img, cp.ndarray):
        img = cp.asnumpy(img)

    h_img, w_img = img.shape
    markedImg = img.copy()
    hist, _, _, _ = generate_histogram(img)
    
    print("Debug: Histogram in histogram_data_hiding")
    print(f"Histogram type: {type(hist)}")
    print(f"Histogram length: {len(hist)}")
    print(f"Histogram sample: {hist[:10]}")
    
    peak = find_max(hist)
    print(f"Peak found in histogram_data_hiding: {peak}")
    
    try:
        # 解碼 PEE 資訊
        total_rotations, pee_stages = decode_pee_info(encoded_pee_info)
    except (TypeError, ValueError) as e:
        print(f"Error decoding PEE info: {e}")
        print(f"Encoded PEE info type: {type(encoded_pee_info)}")
        print(f"Encoded PEE info length: {len(encoded_pee_info)} bytes")
        print(f"Encoded PEE info content: {encoded_pee_info}")
        raise

    # 計算最大可嵌入載荷
    max_payload = hist[peak]
    
    # 將 PEE 資訊轉換為二進制字符串
    embedding_data = ''
    embedding_data += format(total_rotations, '08b')
    for stage in pee_stages:
        for block in stage['block_params']:
            weights = block['weights']
            EL = block['EL']
            payload = block['payload']
            for w in weights:
                # 將浮點數轉換為整數（乘以100並四捨五入），然後轉換為16位二進制
                embedding_data += format(int(round(w * 100)), '016b')
            embedding_data += format(EL, '08b')
            embedding_data += format(payload, '032b')
    
    # 確保我們不超過最大載荷
    embedding_data = embedding_data[:max_payload]
    actual_payload = len(embedding_data)
    
    print(f"Debug: Embedding data length: {actual_payload} bits")
    print(f"Debug: Max payload: {max_payload} bits")
    
    i = 0
    for y in range(h_img):
        for x in range(w_img):
            if i < actual_payload and markedImg[y, x] == peak:
                if flag == 0:
                    markedImg[y, x] -= int(embedding_data[i])
                elif flag == 1:
                    markedImg[y, x] += int(embedding_data[i])
                i += 1
            elif markedImg[y, x] > peak:
                if flag == 1:
                    markedImg[y, x] += 1
            elif markedImg[y, x] < peak:
                if flag == 0:
                    markedImg[y, x] -= 1
    
    print(f"Debug: Actually embedded {i} bits")
    
    return markedImg, peak, actual_payload

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

    final_img, total_payload, pee_stages = pee_process_with_rotation(img, total_rotations, ratio_of_ones)

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