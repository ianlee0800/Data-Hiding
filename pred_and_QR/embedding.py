from numba.core.errors import NumbaPerformanceWarning
import warnings
import numpy as np
import cupy as cp
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

warnings.filterwarnings("ignore", category=RuntimeWarning, module="deap.creator")
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

if 'FitnessMax' not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
if 'Individual' not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

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

def pee_process_with_rotation_cuda(img, total_embeddings, ratio_of_ones, use_different_weights, max_el=7):
    original_img = cp.asarray(img)
    current_img = original_img.copy()
    height, width = img.shape
    total_pixels = height * width
    pee_stages = []
    total_payload = 0
    rotation_images = []
    rotation_histograms = []

    original_hist = cp.histogram(original_img, bins=256, range=(0, 255))[0]

    for embedding in range(total_embeddings):
        print(f"\nStarting embedding {embedding}")
        
        sub_images = split_image(current_img)
        
        stage_info = {
            'embedding': embedding,
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
            sub_data = generate_random_binary_array(sub_img.size, ratio_of_ones)
            sub_data = cp.asarray(sub_data, dtype=cp.uint8)
            
            current_psnr = calculate_psnr(cp.asnumpy(original_img), cp.asnumpy(current_img))
            
            EL = min(max_el, max(1, int(current_psnr / 5)))
            
            if use_different_weights or (embedding == 0 and i == 0):
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
            
            print(f"  Sub-image {i} - PSNR: {sub_psnr:.2f}, SSIM: {sub_ssim:.4f}, Hist Corr: {sub_hist_corr:.4f}")
            
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
    
        stage_info['psnr'] = float(calculate_psnr(cp.asnumpy(original_img), cp.asnumpy(embedded_img)))
        stage_info['ssim'] = float(calculate_ssim(cp.asnumpy(original_img), cp.asnumpy(embedded_img)))
        stage_info['hist_corr'] = float(histogram_correlation(
            cp.asnumpy(original_hist),
            cp.asnumpy(cp.histogram(embedded_img, bins=256, range=(0, 255))[0])
        ))
        stage_info['bpp'] = float(stage_info['payload'] / total_pixels)
        
        pee_stages.append(stage_info)
        total_payload += stage_info['payload']

        print(f"Embedding {embedding} summary:")
        print(f"  Payload: {stage_info['payload']}")
        print(f"  PSNR: {stage_info['psnr']:.2f}")
        print(f"  SSIM: {stage_info['ssim']:.4f}")
        print(f"  Hist Corr: {stage_info['hist_corr']:.4f}")

        if embedding < total_embeddings - 1:
            current_img = cp.rot90(embedded_img)
        else:
            current_img = embedded_img

        rotation_images.append(cp.asnumpy(current_img))
        rotation_histograms.append(cp.asnumpy(cp.histogram(current_img, bins=256, range=(0, 255))[0]))

    final_img = cp.asnumpy(current_img)
    return final_img, int(total_payload), pee_stages, rotation_images, rotation_histograms, total_embeddings

def pee_process_with_split_cuda(img, total_embeddings, ratio_of_ones, use_different_weights, max_el=7):
    original_img = cp.asarray(img)
    height, width = img.shape
    total_pixels = height * width
    pee_stages = []
    total_payload = 0

    sub_images = split_image(original_img)
    sub_rotations = [0, 0, 0, 0]  # 初始化每个子图像的旋转次数
    
    for embedding in range(total_embeddings):
        print(f"\nStarting embedding {embedding}")
        
        stage_info = {
            'embedding': embedding,
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
            if sub_rotations[i] < 4:  # 只有在未达到最大旋转次数时才进行旋转
                sub_img = cp.rot90(sub_img)
                sub_rotations[i] += 1
            
            sub_data = generate_random_binary_array(sub_img.size, ratio_of_ones)
            sub_data = cp.asarray(sub_data, dtype=cp.uint8)
            
            original_sub = original_img[i//2::2, i%2::2]
            current_psnr = calculate_psnr(cp.asnumpy(original_sub), cp.asnumpy(sub_img))
            
            EL = min(max_el, max(1, int(current_psnr / 5)))
            
            if use_different_weights or (embedding == 0 and i == 0):
                weights, fitness = find_best_weights_ga_cuda(sub_img, sub_data, EL)
            
            pred_sub_img = improved_predict_image_cuda(sub_img, weights)
            embedded_sub, payload, _ = pee_embedding_adaptive_cuda(sub_img, sub_data, pred_sub_img, EL)
            
            embedded_sub_images.append(embedded_sub)
            stage_info['payload'] += payload
            
            sub_psnr = calculate_psnr(cp.asnumpy(original_sub), cp.asnumpy(embedded_sub))
            sub_ssim = calculate_ssim(cp.asnumpy(original_sub), cp.asnumpy(embedded_sub))
            sub_hist_corr = histogram_correlation(
                cp.asnumpy(cp.histogram(original_sub, bins=256, range=(0, 255))[0]),
                cp.asnumpy(cp.histogram(embedded_sub, bins=256, range=(0, 255))[0])
            )
            
            print(f"  Sub-image {i} - PSNR: {sub_psnr:.2f}, SSIM: {sub_ssim:.4f}, Hist Corr: {sub_hist_corr:.4f}")
            
            block_info = {
                'weights': cp.asnumpy(weights).tolist(),
                'EL': int(EL),
                'payload': int(payload),
                'psnr': float(sub_psnr),
                'ssim': float(sub_ssim),
                'hist_corr': float(sub_hist_corr),
                'rotation': sub_rotations[i] * 90  # 记录旋转角度
            }
            stage_info['sub_images'].append(block_info)
            stage_info['block_params'].append(block_info)

        embedded_img = merge_image(embedded_sub_images)
    
        stage_info['psnr'] = float(calculate_psnr(cp.asnumpy(original_img), cp.asnumpy(embedded_img)))
        stage_info['ssim'] = float(calculate_ssim(cp.asnumpy(original_img), cp.asnumpy(embedded_img)))
        stage_info['hist_corr'] = float(histogram_correlation(
            cp.asnumpy(cp.histogram(original_img, bins=256, range=(0, 255))[0]),
            cp.asnumpy(cp.histogram(embedded_img, bins=256, range=(0, 255))[0])
        ))
        stage_info['bpp'] = float(stage_info['payload'] / total_pixels)
        
        pee_stages.append(stage_info)
        total_payload += stage_info['payload']

        print(f"Embedding {embedding} summary:")
        print(f"  Payload: {stage_info['payload']}")
        print(f"  PSNR: {stage_info['psnr']:.2f}")
        print(f"  SSIM: {stage_info['ssim']:.4f}")
        print(f"  Hist Corr: {stage_info['hist_corr']:.4f}")

        sub_images = embedded_sub_images

    final_img = cp.asnumpy(merge_image(sub_images))
    
    return final_img, int(total_payload), pee_stages, sub_images, sub_rotations

def histogram_data_hiding(img, pee_info_bits, ratio_of_ones=1):
    print(f"HS Input - Max pixel value: {np.max(img)}")
    print(f"HS Input - Min pixel value: {np.min(img)}")
    h_img, w_img = img.shape
    markedImg = img.copy()
    total_payload = 0
    rounds = 0
    payloads = []

    pee_info_length = len(pee_info_bits)

    # 创建一个掩码来跟踪已经用于嵌入的像素
    embedded_mask = np.zeros_like(markedImg, dtype=bool)

    while np.max(markedImg) < 255:
        rounds += 1
        hist = np.bincount(markedImg[~embedded_mask].ravel(), minlength=256)
        
        print(f"\nRound {rounds}:")
        print(f"Histogram shape: {hist.shape}")
        
        peak = np.argmax(hist[:-1])  # Avoid selecting 255 as peak
        print(f"Histogram peak: {peak}, value: {hist[peak]}")
        
        print(f"Histogram around peak:")
        for i in range(max(0, peak-5), min(256, peak+6)):
            print(f"  Pixel value {i}: {hist[i]}")
        
        max_payload = hist[peak]
        
        if max_payload == 0:
            print("No more available peak values. Stopping embedding.")
            break
        
        if pee_info_length > 0:
            embedding_data = pee_info_bits[:max_payload]
            pee_info_bits = pee_info_bits[max_payload:]
            pee_info_length -= len(embedding_data)
            if len(embedding_data) < max_payload:
                random_bits = generate_random_binary_array(max_payload - len(embedding_data), ratio_of_ones)
                embedding_data += ''.join(map(str, random_bits))
        else:
            embedding_data = ''.join(map(str, generate_random_binary_array(max_payload, ratio_of_ones)))
        
        actual_payload = len(embedding_data)
        
        embedded_count = 0
        modified_count = 0
        
        # 创建一个掩码，标记所有需要移动的像素
        move_mask = (markedImg > peak) & (~embedded_mask)
        
        # 移动所有大于峰值的未嵌入像素
        markedImg[move_mask] += 1
        modified_count += np.sum(move_mask)
        
        # 嵌入数据到峰值像素
        peak_pixels = np.where((markedImg == peak) & (~embedded_mask))
        for i in range(min(len(peak_pixels[0]), actual_payload)):
            y, x = peak_pixels[0][i], peak_pixels[1][i]
            markedImg[y, x] += int(embedding_data[i])
            embedded_mask[y, x] = True
            embedded_count += 1
            modified_count += 1
        
        total_payload += actual_payload
        payloads.append(actual_payload)
        
        print(f"Embedded {actual_payload} bits")
        print(f"Modified {modified_count} pixels")
        print(f"Remaining PEE info: {pee_info_length} bits")
        print(f"Current max pixel value: {np.max(markedImg)}")
        print(f"Current min pixel value: {np.min(markedImg)}")
        
        hist_after = np.bincount(markedImg.ravel(), minlength=256)
        print(f"Histogram after embedding:")
        for i in range(max(0, peak-5), min(256, peak+7)):
            print(f"  Pixel value {i}: {hist_after[i]}")

    print(f"Final max pixel value: {np.max(markedImg)}")
    print(f"Final min pixel value: {np.min(markedImg)}")
    print(f"Total rounds: {rounds}")
    print(f"Total payload: {total_payload}")

    return markedImg, total_payload, payloads, rounds

# 測試函數
def test_pee_process():
    # 創建測試數據
    img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    total_embeddings = 3
    ratio_of_ones = 0.5
    use_different_weights = True
    max_el = 7

    print("Testing pee_process_with_rotation_cuda:")
    final_img_rotation, total_payload_rotation, pee_stages_rotation, rotation_images, rotation_histograms, actual_embeddings = pee_process_with_rotation_cuda(
        img, total_embeddings, ratio_of_ones, use_different_weights, max_el
    )

    print(f"\nRotation method summary:")
    print(f"Total payload: {total_payload_rotation}")
    print(f"Number of PEE stages: {len(pee_stages_rotation)}")
    for i, stage in enumerate(pee_stages_rotation):
        print(f"\nStage {i}:")
        print(f"  Payload: {stage['payload']}")
        print(f"  PSNR: {stage['psnr']:.2f}")
        print(f"  SSIM: {stage['ssim']:.4f}")
        print(f"  Histogram Correlation: {stage['hist_corr']:.4f}")
        print(f"  BPP: {stage['bpp']:.4f}")

    print("\nTesting pee_process_with_split_cuda:")
    final_img_split, total_payload_split, pee_stages_split, sub_images, sub_rotations = pee_process_with_split_cuda(
        img, total_embeddings, ratio_of_ones, use_different_weights, max_el
    )

    print(f"\nSplit method summary:")
    print(f"Total payload: {total_payload_split}")
    print(f"Number of PEE stages: {len(pee_stages_split)}")
    print(f"Number of sub-images: {len(sub_images)}")
    print(f"Sub-rotation values: {sub_rotations}")
    for i, stage in enumerate(pee_stages_split):
        print(f"\nStage {i}:")
        print(f"  Payload: {stage['payload']}")
        print(f"  PSNR: {stage['psnr']:.2f}")
        print(f"  SSIM: {stage['ssim']:.4f}")
        print(f"  Histogram Correlation: {stage['hist_corr']:.4f}")
        print(f"  BPP: {stage['bpp']:.4f}")

    # 比較兩種方法的最終性能
    original_hist = np.histogram(img, bins=256, range=(0, 255))[0]
    
    psnr_rotation = calculate_psnr(img, final_img_rotation)
    ssim_rotation = calculate_ssim(img, final_img_rotation)
    hist_corr_rotation = histogram_correlation(original_hist,
                                               np.histogram(final_img_rotation, bins=256, range=(0, 255))[0])

    psnr_split = calculate_psnr(img, final_img_split)
    ssim_split = calculate_ssim(img, final_img_split)
    hist_corr_split = histogram_correlation(original_hist,
                                            np.histogram(final_img_split, bins=256, range=(0, 255))[0])

    print("\nFinal Performance Comparison:")
    print(f"Rotation method - PSNR: {psnr_rotation:.2f}, SSIM: {ssim_rotation:.4f}, Hist Corr: {hist_corr_rotation:.4f}")
    print(f"Split method    - PSNR: {psnr_split:.2f}, SSIM: {ssim_split:.4f}, Hist Corr: {hist_corr_split:.4f}")

    print("\nDiagnostic Information:")
    print(f"Original image shape: {img.shape}")
    print(f"Final rotation image shape: {final_img_rotation.shape}")
    print(f"Final split image shape: {final_img_split.shape}")
    print(f"Original image min/max: {np.min(img)}/{np.max(img)}")
    print(f"Final rotation image min/max: {np.min(final_img_rotation)}/{np.max(final_img_rotation)}")
    print(f"Final split image min/max: {np.min(final_img_split)}/{np.max(final_img_split)}")

    # 添加直方圖比較
    print("\nHistogram Comparison:")
    print("Original Histogram:", original_hist[:10], "...")  # 只顯示前10個值
    print("Final Rotation Histogram:", np.histogram(final_img_rotation, bins=256, range=(0, 255))[0][:10], "...")
    print("Final Split Histogram:", np.histogram(final_img_split, bins=256, range=(0, 255))[0][:10], "...")

    # 計算和顯示最終的 BPP (Bits Per Pixel)
    bpp_rotation = total_payload_rotation / (img.shape[0] * img.shape[1])
    bpp_split = total_payload_split / (img.shape[0] * img.shape[1])
    print(f"\nFinal BPP (Bits Per Pixel):")
    print(f"Rotation method: {bpp_rotation:.4f}")
    print(f"Split method: {bpp_split:.4f}")

    # 可視化結果（如果需要的話）
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.subplot(132)
    plt.imshow(final_img_rotation, cmap='gray')
    plt.title("Rotation Method Result")
    plt.subplot(133)
    plt.imshow(final_img_split, cmap='gray')
    plt.title("Split Method Result")
    plt.show()

if __name__ == "__main__":
    test_pee_process()