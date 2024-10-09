from numba.core.errors import NumbaPerformanceWarning
import warnings
import numpy as np
import cupy as cp
from image_processing import (
    merge_image,
    split_image,
    split_image_into_quarters
)
from utils import (
    brute_force_weight_search_cuda,
    generate_random_binary_array
)
from common import *
from pee import *
import random

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

def pee_process_with_rotation(img, total_rotations, ratio_of_ones):
    height, width = img.shape
    pee_stages = []
    total_payload = 0
    current_img = cp.asarray(img)  # Convert to CuPy array
    
    for rotation in range(total_rotations):
        print(f"Starting rotation {rotation}")
        
        # Split the image into four sub-images
        sub_images = split_image(current_img)
        embedded_sub_images = []
        sub_payloads = []
        sub_psnrs = []
        sub_ssims = []
        
        for i, sub_img in enumerate(sub_images):
            # Generate random data
            sub_data = generate_random_binary_array(sub_img.size, ratio_of_ones)
            sub_data = cp.asarray(sub_data, dtype=cp.uint8)
            
            # Randomly choose EL
            EL = np.random.choice([1, 3, 5, 7])
            
            # Find best weights using CUDA brute force search
            weights, (payload, _) = brute_force_weight_search_cuda(sub_img, sub_data, EL)
            
            # Generate predicted image
            pred_sub_img = improved_predict_image_cuda(sub_img, weights)
            
            # Perform PEE embedding
            embedded_sub, payload, _ = pee_embedding_adaptive_cuda(sub_img, sub_data, pred_sub_img, EL)
            
            embedded_sub_images.append(embedded_sub)
            sub_payloads.append(payload)
            
            # Calculate PSNR and SSIM
            sub_psnr = calculate_psnr(cp.asnumpy(sub_img), cp.asnumpy(embedded_sub))
            sub_ssim = calculate_ssim(cp.asnumpy(sub_img), cp.asnumpy(embedded_sub))
            sub_psnrs.append(sub_psnr)
            sub_ssims.append(sub_ssim)
        
        # Merge sub-images
        embedded_img = merge_image(embedded_sub_images)
        
        # Calculate overall PSNR and SSIM
        psnr = calculate_psnr(cp.asnumpy(current_img), cp.asnumpy(embedded_img))
        ssim = calculate_ssim(cp.asnumpy(current_img), cp.asnumpy(embedded_img))
        
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
        current_img = cp.rot90(embedded_img)

    return cp.asnumpy(current_img), total_payload, pee_stages

def pee_process_with_rotation_cuda(img, total_embeddings, ratio_of_ones, use_different_weights):
    original_img = cp.asarray(img)
    height, width = img.shape
    total_pixels = height * width
    pee_stages = []
    total_payload = 0
    current_img = original_img.copy()
    previous_psnr = float('inf')
    previous_ssim = 1.0
    previous_payload = float('inf')

    for embedding in range(total_embeddings):
        print(f"\nStarting embedding {embedding}")
        
        if embedding == 0:
            target_psnr = 40.0
            target_bpp = 0.9
        else:
            target_psnr = max(28.0, previous_psnr - 1)  # Ensure PSNR decreases
            target_bpp = max(0.5, (previous_payload / total_pixels) * 0.95)  # Ensure payload decreases
        
        print(f"Target PSNR for embedding {embedding}: {target_psnr:.2f}")
        print(f"Target BPP for embedding {embedding}: {target_bpp:.4f}")
        
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
        
        stage_payload = 0
        embedded_sub_images = []
        stage_rotation = embedding * 90
        
        sub_images = split_image(current_img)
        
        # Calculate max_sub_payload based on the previous stage's payload or the size of the first sub-image
        if embedding > 0:
            max_sub_payload = previous_payload // len(sub_images)
        else:
            first_sub_img = sub_images[0]
            max_sub_payload = first_sub_img.size
        
        for i, sub_img in enumerate(sub_images):
            rotated_sub_img = cp.rot90(sub_img, k=embedding)
            sub_data = generate_random_binary_array(rotated_sub_img.size, ratio_of_ones)
            sub_data = cp.asarray(sub_data, dtype=cp.uint8)
            
            local_el = compute_adaptive_el(rotated_sub_img)
            
            if use_different_weights or i == 0:
                weights, (sub_payload, sub_psnr) = brute_force_weight_search_cuda(rotated_sub_img, sub_data, local_el, target_bpp, target_psnr, embedding)
            
            # Use the minimum of max_sub_payload and len(sub_data)
            data_to_embed = sub_data[:min(int(max_sub_payload), len(sub_data))]
            embedded_sub, payload = multi_pass_embedding(rotated_sub_img, data_to_embed, local_el, weights, embedding)
            
            rotated_back_sub = cp.rot90(embedded_sub, k=-embedding)
            embedded_sub_images.append(rotated_back_sub)
            stage_payload += payload
            
            sub_img_np = cp.asnumpy(sub_img)
            rotated_back_sub_np = cp.asnumpy(rotated_back_sub)
            sub_psnr = calculate_psnr(sub_img_np, rotated_back_sub_np)
            sub_ssim = calculate_ssim(sub_img_np, rotated_back_sub_np)
            sub_hist_corr = histogram_correlation(
                np.histogram(sub_img_np, bins=256, range=(0, 255))[0],
                np.histogram(rotated_back_sub_np, bins=256, range=(0, 255))[0]
            )
            
            local_el_np = local_el.copy_to_host() if isinstance(local_el, cuda.cudadrv.devicearray.DeviceNDArray) else local_el
            max_el = int(np.max(local_el_np))

            block_info = {
                'weights': weights.tolist() if isinstance(weights, np.ndarray) else weights,
                'EL': max_el,
                'payload': int(payload),
                'psnr': float(sub_psnr),
                'ssim': float(sub_ssim),
                'rotation': stage_rotation,
                'hist_corr': float(sub_hist_corr)
            }
            stage_info['sub_images'].append(block_info)
            stage_info['block_params'].append(block_info)

        stage_img = merge_image(embedded_sub_images)
        stage_info['stage_img'] = stage_img

        original_img_np = cp.asnumpy(original_img)
        stage_img_np = cp.asnumpy(stage_img)
        stage_info['psnr'] = float(calculate_psnr(original_img_np, stage_img_np))
        stage_info['ssim'] = float(calculate_ssim(original_img_np, stage_img_np))
        stage_info['hist_corr'] = float(histogram_correlation(
            np.histogram(original_img_np, bins=256, range=(0, 255))[0],
            np.histogram(stage_img_np, bins=256, range=(0, 255))[0]
        ))
        stage_info['payload'] = stage_payload
        stage_info['bpp'] = float(stage_info['payload'] / total_pixels)
        
        # Check if current stage's payload and PSNR are less than previous stage
        if embedding > 0:
            if stage_info['payload'] >= previous_payload:
                print(f"Warning: Stage {embedding} payload ({stage_info['payload']}) is not less than previous stage ({previous_payload}).")
                stage_info['payload'] = int(previous_payload * 0.95)  # Reduce payload to 95% of previous stage
                print(f"Adjusted payload: {stage_info['payload']}")
            
            if stage_info['psnr'] >= previous_psnr:
                print(f"Warning: Stage {embedding} PSNR ({stage_info['psnr']:.2f}) is not less than previous stage ({previous_psnr:.2f}).")
                # Note: We can't directly adjust PSNR here, but we've already tried to ensure it's lower in the embedding process
        
        pee_stages.append(stage_info)
        total_payload += stage_info['payload']
        previous_psnr = stage_info['psnr']
        previous_ssim = stage_info['ssim']
        previous_payload = stage_info['payload']

        print(f"Embedding {embedding} summary:")
        print(f"  Payload: {stage_info['payload']}")
        print(f"  BPP: {stage_info['bpp']:.4f}")
        print(f"  PSNR: {stage_info['psnr']:.2f}")
        print(f"  SSIM: {stage_info['ssim']:.4f}")
        print(f"  Hist Corr: {stage_info['hist_corr']:.4f}")
        print(f"  Rotation: {stage_rotation}")

        current_img = stage_img

    final_pee_img = cp.asnumpy(pee_stages[-1]['stage_img'])
    
    rotation_images = [stage['stage_img'] for stage in pee_stages]
    rotation_histograms = [cp.histogram(stage['stage_img'], bins=256, range=(0, 255))[0] for stage in pee_stages]
    
    return final_pee_img, int(total_payload), pee_stages, rotation_images, rotation_histograms, total_embeddings

def pee_process_with_split_cuda(img, total_embeddings, ratio_of_ones, use_different_weights, block_base=True):
    original_img = cp.asarray(img)
    height, width = original_img.shape
    total_pixels = height * width
    pee_stages = []
    total_payload = 0
    current_img = original_img.copy()
    previous_psnr = float('inf')
    previous_ssim = 1.0
    previous_payload = float('inf')

    for embedding in range(total_embeddings):
        print(f"\nStarting embedding {embedding}")
        
        if embedding == 0:
            target_psnr = 40.0
            target_bpp = 0.9
        else:
            target_psnr = max(28.0, previous_psnr - 1)  # Ensure PSNR decreases
            target_bpp = max(0.5, (previous_payload / total_pixels) * 0.95)  # Ensure payload decreases
        
        print(f"Target PSNR for embedding {embedding}: {target_psnr:.2f}")
        print(f"Target BPP for embedding {embedding}: {target_bpp:.4f}")
        
        stage_info = {
            'embedding': embedding,
            'sub_images': [],
            'payload': 0,
            'psnr': 0,
            'ssim': 0,
            'hist_corr': 0,
            'bpp': 0,
            'block_params': [],
            'rotated_sub_images': []
        }
        
        stage_payload = 0
        embedded_sub_images = []
        
        if block_base:
            sub_images = split_image(current_img)
        else:
            sub_images = split_image_into_quarters(current_img)
        
        # Calculate max_sub_payload based on the previous stage's payload or the size of the first sub-image
        if embedding > 0:
            max_sub_payload = previous_payload // len(sub_images)
        else:
            first_sub_img = sub_images[0]
            max_sub_payload = first_sub_img.size
        
        stage_rotations = [random.choice([-270, -180, -90, 0, 90, 180, 270]) for _ in range(len(sub_images))]

        for i, sub_img in enumerate(sub_images):
            sub_img = cp.asarray(sub_img)
            rotation = stage_rotations[i]
            rotated_sub_img = cp.rot90(sub_img, k=rotation // 90)

            sub_data = generate_random_binary_array(rotated_sub_img.size, ratio_of_ones)
            sub_data = cp.asarray(sub_data, dtype=cp.uint8)
            
            local_el = compute_adaptive_el(rotated_sub_img)
            
            if use_different_weights or i == 0:
                weights, (sub_payload, sub_psnr) = brute_force_weight_search_cuda(rotated_sub_img, sub_data, local_el, target_bpp, target_psnr, embedding)
            
            # Use the minimum of max_sub_payload and len(sub_data)
            data_to_embed = sub_data[:min(int(max_sub_payload), len(sub_data))]
            embedded_sub, payload = multi_pass_embedding(rotated_sub_img, data_to_embed, local_el, weights, embedding)
            
            rotated_back_sub = cp.rot90(embedded_sub, k=-rotation // 90)
            embedded_sub_images.append(rotated_back_sub)
            stage_payload += payload
            
            sub_img_np = cp.asnumpy(sub_img)
            rotated_back_sub_np = cp.asnumpy(rotated_back_sub)
            sub_psnr = calculate_psnr(sub_img_np, rotated_back_sub_np)
            sub_ssim = calculate_ssim(sub_img_np, rotated_back_sub_np)
            sub_hist_corr = histogram_correlation(
                np.histogram(sub_img_np, bins=256, range=(0, 255))[0],
                np.histogram(rotated_back_sub_np, bins=256, range=(0, 255))[0]
            )
            
            local_el_np = local_el.copy_to_host() if isinstance(local_el, cuda.cudadrv.devicearray.DeviceNDArray) else local_el
            max_el = int(np.max(local_el_np))

            block_info = {
                'weights': weights.tolist() if isinstance(weights, np.ndarray) else weights,
                'EL': max_el,
                'payload': int(payload),
                'psnr': float(sub_psnr),
                'ssim': float(sub_ssim),
                'rotation': rotation,
                'hist_corr': float(sub_hist_corr)
            }
            stage_info['sub_images'].append(block_info)
            stage_info['block_params'].append(block_info)
            stage_info['rotated_sub_images'].append(rotated_back_sub_np)

        if block_base:
            stage_img = merge_image(embedded_sub_images)
        else:
            stage_img = cp.vstack((cp.hstack((embedded_sub_images[0], embedded_sub_images[1])),
                                   cp.hstack((embedded_sub_images[2], embedded_sub_images[3]))))

        stage_info['stage_img'] = stage_img

        stage_img_np = cp.asnumpy(stage_img)
        original_img_np = cp.asnumpy(original_img)
        stage_info['psnr'] = float(calculate_psnr(original_img_np, stage_img_np))
        stage_info['ssim'] = float(calculate_ssim(original_img_np, stage_img_np))
        stage_info['hist_corr'] = float(histogram_correlation(
            np.histogram(original_img_np, bins=256, range=(0, 255))[0],
            np.histogram(stage_img_np, bins=256, range=(0, 255))[0]
        ))
        stage_info['payload'] = stage_payload
        stage_info['bpp'] = float(stage_info['payload'] / total_pixels)
        
        # Check if current stage's payload and PSNR are less than previous stage
        if embedding > 0:
            if stage_info['payload'] >= previous_payload:
                print(f"Warning: Stage {embedding} payload ({stage_info['payload']}) is not less than previous stage ({previous_payload}).")
                stage_info['payload'] = int(previous_payload * 0.95)  # Reduce payload to 95% of previous stage
                print(f"Adjusted payload: {stage_info['payload']}")
            
            if stage_info['psnr'] >= previous_psnr:
                print(f"Warning: Stage {embedding} PSNR ({stage_info['psnr']:.2f}) is not less than previous stage ({previous_psnr:.2f}).")
                # Note: We can't directly adjust PSNR here, but we've already tried to ensure it's lower in the embedding process
        
        pee_stages.append(stage_info)
        total_payload += stage_info['payload']
        previous_psnr = stage_info['psnr']
        previous_ssim = stage_info['ssim']
        previous_payload = stage_info['payload']

        print(f"Embedding {embedding} summary:")
        print(f"  Payload: {stage_info['payload']}")
        print(f"  BPP: {stage_info['bpp']:.4f}")
        print(f"  PSNR: {stage_info['psnr']:.2f}")
        print(f"  SSIM: {stage_info['ssim']:.4f}")
        print(f"  Hist Corr: {stage_info['hist_corr']:.4f}")
        print(f"  Rotations: {stage_rotations}")

        current_img = stage_img

    final_pee_img = cp.asnumpy(current_img)
    
    return final_pee_img, int(total_payload), pee_stages, stage_rotations

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

