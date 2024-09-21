import numpy as np
import cupy as cp
from image_processing import (
    generate_histogram,
    merge_sub_images,
    split_image_into_blocks
)
from utils import (
    decode_pee_info,
    find_best_weights_ga,
)
from common import *
from pee import *

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
        total_rotations, weights, payloads, ELs = decode_pee_info(encoded_pee_info)
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
    for i in range(total_rotations):
        for w in weights[i]:
            # 將浮點數轉換為整數（乘以100並四捨五入），然後轉換為16位二進制
            embedding_data += format(int(round(w * 100)), '016b')
        embedding_data += format(ELs[i], '08b')
        embedding_data += format(payloads[i], '032b')
    
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

def pee_embedding_adaptive_cuda_split(img, data, base_EL, block_size=8):
    sub_images = split_image_into_blocks(img)
    embedded_sub_images = []
    total_payload = 0
    embedded_data = []
    block_params = []
    
    print(f"Debug: Number of sub-images: {len(sub_images)}")

    for i, sub_img in enumerate(sub_images):
        print(f"Debug: Processing sub-image {i}")
        sub_img_cp = cp.asarray(sub_img)
        sub_data = data[total_payload:total_payload + sub_img.size]
        
        # 為每個區塊獨立計算參數
        current_EL = choose_el(cp.asnumpy(sub_img_cp), i, total_payload)
        weights, _ = find_best_weights_ga(cp.asnumpy(sub_img_cp), sub_data, current_EL)
        
        print(f"Debug: Sub-image {i} - EL: {current_EL}, Weights: {weights}")

        embedded_img, payload, sub_embedded_data = pee_embedding_adaptive_cuda(
            sub_img_cp, sub_data, weights, current_EL, block_size)
        
        embedded_sub_images.append(cp.asnumpy(embedded_img))
        total_payload += payload
        embedded_data.extend(sub_embedded_data)
        
        # 儲存每個區塊的參數
        block_params.append({
            'weights': weights,
            'EL': current_EL,
            'payload': payload
        })
    
    merged_img = merge_sub_images(*embedded_sub_images)
    
    print(f"Debug: Total payload: {total_payload}")

    return merged_img, total_payload, embedded_data, block_params

def process_rotation(rotation, img, ratio_of_ones, toolbox):
    print(f"Starting rotation {rotation}")
    try:
        start_time = time.time()
        current_img = cp.asarray(img)
        if rotation > 0:
            current_img = cp.ascontiguousarray(cp.rot90(current_img, rotation))

        print(f"Rotation {rotation}: Generating random data")
        random_data = cp.random.binomial(1, ratio_of_ones, current_img.size).astype(cp.uint8)

        print(f"Rotation {rotation}: Choosing EL")
        EL = choose_el(current_img.get(), rotation, 0)

        print(f"Rotation {rotation}: Finding best weights")
        
        weights, fitness = find_best_weights_ga(cp.asnumpy(current_img), cp.asnumpy(random_data), EL)
        if isinstance(weights, np.ndarray) and len(weights) > 0 and isinstance(fitness, tuple) and len(fitness) == 2:
            print(f"Rotation {rotation}: Best weights: {weights}, Fitness: {fitness}")
        else:
            print(f"Rotation {rotation}: Failed to find valid weights")
            weights = np.array([1, 1, 1, 1])  # 使用默認權重
            fitness = (0.0, 0.0)
        
        print(f"Rotation {rotation}: Predicting image")
        weights_gpu = cp.asarray(weights)
        pred_img = improved_predict_image_cuda(current_img, weights_gpu)

        print(f"Rotation {rotation}: Performing PEE embedding")
        embedded_img, payload, embedded_data = pee_embedding_adaptive_cuda(current_img, random_data, pred_img, EL)

        print(f"Rotation {rotation}: Calculating PSNR and SSIM")
        rotated_back = cp.rot90(embedded_img, -rotation % 4)
        psnr = calculate_psnr(img, rotated_back.get())
        ssim = calculate_ssim(img, rotated_back.get())

        end_time = time.time()
        print(f"Rotation {rotation} completed in {end_time - start_time:.2f} seconds")

        return {
            'rotation': rotation,
            'image': embedded_img.get(),
            'payload': payload,
            'weights': weights.tolist(),
            'EL': EL,
            'psnr': psnr,
            'ssim': ssim
        }
    except Exception as e:
        print(f"Error in rotation {rotation}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def pee_process_with_rotation_cuda(img, rotations, ratio_of_ones):
    try:
        print("Initializing toolbox")
        toolbox = base.Toolbox()
        toolbox.register("attr_int", random.randint, 1, 15)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=4)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=1, up=15, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        print("Starting multiprocessing")
        with multiprocessing.Pool() as pool:
            process_func = partial(process_rotation, img=img, ratio_of_ones=ratio_of_ones, toolbox=toolbox)
            pee_stages = pool.map(process_func, range(rotations))

        # Filter out any None results from failed rotations
        pee_stages = [stage for stage in pee_stages if stage is not None]

        if not pee_stages:
            print("All rotations failed")
            return img, 0, []

        total_payload = sum(stage['payload'] for stage in pee_stages)
        final_img = pee_stages[-1]['image']

        print(f"PEE process completed. Total payload: {total_payload}")
        return final_img, total_payload, pee_stages

    except Exception as e:
        print(f"Unexpected error in PEE process: {str(e)}")
        import traceback
        traceback.print_exc()

    return img, 0, []

if __name__ == "__main__":
    # 測試代碼
    print("Starting test")
    test_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    rotations = 2
    ratio_of_ones = 0.5

    print("Calling pee_process_with_rotation_cuda")
    final_img, total_payload, pee_stages = pee_process_with_rotation_cuda(test_img, rotations, ratio_of_ones)

    print("Test completed")
    print(f"pee_process_with_rotation_cuda test result:")
    print(f"Final image shape: {final_img.shape}")
    print(f"Total payload: {total_payload}")
    print(f"Number of PEE stages: {len(pee_stages)}")

    for stage in pee_stages:
        print(f"Rotation {stage['rotation']}:")
        print(f"  Payload: {stage['payload']}")
        print(f"  PSNR: {stage['psnr']:.2f}")
        print(f"  SSIM: {stage['ssim']:.4f}")