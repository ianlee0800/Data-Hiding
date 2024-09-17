import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
from image_processing import (
    read_image, 
    save_image,
    save_histogram,
    save_difference_histogram,
    improved_predict_image,
    generate_histogram,
    CUDA_AVAILABLE
)
from embedding import histogram_data_hiding, pee_embedding_adaptive_cuda
from extraction import histogram_data_extraction
from utils import generate_random_binary_array, find_best_weights
from common import calculate_psnr, calculate_ssim, histogram_correlation

def main():
    # Parameter settings
    imgName = "baboon" # Image name without extension
    filetype = "png"
    total_rotations = 4  # Total number of rotations
    EL = 3  # Embedding Limit (greater than 1)
    ratio_of_ones = 0.5  # Ratio of ones in random data, easily adjustable

    # Create necessary directories
    os.makedirs(f"./pred_and_QR/outcome/histogram/{imgName}", exist_ok=True)
    os.makedirs(f"./pred_and_QR/outcome/histogram/{imgName}/difference", exist_ok=True)

    # Read original image
    origImg = read_image(f"./pred_and_QR/image/{imgName}.{filetype}")

    # Save original image histogram
    save_histogram(origImg, f"./pred_and_QR/outcome/histogram/{imgName}/original_histogram.png", "Original Image Histogram")

    print(f"Starting encoding process... ({'CUDA' if CUDA_AVAILABLE else 'CPU'} mode)")

    # X1: Improved Adaptive Prediction-Error Expansion (PEE) Embedding
    print("\nX1: Improved Adaptive Prediction-Error Expansion (PEE) Embedding")
    pee_stages = []
    current_img = origImg.copy()
    total_payload = 0
    total_pixels = origImg.size
    
    for i in range(total_rotations + 1):
        weights = find_best_weights(current_img)
        random_data = generate_random_binary_array(current_img.size, ratio_of_ones=ratio_of_ones)
        
        img_pee, payload_pee, embedded_data = pee_embedding_adaptive_cuda(
            current_img,
            random_data,
            weights
        )
        
        # 如果 img_pee 是 CuPy 数组，转换为 NumPy 数组
        if isinstance(img_pee, cp.ndarray):
            img_pee = cp.asnumpy(img_pee)
        
        pee_stages.append({
            'image': img_pee,
            'payload': payload_pee,
            'weights': weights
        })
        
        # Calculate and save results
        psnr = calculate_psnr(origImg, img_pee)
        ssim = calculate_ssim(origImg, img_pee)
        hist_orig, _, _, _ = generate_histogram(origImg)
        hist_pee, _, _, _ = generate_histogram(img_pee)
        corr = histogram_correlation(hist_orig, hist_pee)
        
        current_bpp = payload_pee / total_pixels
        
        print(f"X1 Rotation {i}: Payload={payload_pee}, BPP={current_bpp:.4f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}, Histogram Correlation={corr:.4f}")
        
        # Save image
        save_image(img_pee, f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1_rotation_{i}.{filetype}")
        
        # Save histogram
        save_histogram(img_pee, 
                       f"./pred_and_QR/outcome/histogram/{imgName}/X1_rotation_{i}_histogram.png", 
                       f"Histogram of PEE Rotation {i}")
        
        # Save difference histograms
        diff_before = current_img - improved_predict_image(current_img, weights)
        save_difference_histogram(diff_before, 
                                  f"./pred_and_QR/outcome/histogram/{imgName}/difference/X1_rotation_{i}_diff_before.png", 
                                  f"Difference Histogram Before PEE (Rotation {i})")
        
        diff_after = img_pee - current_img
        save_difference_histogram(diff_after, 
                                  f"./pred_and_QR/outcome/histogram/{imgName}/difference/X1_rotation_{i}_diff_after.png", 
                                  f"Difference Histogram After PEE (Rotation {i})")
        
        if i < total_rotations:
            current_img = np.rot90(img_pee)

    # X2: Histogram Shifting Embedding
    print("\nX2: Histogram Shifting Embedding")
    embedding_info = {
        'total_rotations': total_rotations + 1,
        'EL': EL,
        'weights': [stage['weights'] for stage in pee_stages],
        'payloads': [stage['payload'] for stage in pee_stages]
    }
    
    img_hs, peak, payload_hs = histogram_data_hiding(pee_stages[-1]['image'], 1, embedding_info)
    
    # Calculate and save results
    psnr_hs = calculate_psnr(origImg, img_hs)
    ssim_hs = calculate_ssim(origImg, img_hs)
    hist_hs, _, _, _ = generate_histogram(img_hs)
    corr_hs = histogram_correlation(hist_orig, hist_hs)
    
    final_bpp = payload_hs / total_pixels
    
    print(f"X2 (Histogram Shifting): Peak={peak}, Payload={payload_hs}, BPP={final_bpp:.4f}")
    print(f"X2 (Histogram Shifting): PSNR={psnr_hs:.2f}, SSIM={ssim_hs:.4f}, Histogram Correlation={corr_hs:.4f}")
    
    # Save final image
    save_image(img_hs, f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X2.{filetype}")
    
    # Save histogram
    save_histogram(img_hs, 
                   f"./pred_and_QR/outcome/histogram/{imgName}/X2_final_histogram.png", 
                   "Histogram of Final Image")

    # Output final results
    print("\nEncoding Process Summary:")
    print(f"Original Image: {imgName}")
    print(f"Total Rotations: {total_rotations + 1}, EL={EL}")
    for i, stage in enumerate(pee_stages):
        stage_bpp = stage['payload'] / total_pixels
        print(f"X1 Rotation {i}: Weights={stage['weights']}, Payload={stage['payload']}, BPP={stage_bpp:.4f}")
    print(f"X2 (Histogram Shifting): Peak={peak}, Payload={payload_hs}, BPP={final_bpp:.4f}")
    total_payload = sum(stage['payload'] for stage in pee_stages) + payload_hs
    total_bpp = total_payload / total_pixels
    print(f"Total Payload={total_payload}, Total BPP={total_bpp:.4f}")
    print(f"Final PSNR={psnr_hs:.2f}, SSIM={ssim_hs:.4f}, Histogram Correlation={corr_hs:.4f}")

    print("...Encoding process completed...")

if __name__ == "__main__":
    main()