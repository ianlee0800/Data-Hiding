import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from image_processing import (
    read_image, 
    save_image,
    check_quality_after_stage,
    improved_predict_image,
    generate_histogram,
    save_histogram,
    save_difference_histogram
)
from embedding import (
    pee_embedding_adaptive,
    histogram_data_hiding
)
from extraction import (
    pee_extraction_adaptive,
    histogram_data_extraction
)
from utils import (
    generate_random_binary_array,
    find_best_weights
)
from common import calculate_psnr, calculate_ssim, histogram_correlation

def main():
    # Parameter settings
    imgName = "airplane"
    filetype = "png"
    total_rotations = 4  # Total number of rotations
    EL = 3  # Embedding Limit (greater than 1)
    ratio_of_ones = 0  # Ratio of ones in random data, easily adjustable

    # Read original image
    origImg = read_image(f"./pred_and_QR/image/{imgName}.{filetype}")

    print("Starting encoding process...")

    # X1: Improved Adaptive Prediction-Error Expansion (PEE) Embedding
    print("\nX1: Improved Adaptive Prediction-Error Expansion (PEE) Embedding")
    pee_stages = []
    current_img = origImg.copy()
    total_payload = 0
    total_pixels = origImg.size
    
    for i in range(total_rotations + 1):
        weights = find_best_weights(current_img)
        random_data = generate_random_binary_array(current_img.size, ratio_of_ones=ratio_of_ones)
        
        img_pee, embedded_data, payload_pee = pee_embedding_adaptive(
            current_img,
            random_data,
            weights
        )
        
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
        
        # Save image and histograms (code remains unchanged)
        
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
    
    # Save final image and histogram (code remains unchanged)

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