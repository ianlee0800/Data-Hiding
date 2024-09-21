import cupy as cp
import os
import sys
from image_processing import (
    read_image, 
    save_image,
    save_histogram,
    save_difference_histogram,
    generate_histogram,
    CUDA_AVAILABLE,
    image_rerotation
)
from embedding import (
    histogram_data_hiding,
    pee_process_with_rotation_cuda
    )
from utils import (
    encode_pee_info
    )
from common import (
    calculate_psnr,
    calculate_ssim,
    histogram_correlation
    )

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def main():
    # Parameter settings
    imgName = "airplane"  # Image name without extension
    filetype = "png"
    total_rotations = 4  # Total number of rotations
    ratio_of_ones = 0.5  # Ratio of ones in random data, easily adjustable

    # Create necessary directories
    os.makedirs(f"./pred_and_QR/outcome/histogram/{imgName}", exist_ok=True)
    os.makedirs(f"./pred_and_QR/outcome/histogram/{imgName}/difference", exist_ok=True)

    try:
        # 清理 GPU 內存
        cp.get_default_memory_pool().free_all_blocks()

        # Read original image
        origImg = read_image(f"./pred_and_QR/image/{imgName}.{filetype}")

        # Save original image histogram
        save_histogram(origImg, f"./pred_and_QR/outcome/histogram/{imgName}/original_histogram.png", "Original Image Histogram")

        print(f"Starting encoding process... ({'CUDA' if cp.cuda.is_available() else 'CPU'} mode)")

        # X1: Improved Adaptive Prediction-Error Expansion (PEE) Embedding
        print("\nX1: Improved Adaptive Prediction-Error Expansion (PEE) Embedding with Genetic Algorithm")
        final_img, total_payload, pee_stages = pee_process_with_rotation_cuda(origImg, total_rotations, ratio_of_ones)
        
        for i, stage in enumerate(pee_stages):
            print(f"X1 Rotation {i}:")
            print(f"  Total Payload: {stage['payload']}")
            print(f"  PSNR: {stage['psnr']:.2f}")
            print(f"  SSIM: {stage['ssim']:.4f}")
            print(f"  Number of blocks: {len(stage['block_params'])}")
            print("  Block details:")
            for j, block in enumerate(stage['block_params']):
                print(f"    Block {j}: EL={block['EL']}, Weights={block['weights']}, Payload={block['payload']}")
            
            # 計算該旋轉階段的平均 EL 和權重
            avg_el = sum(block['EL'] for block in stage['block_params']) / len(stage['block_params'])
            avg_weights = [sum(block['weights'][k] for block in stage['block_params']) / len(stage['block_params']) for k in range(4)]
            
            print(f"  Average EL: {avg_el:.2f}")
            print(f"  Average Weights: {[f'{w:.2f}' for w in avg_weights]}")
            
            # 保存圖像
            save_image(stage['image'], f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1_rotation_{i}.{filetype}")
            
            # 保存直方圖
            save_histogram(stage['image'], 
                        f"./pred_and_QR/outcome/histogram/{imgName}/X1_rotation_{i}_histogram.png", 
                        f"Histogram of PEE Rotation {i}")
            
            # 保存差異直方圖
            if i > 0:
                prev_img = pee_stages[i-1]['image']
            else:
                prev_img = origImg
            
            diff_before = stage['image'] - prev_img
            save_difference_histogram(diff_before, 
                                    f"./pred_and_QR/outcome/histogram/{imgName}/difference/X1_rotation_{i}_diff_before.png", 
                                    f"Difference Histogram Before PEE (Rotation {i})")
            
            diff_after = stage['image'] - prev_img
            save_difference_histogram(diff_after, 
                                    f"./pred_and_QR/outcome/histogram/{imgName}/difference/X1_rotation_{i}_diff_after.png", 
                                    f"Difference Histogram After PEE (Rotation {i})")

        # Prepare PEE info for HS stage
        print("Debug: Preparing to encode PEE info")
        print(f"Total rotations: {total_rotations}")
        print(f"Number of stages: {len(pee_stages)}")

        encoded_pee_info = encode_pee_info(total_rotations, pee_stages)

        print("Debug: PEE info encoded successfully")
        print(f"Encoded PEE info length: {len(encoded_pee_info)} bytes")

        # X2: Histogram Shifting Embedding
        print("\nX2: Histogram Shifting Embedding")
        img_hs, peak, payload_hs = histogram_data_hiding(final_img, 1, encoded_pee_info)
        
        # Calculate and save results
        psnr_hs = calculate_psnr(origImg, img_hs)
        ssim_hs = calculate_ssim(origImg, img_hs)
        hist_hs, _, _, _ = generate_histogram(img_hs)
        hist_orig, _, _, _ = generate_histogram(origImg)
        corr_hs = histogram_correlation(hist_orig, hist_hs)
        
        total_pixels = origImg.size
        final_bpp = (total_payload + payload_hs) / total_pixels
        
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
        print(f"Total Rotations: {total_rotations}")
        for i, stage in enumerate(pee_stages):
            avg_el = sum(block['EL'] for block in stage['block_params']) / len(stage['block_params'])
            avg_weights = [sum(block['weights'][k] for block in stage['block_params']) / len(stage['block_params']) for k in range(4)]
            stage_bpp = stage['payload'] / total_pixels
            print(f"X1 Rotation {i}: Avg EL={avg_el:.2f}, Avg Weights={[f'{w:.2f}' for w in avg_weights]}, Payload={stage['payload']}, BPP={stage_bpp:.4f}")
        print(f"X2 (Histogram Shifting): Peak={peak}, Payload={payload_hs}, BPP={final_bpp:.4f}")
        total_payload += payload_hs
        total_bpp = total_payload / total_pixels
        print(f"Total Payload={total_payload}, Total BPP={total_bpp:.4f}")
        print(f"Final PSNR={psnr_hs:.2f}, SSIM={ssim_hs:.4f}, Histogram Correlation={corr_hs:.4f}")

        print("...Encoding process completed...")

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # 清理 GPU 內存
        cp.get_default_memory_pool().free_all_blocks()

if __name__ == "__main__":
    main()