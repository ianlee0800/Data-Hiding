import warnings
from numba.core.errors import NumbaPerformanceWarning
import os
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import json
from image_processing import (
    save_image,
    save_histogram,
    generate_histogram,
    create_collage
)
from embedding import (
    pee_process_with_rotation_cuda,
    pee_process_with_split_cuda  # 新增這行
)
from utils import (
    create_pee_info_table
)
from common import *

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # Parameter settings
    imgName = "airplane"  # Image name without extension
    filetype = "png"
    total_embeddings = 5
    ratio_of_ones = 0.5
    use_different_weights = True
    split_first = True  # Use split_first to choose PEE method
    block_base = False  # New parameter to choose between block-based and quarter-based splitting

    # Create necessary directories
    os.makedirs(f"./pred_and_QR/outcome/histogram/{imgName}", exist_ok=True)
    os.makedirs(f"./pred_and_QR/outcome/histogram/{imgName}/difference", exist_ok=True)
    os.makedirs(f"./pred_and_QR/outcome/image/{imgName}", exist_ok=True)
    
    ensure_dir(f"./pred_and_QR/outcome/{imgName}/pee_info.npy")
    
    try:
        # Clean GPU memory
        cp.get_default_memory_pool().free_all_blocks()

        # Read original image
        origImg = cv2.imread(f"./pred_and_QR/image/{imgName}.{filetype}", cv2.IMREAD_GRAYSCALE)
        if origImg is None:
            raise ValueError(f"Failed to read image: ./pred_and_QR/image/{imgName}.{filetype}")
        origImg = np.array(origImg).astype(np.uint8)

        # Save original image histogram
        save_histogram(origImg, f"./pred_and_QR/outcome/histogram/{imgName}/original_histogram.png", "Original Image Histogram")

        print(f"Starting encoding process... ({'CUDA' if cp.cuda.is_available() else 'CPU'} mode)")
        
        # X1: Improved Adaptive Prediction-Error Expansion (PEE) Embedding
        print("\nX1: Improved Adaptive Prediction-Error Expansion (PEE) Embedding")
        try:
            if split_first:
                final_pee_img, total_payload, pee_stages, stage_rotations = pee_process_with_split_cuda(
                    origImg, total_embeddings, ratio_of_ones, use_different_weights, block_base
                )
            else:
                final_pee_img, total_payload, pee_stages, rotation_images, rotation_histograms, actual_embeddings = pee_process_with_rotation_cuda(
                    origImg, total_embeddings, ratio_of_ones, use_different_weights
                )

            # Create and print PEE information table
            total_pixels = origImg.size
            pee_table = create_pee_info_table(pee_stages, use_different_weights, total_pixels)
            print(pee_table)

            # Save each stage's image and histogram
            for i, stage in enumerate(pee_stages):
                # Save the 0-degree oriented stage image
                save_image(cp.asnumpy(stage['stage_img']), f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1_stage_{i}_combined_0deg.png")
                
                # Create and save the collage-style rotated image
                if 'rotated_sub_images' in stage:
                    collage = create_collage(stage['rotated_sub_images'])
                    save_image(collage, f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1_stage_{i}_combined_rotated_collage.png")
                
                # Save histogram (use 0-degree version for consistency)
                histogram_filename = f"./pred_and_QR/outcome/histogram/{imgName}/{imgName}_X1_stage_{i}_histogram.png"
                plt.figure(figsize=(10, 6))
                plt.bar(range(256), generate_histogram(cp.asnumpy(stage['stage_img'])), alpha=0.7)
                plt.title(f"Histogram after PEE Stage {i}")
                plt.xlabel("Pixel Value")
                plt.ylabel("Frequency")
                plt.savefig(histogram_filename)
                plt.close()

            # Save final PEE image
            save_image(final_pee_img, f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1_final.png")

            # Prepare PEE information
            pee_info = {
                'total_embeddings': total_embeddings,
                'stages': [
                    {
                        'embedding': stage['embedding'],
                        'block_params': stage['block_params']
                    } for stage in pee_stages
                ]
            }

            # Save PEE information as a NumPy file
            pee_info_path = f"./pred_and_QR/outcome/{imgName}/pee_info.npy"
            ensure_dir(pee_info_path)
            np.save(pee_info_path, pee_info)
            print(f"PEE information saved to {pee_info_path}")

            # Calculate and print final results
            final_bpp = total_payload / total_pixels
            final_psnr = calculate_psnr(origImg, final_pee_img)
            final_ssim = calculate_ssim(origImg, final_pee_img)
            hist_orig = generate_histogram(origImg)
            hist_final = generate_histogram(final_pee_img)
            final_hist_corr = histogram_correlation(hist_orig, hist_final)

            print("\nFinal PEE Results:")
            print(f"Total Payload: {total_payload}")
            print(f"Final BPP: {final_bpp:.4f}")
            print(f"Final PSNR: {final_psnr:.2f}")
            print(f"Final SSIM: {final_ssim:.4f}")
            print(f"Final Histogram Correlation: {final_hist_corr:.4f}")

            # Save final results
            final_results = {
                'method': 'Split' if split_first else 'Rotation',
                'split_method': 'Block-based' if block_base else 'Quarter-based',
                'total_payload': total_payload,
                'final_bpp': final_bpp,
                'final_psnr': final_psnr,
                'final_ssim': final_ssim,
                'final_hist_corr': final_hist_corr
            }
            np.save(f"./pred_and_QR/outcome/{imgName}/final_results.npy", final_results)
            print(f"Final results saved to ./pred_and_QR/outcome/{imgName}/final_results.npy")

        except Exception as e:
            print(f"Error occurred in PEE process: {e}")
            import traceback
            traceback.print_exc()
            return

        print("PEE encoding process completed.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean GPU memory
        cp.get_default_memory_pool().free_all_blocks()

if __name__ == "__main__":
    main()