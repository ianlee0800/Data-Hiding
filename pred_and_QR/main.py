import warnings
from numba.core.errors import NumbaPerformanceWarning
import os
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
from image_processing import (
    save_image,
    save_histogram,
    generate_histogram,
    create_collage
)
from embedding import (
    pee_process_with_rotation_cuda,
    pee_process_with_split_cuda
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
    el_mode = 2  # 0: 無限制, 1: 漸增, 2: 漸減
    use_different_weights = True
    split_first = True  # Use split_first to choose PEE method
    split_size = 4  # 新增參數：切割尺寸 (4 表示 4x4=16 塊)
    block_base = False  # 選擇 block-based 或 quarter-based 分割

    # Create necessary directories
    os.makedirs(f"./pred_and_QR/outcome/histogram/{imgName}", exist_ok=True)
    os.makedirs(f"./pred_and_QR/outcome/histogram/{imgName}/difference", exist_ok=True)
    os.makedirs(f"./pred_and_QR/outcome/image/{imgName}", exist_ok=True)
    os.makedirs(f"./pred_and_QR/outcome/plots/{imgName}", exist_ok=True)  # 新增資料夾用於存放曲線圖
    
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
                    origImg, total_embeddings, ratio_of_ones, use_different_weights, split_size, el_mode, block_base
                )
            else:
                final_pee_img, total_payload, pee_stages = pee_process_with_rotation_cuda(
                    origImg, total_embeddings, ratio_of_ones, use_different_weights, split_size, el_mode
                )

            # Create and print PEE information table
            total_pixels = origImg.size
            pee_table = create_pee_info_table(pee_stages, use_different_weights, total_pixels, split_size)
            print(pee_table)

            # Save each stage's image and histogram
            for i, stage in enumerate(pee_stages):
                # Save the stage image
                save_image(cp.asnumpy(stage['stage_img']), 
                         f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1_stage_{i}_combined.png")
                
                # Create and save the collage-style image if available
                if 'rotated_sub_images' in stage and len(stage['rotated_sub_images']) > 0:
                    collage = create_collage(stage['rotated_sub_images'])
                    save_image(collage, f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1_stage_{i}_rotated_collage.png")
                
                # Save histogram
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
                'split_size': split_size,
                'block_base': block_base,
                'stages': [
                    {
                        'embedding': stage['embedding'],
                        'block_params': stage['block_params']
                    } for stage in pee_stages
                ]
            }

            # Save PEE information
            pee_info_path = f"./pred_and_QR/outcome/{imgName}/pee_info.npy"
            ensure_dir(pee_info_path)
            np.save(pee_info_path, pee_info)
            print(f"PEE information saved to {pee_info_path}")

            # Calculate and save BPP-PSNR data with accumulated payload
            bpp_psnr_data = []
            accumulated_payload = 0
            for stage in pee_stages:
                accumulated_payload += stage['payload']  # 累計 payload
                bpp_psnr_data.append({
                    'stage': stage['embedding'],
                    'bpp': accumulated_payload / total_pixels,  # 使用累計的 payload 計算 BPP
                    'psnr': stage['psnr']
                })

            # Plot and save BPP-PSNR curve with improved visualization
            plt.figure(figsize=(12, 8))
            bpps = [data['bpp'] for data in bpp_psnr_data]
            psnrs = [data['psnr'] for data in bpp_psnr_data]
            
            # 繪製折線圖
            plt.plot(bpps, psnrs, 'b.-', linewidth=2, markersize=8, 
                    label=f'Split Size: {split_size}x{split_size}')
            
            # 為每個點添加標籤
            for i, (bpp, psnr) in enumerate(zip(bpps, psnrs)):
                plt.annotate(f'Stage {i}\n({bpp:.3f}, {psnr:.2f})',
                            (bpp, psnr), 
                            textcoords="offset points",
                            xytext=(0,10), 
                            ha='center',
                            bbox=dict(boxstyle='round,pad=0.5', 
                                     fc='yellow', 
                                     alpha=0.3),
                            fontsize=8)
            
            plt.xlabel('Accumulated Bits Per Pixel (BPP)', fontsize=12)
            plt.ylabel('PSNR (dB)', fontsize=12)
            title = f'BPP-PSNR Curve for {imgName}\n'
            title += f'Split Size: {split_size}x{split_size}, {"Block-based" if block_base else "Quarter-based"}'
            plt.title(title, fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=10)
            
            # 調整坐標軸範圍，確保有足夠空間顯示標籤
            plt.margins(0.1)
            
            # 添加網格
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 保存圖片，使用高DPI以確保品質
            plt.savefig(f"./pred_and_QR/outcome/plots/{imgName}/bpp_psnr_curve.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()

            # Save BPP-PSNR data with more information
            np.save(f"./pred_and_QR/outcome/plots/{imgName}/bpp_psnr_data.npy",
                   {
                       'bpp': bpps,
                       'psnr': psnrs,
                       'split_size': split_size,
                       'block_base': block_base,
                       'stages': bpp_psnr_data  # 包含完整的階段資訊
                   })

            # Calculate and print final results
            final_bpp = total_payload / total_pixels
            final_psnr = calculate_psnr(origImg, final_pee_img)
            final_ssim = calculate_ssim(origImg, final_pee_img)
            hist_orig = generate_histogram(origImg)
            hist_final = generate_histogram(final_pee_img)
            final_hist_corr = histogram_correlation(hist_orig, hist_final)

            print("\nFinal Results:")
            print(f"Total Payload: {total_payload}")
            print(f"Final BPP: {final_bpp:.4f}")
            print(f"Final PSNR: {final_psnr:.2f}")
            print(f"Final SSIM: {final_ssim:.4f}")
            print(f"Final Histogram Correlation: {final_hist_corr:.4f}")

            # Save final results
            final_results = {
                'method': 'Split' if split_first else 'Rotation',
                'split_method': 'Block-based' if block_base else 'Quarter-based',
                'split_size': split_size,
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