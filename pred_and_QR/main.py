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
    add_grid_lines
)
from embedding import (
    pee_process_with_rotation_cuda,
    pee_process_with_split_cuda
)
from utils import (
    create_pee_info_table
)
from common import *
from quadtree import pee_process_with_quadtree_cuda

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # Parameter settings
    imgName = "bridge"  # Image name without extension
    filetype = "png"
    total_embeddings = 5
    ratio_of_ones = 0.5
    el_mode = 0  # 0: 無限制, 1: 漸增, 2: 漸減
    use_different_weights = False # 是否使用不同的權重
    
    # 方法選擇參數
    method = "split"  # 可選："rotation", "split", "quadtree"
    
    # 各方法共用參數
    split_size = 2  # 用於 rotation 和 split 方法
    block_base = False  # 用於 split 方法
    
    # quad tree 特定參數
    quad_tree_params = {
        'min_block_size': 16,  # 支援到16x16
        'variance_threshold': 300
    }

    # Create necessary directories
    os.makedirs(f"./pred_and_QR/outcome/histogram/{imgName}", exist_ok=True)
    os.makedirs(f"./pred_and_QR/outcome/histogram/{imgName}/difference", exist_ok=True)
    os.makedirs(f"./pred_and_QR/outcome/image/{imgName}", exist_ok=True)
    os.makedirs(f"./pred_and_QR/outcome/plots/{imgName}", exist_ok=True)
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
        print(f"Using method: {method}")
        
        try:
            # 執行選定的方法
            if method == "rotation":
                final_pee_img, total_payload, pee_stages = pee_process_with_rotation_cuda(
                    origImg,
                    total_embeddings,
                    ratio_of_ones,
                    use_different_weights,
                    split_size,
                    el_mode
                )
            elif method == "split":
                final_pee_img, total_payload, pee_stages = pee_process_with_split_cuda(
                    origImg,
                    total_embeddings,
                    ratio_of_ones,
                    use_different_weights,
                    split_size,
                    el_mode,
                    block_base
                )
            elif method == "quadtree":
                final_pee_img, total_payload, pee_stages = pee_process_with_quadtree_cuda(
                    origImg,
                    total_embeddings,
                    ratio_of_ones,
                    use_different_weights,
                    quad_tree_params['min_block_size'],
                    quad_tree_params['variance_threshold'],
                    el_mode,
                    rotation_mode='random'  # 明確指定使用 random mode
                )

            # Create and print PEE information table
            total_pixels = origImg.size
            pee_table = create_pee_info_table(pee_stages, use_different_weights, total_pixels, 
                                           split_size, method == "quadtree")
            print(pee_table)

            # Save each stage's image and histogram
            for i, stage in enumerate(pee_stages):
                # 使用原始圖像（無格線）進行質量評估
                stage_img = cp.asnumpy(stage['stage_img'])
                
                # 保存原始階段圖像（旋轉回0度後的圖像）
                save_image(stage_img, 
                         f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1_stage_{i}_combined.png")
                
                # 保存旋轉狀態的圖像（未旋轉回0度）
                if 'rotated_stage_img' in stage:
                    rotated_img = cp.asnumpy(stage['rotated_stage_img'])
                    save_image(rotated_img,
                             f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1_stage_{i}_rotated.png")
                    
                    # 為旋轉狀態的圖像添加格線
                    grid_rotated_img = add_grid_lines(rotated_img.copy(), stage['block_info'])
                    save_image(grid_rotated_img,
                             f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1_stage_{i}_rotated_with_grid.png")

                # 為階段圖像添加格線並保存
                if method == "quadtree":
                    grid_image = add_grid_lines(stage_img.copy(), stage['block_info'])
                    save_image(grid_image, 
                             f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1_stage_{i}_with_grid.png")
                
                # 打印區塊統計資訊
                if method == "quadtree":
                    print(f"\nBlock statistics for Stage {i}:")
                    for size in sorted([int(s) for s in stage['block_info'].keys()], reverse=True):
                        block_count = len(stage['block_info'][str(size)]['blocks'])
                        if block_count > 0:
                            rotation = stage['block_rotations'][size] if 'block_rotations' in stage else 0
                            print(f"  {size}x{size} blocks: {block_count}, Rotation: {rotation}°")
                    print("")

                # Create and save histogram
                histogram_filename = f"./pred_and_QR/outcome/histogram/{imgName}/{imgName}_X1_stage_{i}_histogram.png"
                plt.figure(figsize=(10, 6))
                plt.bar(range(256), generate_histogram(stage_img), alpha=0.7)
                plt.title(f"Histogram after PEE Stage {i}")
                plt.xlabel("Pixel Value")
                plt.ylabel("Frequency")
                plt.savefig(histogram_filename)
                plt.close()

            # Save final PEE image
            save_image(final_pee_img, 
                      f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1_final.png")
            
            # 如果是 quadtree 方法，為最終圖像添加格線
            if method == "quadtree":
                final_grid_image = add_grid_lines(final_pee_img.copy(), pee_stages[-1]['block_info'])
                save_image(final_grid_image, 
                          f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1_final_with_grid.png")

            # Calculate and save BPP-PSNR data
            bpp_psnr_data = []
            accumulated_payload = 0
            for stage in pee_stages:
                accumulated_payload += stage['payload']
                bpp_psnr_data.append({
                    'stage': stage['embedding'],
                    'bpp': accumulated_payload / total_pixels,
                    'psnr': stage['psnr']
                })

            # Plot and save BPP-PSNR curve
            plt.figure(figsize=(12, 8))
            bpps = [data['bpp'] for data in bpp_psnr_data]
            psnrs = [data['psnr'] for data in bpp_psnr_data]
            
            plt.plot(bpps, psnrs, 'b.-', linewidth=2, markersize=8, 
                    label=f'Method: {method}')
            
            # Add labels for each point
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
            if method == "quadtree":
                title += f'Min Block Size: {quad_tree_params["min_block_size"]}'
            elif method == "split":
                title += f'Split Size: {split_size}x{split_size}, {"Block-based" if block_base else "Quarter-based"}'
            else:
                title += f'Split Size: {split_size}x{split_size}'
            plt.title(title, fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=10)
            
            plt.margins(0.1)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.savefig(f"./pred_and_QR/outcome/plots/{imgName}/bpp_psnr_curve.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()

            # Save BPP-PSNR data
            np.save(f"./pred_and_QR/outcome/plots/{imgName}/bpp_psnr_data.npy",
                   {
                       'method': method,
                       'bpp': bpps,
                       'psnr': psnrs,
                       'split_size': split_size if method != "quadtree" else None,
                       'block_base': block_base if method == "split" else None,
                       'min_block_size': quad_tree_params['min_block_size'] if method == "quadtree" else None,
                       'stages': bpp_psnr_data
                   })

            # Calculate and print final results using original image (no grid)
            final_bpp = total_payload / total_pixels
            final_psnr = calculate_psnr(origImg, final_pee_img)
            final_ssim = calculate_ssim(origImg, final_pee_img)
            hist_orig = generate_histogram(origImg)
            hist_final = generate_histogram(final_pee_img)
            final_hist_corr = histogram_correlation(hist_orig, hist_final)

            print("\nFinal Results:")
            print(f"Method: {method}")
            print(f"Total Payload: {total_payload}")
            print(f"Final BPP: {final_bpp:.4f}")
            print(f"Final PSNR: {final_psnr:.2f}")
            print(f"Final SSIM: {final_ssim:.4f}")
            print(f"Final Histogram Correlation: {final_hist_corr:.4f}")

            # Save final results
            final_results = {
                'method': method,
                'total_payload': total_payload,
                'final_bpp': final_bpp,
                'final_psnr': final_psnr,
                'final_ssim': final_ssim,
                'final_hist_corr': final_hist_corr
            }
            
            if method == "quadtree":
                final_results.update({
                    'min_block_size': quad_tree_params['min_block_size'],
                    'variance_threshold': quad_tree_params['variance_threshold']
                })
            else:
                final_results.update({
                    'split_size': split_size,
                    'block_base': block_base if method == "split" else None
                })
            
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