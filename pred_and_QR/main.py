import warnings
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=RuntimeWarning, module="deap.creator")
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


from deap import base, creator, tools
if 'FitnessMax' not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
if 'Individual' not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

import os
import sys
import cupy as cp
from prettytable import PrettyTable
from image_processing import (
    read_image, 
    save_image,
    save_histogram,
    save_difference_histogram,
    generate_histogram
)
from embedding import (
    histogram_data_hiding,
    pee_process_with_rotation_cuda
)
from utils import (
    encode_pee_info
)
from common import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def main():
    # Parameter settings
    imgName = "airplane"  # Image name without extension
    filetype = "png"
    total_rotations = 5  # Total number of rotations
    ratio_of_ones = 0.5  # Ratio of ones in random data, easily adjustable

    # Create necessary directories
    os.makedirs(f"./pred_and_QR/outcome/histogram/{imgName}", exist_ok=True)
    os.makedirs(f"./pred_and_QR/outcome/histogram/{imgName}/difference", exist_ok=True)

    try:
        # 清理 GPU 内存
        cp.get_default_memory_pool().free_all_blocks()

        # Read original image
        origImg = read_image(f"./pred_and_QR/image/{imgName}.{filetype}").astype(np.uint8)

        # Save original image histogram
        save_histogram(origImg, f"./pred_and_QR/outcome/histogram/{imgName}/original_histogram.png", "Original Image Histogram")

        print(f"Starting encoding process... ({'CUDA' if cp.cuda.is_available() else 'CPU'} mode)")

        # X1: Improved Adaptive Prediction-Error Expansion (PEE) Embedding
        print("\nX1: Improved Adaptive Prediction-Error Expansion (PEE) Embedding")
        final_img, total_payload, pee_stages = pee_process_with_rotation_cuda(origImg, total_rotations, ratio_of_ones)

        if pee_stages:
            table = PrettyTable()
            table.field_names = ["Rotation", "Block", "Payload", "PSNR", "SSIM", "EL", "Weights"]
            
            for i, stage in enumerate(pee_stages):
                for j, block in enumerate(stage['block_params']):
                    table.add_row([
                        i if j == 0 else "",
                        j,
                        stage['payload'] if j == 0 else "",
                        f"{stage['psnr']:.2f}" if j == 0 else "",
                        f"{stage['ssim']:.4f}" if j == 0 else "",
                        block['EL'],
                        ", ".join([f"{w:.2f}" for w in block['weights']])
                    ])
            
            print(table)

            # 准备PEE信息用于histogram shifting
            pee_info = {
                'total_rotations': total_rotations,
                'stages': [
                    {
                        'rotation': stage['rotation'],
                        'block_params': stage['block_params']
                    } for stage in pee_stages
                ]
            }

            # 编码PEE信息
            encoded_pee_info = encode_pee_info(pee_info)
        else:
            print("No PEE stages were generated. Using original image for histogram shifting.")
            final_img = origImg
            total_payload = 0
            encoded_pee_info = encode_pee_info({'total_rotations': 0, 'stages': []})

        # X2: Histogram Shifting Embedding
        print("\nX2: Histogram Shifting Embedding")
        try:
            img_hs, peak, payload_hs = histogram_data_hiding(to_numpy(final_img), 1, encoded_pee_info)
            
            # 计算和保存结果
            psnr_hs = calculate_psnr(origImg, img_hs)
            ssim_hs = calculate_ssim(origImg, img_hs)
            hist_hs, _, _, _ = generate_histogram(img_hs)
            hist_orig, _, _, _ = generate_histogram(origImg)
            corr_hs = histogram_correlation(hist_orig, hist_hs)
            
            total_pixels = origImg.size
            final_bpp = (total_payload + payload_hs) / total_pixels
            
            hs_table = PrettyTable()
            hs_table.field_names = ["Peak", "Payload", "BPP", "PSNR", "SSIM", "Histogram Correlation"]
            hs_table.add_row([
                peak,
                payload_hs,
                f"{final_bpp:.4f}",
                f"{psnr_hs:.2f}",
                f"{ssim_hs:.4f}",
                f"{corr_hs:.4f}"
            ])
            
            print(hs_table)

            # 保存最终图像
            save_image(img_hs, f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X2.{filetype}")
            
            # 保存直方图
            save_histogram(img_hs, 
                        f"./pred_and_QR/outcome/histogram/{imgName}/X2_final_histogram.png", 
                        "Histogram of Final Image")

            # 输出最终结果
            print("\nEncoding Process Summary:")
            summary_table = PrettyTable()
            summary_table.field_names = ["Stage", "Payload", "BPP"]
            
            if pee_stages:
                for i, stage in enumerate(pee_stages):
                    stage_bpp = stage['payload'] / total_pixels
                    summary_table.add_row([f"X1 Rotation {i}", stage['payload'], f"{stage_bpp:.4f}"])
            
            summary_table.add_row(["X2 (Histogram Shifting)", payload_hs, f"{final_bpp:.4f}"])
            summary_table.add_row(["Total", total_payload + payload_hs, f"{final_bpp:.4f}"])
            
            print(summary_table)

        except Exception as e:
            print(f"Error in histogram data hiding: {e}")
            print("Skipping histogram shifting embedding...")

        print("Encoding process completed.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理 GPU 内存
        cp.get_default_memory_pool().free_all_blocks()

if __name__ == "__main__":
    main()