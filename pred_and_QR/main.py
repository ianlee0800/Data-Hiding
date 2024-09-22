import warnings
from numba.core.errors import NumbaPerformanceWarning
import os
import sys
import cupy as cp
import numpy as np
import json
from prettytable import PrettyTable
from deap import base, creator, tools
from image_processing import (
    read_image, 
    save_image,
    save_histogram,
    generate_histogram
)
from embedding import (
    histogram_data_hiding,
    choose_pee_implementation,
)
from utils import (
    encode_pee_info
)
from common import *

warnings.filterwarnings("ignore", category=RuntimeWarning, module="deap.creator")
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

if 'FitnessMax' not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
if 'Individual' not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

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
    os.makedirs(f"./pred_and_QR/outcome/image/{imgName}", exist_ok=True)

    try:
        # 清理 GPU 内存
        cp.get_default_memory_pool().free_all_blocks()

        # Read original image
        origImg = cp.asarray(read_image(f"./pred_and_QR/image/{imgName}.{filetype}").astype(np.uint8))

        # Save original image histogram
        save_histogram(origImg, f"./pred_and_QR/outcome/histogram/{imgName}/original_histogram.png", "Original Image Histogram")

        # 选择实现
        pee_process, _, _ = choose_pee_implementation(use_cuda=True)
        
        print(f"Starting encoding process... ({'CUDA' if cp.cuda.is_available() else 'CPU'} mode)")

        # X1: Improved Adaptive Prediction-Error Expansion (PEE) Embedding
        print("\nX1: Improved Adaptive Prediction-Error Expansion (PEE) Embedding")
        try:
            final_img, total_payload, pee_stages = pee_process(origImg, total_rotations, ratio_of_ones)
        except Exception as e:
            print(f"Error occurred in CUDA PEE process: {e}")
            print("Falling back to CPU implementation...")
            pee_process, _, _ = choose_pee_implementation(use_cuda=False)
            final_img, total_payload, pee_stages = pee_process(origImg, total_rotations, ratio_of_ones)

        if pee_stages:
            table = PrettyTable()
            table.field_names = ["Rotation", "Sub-image", "Payload", "BPP", "PSNR", "SSIM", "Hist Corr", "EL", "Weights"]
            
            total_pixels = origImg.size
            for i, stage in enumerate(pee_stages):
                for j, sub_img in enumerate(stage['sub_images']):
                    table.add_row([
                        i if j == 0 else "",
                        j,
                        sub_img['payload'],
                        f"{sub_img['payload'] / (total_pixels / 4):.4f}",
                        f"{sub_img['psnr']:.2f}",
                        f"{sub_img['ssim']:.4f}",
                        f"{sub_img['hist_corr']:.4f}",
                        sub_img['EL'],
                        ", ".join([f"{w:.2f}" for w in sub_img['weights']])
                    ])
                table.add_row(["-" * 5] * 9)
                table.add_row([
                    i, "Total",
                    stage['payload'],
                    f"{stage['bpp']:.4f}",
                    f"{stage['psnr']:.2f}",
                    f"{stage['ssim']:.4f}",
                    f"{stage['hist_corr']:.4f}",
                    "-", "-"
                ])
                table.add_row(["-" * 5] * 9)
            
            print(table)

            # 准备PEE信息用于histogram shifting和存储
            pee_info = {
                'total_rotations': total_rotations,
                'total_payload': total_payload,
                'image_size': origImg.shape,
                'stages': pee_stages,
                'final_psnr': calculate_psnr(origImg, final_img),
                'final_ssim': calculate_ssim(origImg, final_img),
                'final_hist_corr': histogram_correlation(
                    np.histogram(origImg, bins=256, range=(0, 256))[0],
                    np.histogram(final_img, bins=256, range=(0, 256))[0]
                )
            }

            # 保存PEE信息
            with open(f"./pred_and_QR/outcome/{imgName}/pee_info.json", 'w') as f:
                json.dump(pee_info, f, indent=2)

            # 编码PEE信息用于histogram shifting
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
                    stage_bpp = stage['bpp']
                    summary_table.add_row([f"X1 Rotation {i}", stage['payload'], f"{stage_bpp:.4f}"])
            
            summary_table.add_row(["X2 (Histogram Shifting)", payload_hs, f"{payload_hs / total_pixels:.4f}"])
            summary_table.add_row(["Total", total_payload + payload_hs, f"{final_bpp:.4f}"])
            
            print(summary_table)

            # 保存最终结果
            final_results = {
                'pee_total_payload': total_payload,
                'hs_payload': payload_hs,
                'total_payload': total_payload + payload_hs,
                'final_bpp': final_bpp,
                'final_psnr': psnr_hs,
                'final_ssim': ssim_hs,
                'final_hist_corr': corr_hs
            }
            with open(f"./pred_and_QR/outcome/{imgName}/final_results.json", 'w') as f:
                json.dump(final_results, f, indent=2)

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