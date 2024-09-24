import warnings
from numba.core.errors import NumbaPerformanceWarning
import os
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import json
from prettytable import PrettyTable
from deap import base, creator, tools
from image_processing import (
    read_image, 
    save_image,
    save_histogram,
    generate_histogram,
    save_difference_histogram
)
from embedding import (
    histogram_data_hiding,
    choose_pee_implementation,
    pee_process_with_rotation_cuda
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

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

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
    
    # 确保必要的目录存在
    ensure_dir(f"./pred_and_QR/outcome/{imgName}/pee_info.json")
    ensure_dir(f"./pred_and_QR/outcome/histogram/{imgName}")
    ensure_dir(f"./pred_and_QR/outcome/image/{imgName}")
    
    # 定义 save_stage_image 函数
    def save_stage_image(img, stage, rotation=None):
        """
        保存每个阶段的图像
        
        :param img: 要保存的图像
        :param stage: 阶段标识符（'X1' 或 'X2'）
        :param rotation: 旋转次数（仅适用于 X1 阶段）
        """
        if stage == 'X1':
            filename = f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1_rotation_{rotation}.png"
        else:
            filename = f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X2.png"
        
        save_image(img, filename)
        print(f"Saved {filename}")
    
    try:
        # 清理 GPU 内存
        cp.get_default_memory_pool().free_all_blocks()

        # Read original image
        origImg = np.array(read_image(f"./pred_and_QR/image/{imgName}.{filetype}")).astype(np.uint8)

        # Save original image histogram
        save_histogram(origImg, f"./pred_and_QR/outcome/histogram/{imgName}/original_histogram.png", "Original Image Histogram")

        # 选择实现
        pee_process, _, _ = choose_pee_implementation(use_cuda=True)
        
        print(f"Starting encoding process... ({'CUDA' if cp.cuda.is_available() else 'CPU'} mode)")
        
        # 创建保存直方图的目录
        os.makedirs(f"./pred_and_QR/outcome/histogram/{imgName}", exist_ok=True)

        # 保存原始图像的直方图
        save_histogram(origImg, f"./pred_and_QR/outcome/histogram/{imgName}/{imgName}_original_histogram.png", "Original Image Histogram")

        # X1: Improved Adaptive Prediction-Error Expansion (PEE) Embedding
        print("\nX1: Improved Adaptive Prediction-Error Expansion (PEE) Embedding")
        try:
            final_img, total_payload, pee_stages, rotation_images, rotation_histograms, total_rotations = pee_process_with_rotation_cuda(origImg, total_rotations, ratio_of_ones, target_payload=480000)

            # 保存每次旋转后的图像和直方图
            for i, (img, hist) in enumerate(zip(rotation_images, rotation_histograms)):
                # 保存图像
                save_stage_image(img, 'X1', i)
                
                # 保存直方图
                histogram_filename = f"./pred_and_QR/outcome/histogram/{imgName}/{imgName}_X1_rotation_{i}_histogram.png"
                plt.figure(figsize=(10, 6))
                plt.bar(range(256), hist, alpha=0.7)
                plt.title(f"Histogram after PEE Rotation {i}")
                plt.xlabel("Pixel Value")
                plt.ylabel("Frequency")
                plt.savefig(histogram_filename)
                plt.close()
        
        except Exception as e:
            print(f"Error occurred in CUDA PEE process: {e}")
            print("Falling back to CPU implementation...")
            pee_process, _, _ = choose_pee_implementation(use_cuda=False)
            final_img, total_payload, pee_stages = pee_process(origImg, total_rotations, ratio_of_ones)

        if pee_stages:
            table = PrettyTable()
            table.field_names = ["Rotation", "Sub-image", "Payload", "BPP", "PSNR", "SSIM", "Hist Corr", "EL", "Weights"]
            
            for i, stage in enumerate(pee_stages):
                for j, block in enumerate(stage['block_params']):
                    table.add_row([
                        i if j == 0 else "",
                        j,
                        block['payload'],
                        f"{block['payload'] / (origImg.size / 4):.4f}",
                        f"{block['psnr']:.2f}",
                        f"{block['ssim']:.4f}",
                        f"{block['hist_corr']:.4f}",
                        block['EL'],
                        ", ".join([f"{w:.2f}" for w in block['weights']])
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

            # 保存 PEE 信息
            pee_info_path = f"./pred_and_QR/outcome/{imgName}/pee_info.json"
            ensure_dir(pee_info_path)
            with open(pee_info_path, 'w') as f:
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
            # 將圖像旋轉回原始方向
            final_img_np = np.rot90(to_numpy(final_img), k=-total_rotations)
            
            img_hs, payload_hs, hs_payloads, hs_rounds = histogram_data_hiding(final_img_np, encoded_pee_info, ratio_of_ones)
            
            if payload_hs == 0:
                print("Warning: No data embedded during Histogram Shifting.")
            
            # 計算 bpp
            total_pixels = origImg.size
            bpp_hs = payload_hs / total_pixels
            
            # 保存直方圖移位後的圖像
            save_stage_image(img_hs, 'X2')
            
            # 保存直方圖移位後的直方圖
            save_histogram(img_hs, f"./pred_and_QR/outcome/histogram/{imgName}/{imgName}_X2_histogram.png", "Histogram after X2 (Histogram Shifting)")

            # 保存差異直方圖
            diff = origImg.astype(np.float32) - img_hs.astype(np.float32)
            save_difference_histogram(diff, f"./pred_and_QR/outcome/histogram/{imgName}/{imgName}_difference_histogram.png", "Difference Histogram (Original - Final)")
            
            # 計算和保存結果
            psnr_hs = calculate_psnr(to_numpy(origImg), img_hs)
            ssim_hs = calculate_ssim(to_numpy(origImg), img_hs)
            hist_hs, _, _, _ = generate_histogram(img_hs)
            hist_orig, _, _, _ = generate_histogram(to_numpy(origImg))
            corr_hs = histogram_correlation(hist_orig, hist_hs)
            
            final_bpp = (total_payload + payload_hs) / total_pixels
            
            hs_table = PrettyTable()
            hs_table.field_names = ["Total Payload", "Rounds", "BPP", "PSNR", "SSIM", "Histogram Correlation"]
            hs_table.add_row([
                payload_hs,
                hs_rounds,
                f"{bpp_hs:.4f}",
                f"{psnr_hs:.2f}",
                f"{ssim_hs:.4f}",
                f"{corr_hs:.4f}"
            ])
            
            print(hs_table)

            # 輸出每輪 HS 的詳細信息
            print("\nHistogram Shifting Rounds Details:")
            hs_details_table = PrettyTable()
            hs_details_table.field_names = ["Round", "Payload", "BPP"]
            for i, payload in enumerate(hs_payloads):
                hs_details_table.add_row([i+1, payload, f"{payload / total_pixels:.4f}"])
            print(hs_details_table)

            # 输出最终结果
            print("\nEncoding Process Summary:")
            summary_table = PrettyTable()
            summary_table.field_names = ["Stage", "Payload", "BPP"]
            
            if pee_stages:
                for i, stage in enumerate(pee_stages):
                    stage_bpp = stage['bpp']
                    summary_table.add_row([f"X1 Rotation {i}", stage['payload'], f"{stage_bpp:.4f}"])
            
            pee_info_size = len(encoded_pee_info) * 8  # PEE 信息的比特数
            
            summary_table.add_row(["X2 (Histogram Shifting - PEE Info)", pee_info_size, f"{pee_info_size / total_pixels:.4f}"])
            for i, hs_payload in enumerate(hs_payloads[1:], start=1):  # 跳过第一轮，因为它包含在 PEE Info 中
                summary_table.add_row([f"X2 (Histogram Shifting - Round {i})", hs_payload, f"{hs_payload / total_pixels:.4f}"])
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
            import traceback
            traceback.print_exc()
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