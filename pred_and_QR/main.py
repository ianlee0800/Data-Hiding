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
    pee_process_with_rotation_cuda,
    pee_process_with_split_cuda  # 新增這行
)
from utils import (
    encode_pee_info,
    create_pee_info_table
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
    total_embeddings = 5
    ratio_of_ones = 0.5
    use_different_weights = False
    max_el = 7
    split_first = True

    # Create necessary directories
    os.makedirs(f"./pred_and_QR/outcome/histogram/{imgName}", exist_ok=True)
    os.makedirs(f"./pred_and_QR/outcome/histogram/{imgName}/difference", exist_ok=True)
    os.makedirs(f"./pred_and_QR/outcome/image/{imgName}", exist_ok=True)
    
    ensure_dir(f"./pred_and_QR/outcome/{imgName}/pee_info.json")
    
    try:
        # 清理 GPU 內存
        cp.get_default_memory_pool().free_all_blocks()

        # Read original image
        origImg = np.array(read_image(f"./pred_and_QR/image/{imgName}.{filetype}")).astype(np.uint8)

        # Save original image histogram
        save_histogram(origImg, f"./pred_and_QR/outcome/histogram/{imgName}/original_histogram.png", "Original Image Histogram")

        print(f"Starting encoding process... ({'CUDA' if cp.cuda.is_available() else 'CPU'} mode)")
        
        # X1: Improved Adaptive Prediction-Error Expansion (PEE) Embedding
        print("\nX1: Improved Adaptive Prediction-Error Expansion (PEE) Embedding")
        try:
            if split_first:
                final_img, total_payload, pee_stages, sub_images, sub_rotations = pee_process_with_split_cuda(
                    origImg, total_embeddings, ratio_of_ones, use_different_weights, max_el
                )
                images_to_save = sub_images
            else:
                final_img, total_payload, pee_stages, rotation_images, rotation_histograms, actual_embeddings = pee_process_with_rotation_cuda(
                    origImg, total_embeddings, ratio_of_ones, use_different_weights, max_el
                )
                images_to_save = rotation_images

            # 创建并打印PEE信息表
            total_pixels = origImg.size
            pee_table = create_pee_info_table(pee_stages, use_different_weights, total_pixels)
            print(pee_table)

            # 保存每个阶段的图像和直方图
            for i in range(min(len(pee_stages), len(images_to_save))):
                stage_img = images_to_save[i]
                if isinstance(stage_img, cp.ndarray):
                    stage_img = cp.asnumpy(stage_img)
                save_image(stage_img, f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1_stage_{i}.png")
                
                histogram_filename = f"./pred_and_QR/outcome/histogram/{imgName}/{imgName}_X1_stage_{i}_histogram.png"
                plt.figure(figsize=(10, 6))
                plt.bar(range(256), generate_histogram(stage_img), alpha=0.7)
                plt.title(f"Histogram after PEE Stage {i}")
                plt.xlabel("Pixel Value")
                plt.ylabel("Frequency")
                plt.savefig(histogram_filename)
                plt.close()

            # 準備PEE信息用於histogram shifting
            pee_info = {
                'total_embeddings': total_embeddings,
                'stages': [
                    {
                        'embedding': stage['embedding'],
                        'block_params': stage['block_params']
                    } for stage in pee_stages
                ]
            }

            # 保存 PEE 信息
            pee_info_path = f"./pred_and_QR/outcome/{imgName}/pee_info.json"
            ensure_dir(pee_info_path)
            with open(pee_info_path, 'w') as f:
                json.dump(pee_info, f, indent=2)

            # 讀取 pee_info.json 文件內容
            with open(pee_info_path, 'rb') as f:
                pee_info_content = f.read()

            # 計算 PEE 信息的實際大小（以比特為單位）
            pee_info_size = len(pee_info_content) * 8

            print(f"PEE info size: {pee_info_size} bits ({len(pee_info_content)} bytes)")

            # 將 PEE 信息內容轉換為比特流
            pee_info_bits = ''.join(format(byte, '08b') for byte in pee_info_content)

        except Exception as e:
            print(f"Error occurred in PEE process: {e}")
            import traceback
            traceback.print_exc()
            return

        # X2: Histogram Shifting Embedding
        print("\nX2: Histogram Shifting Embedding")
        try:
            # 將圖像旋轉回原始方向
            final_img_np = np.rot90(to_numpy(final_img), k=-total_embeddings)
            
            print(f"Before HS - Max pixel value: {np.max(final_img_np)}")
            print(f"Before HS - Min pixel value: {np.min(final_img_np)}")
            
            img_hs, payload_hs, hs_payloads, hs_rounds = histogram_data_hiding(final_img_np, pee_info_bits, ratio_of_ones)
            
            if payload_hs == 0:
                print("Warning: No data embedded during Histogram Shifting.")
            
            # 計算 bpp
            total_pixels = origImg.size
            bpp_hs = payload_hs / total_pixels
            
            # 保存直方圖移位後的圖像
            save_image(img_hs, f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X2.png")
            
            # 保存直方圖移位後的直方圖
            hist_hs = generate_histogram(img_hs)
            save_histogram(img_hs, f"./pred_and_QR/outcome/histogram/{imgName}/{imgName}_X2_histogram.png", "Histogram after X2 (Histogram Shifting)")

            # 保存差異直方圖
            diff = origImg.astype(np.float32) - img_hs.astype(np.float32)
            save_difference_histogram(diff, f"./pred_and_QR/outcome/histogram/{imgName}/{imgName}_difference_histogram.png", "Difference Histogram (Original - Final)")
            
            # 計算和保存結果
            psnr_hs = calculate_psnr(to_numpy(origImg), img_hs)
            ssim_hs = calculate_ssim(to_numpy(origImg), img_hs)
            hist_orig = generate_histogram(to_numpy(origImg))
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

            # 輸出最終結果
            print("\nEncoding Process Summary:")
            summary_table = PrettyTable()
            summary_table.field_names = ["Stage", "Payload", "BPP"]
            
            total_payload = 0
            
            # 添加 X1 階段（PEE）的數據
            for i, stage in enumerate(pee_stages):
                stage_payload = stage['payload']
                total_payload += stage_payload
                summary_table.add_row([
                    f"X1 Stage {i}",
                    stage_payload,
                    f"{stage_payload / total_pixels:.4f}"
                ])

            # 添加 X2 階段（Histogram Shifting）的數據
            total_payload += pee_info_size
            summary_table.add_row([
                "X2 (Histogram Shifting - PEE Info)",
                pee_info_size,
                f"{pee_info_size / total_pixels:.4f}"
            ])

            for i, hs_payload in enumerate(hs_payloads[1:], start=1):  # 跳過第一輪，因為它包含在 PEE Info 中
                total_payload += hs_payload
                summary_table.add_row([
                    f"X2 (Histogram Shifting - Round {i})",
                    hs_payload,
                    f"{hs_payload / total_pixels:.4f}"
                ])

            # 添加總計行
            summary_table.add_row([
                "Total",
                total_payload,
                f"{total_payload / total_pixels:.4f}"
            ])
            
            print(summary_table)

            # 保存最終結果
            final_results = {
                'method': 'Split' if split_first else 'Rotation',
                'pee_total_payload': sum(stage['payload'] for stage in pee_stages),
                'hs_payload': payload_hs,
                'total_payload': total_payload,
                'final_bpp': total_payload / total_pixels,
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

        print("Encoding process completed.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理 GPU 內存
        cp.get_default_memory_pool().free_all_blocks()

if __name__ == "__main__":
    main()