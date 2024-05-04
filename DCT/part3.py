import os
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from part1 import get_image, perform_idct, perform_dct
from part2 import extract_secret

# 創建輸出目錄(如果不存在)
output_dir = "./DCT/part 3 results"
os.makedirs(output_dir, exist_ok=True)

# 定義輸入目錄
input_dir = "./DCT/part 2 results"

# 實現低通濾波函數
def low_pass_filter(image):
    kernel = np.ones((3, 3), dtype=np.float32) / 9  # 3x3 低通濾波核
    filtered_image = convolve2d(image, kernel, mode='same', boundary='symm')
    return filtered_image

# 實現中值濾波函數
def median_filter(image):
    filtered_image = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            neighborhood = image[i - 1:i + 2, j - 1:j + 2]
            filtered_image[i, j] = np.median(neighborhood)
    return filtered_image

if __name__ == "__main__":
    # 加載原始圖像
    original_image = get_image(f"./DCT/images/bridge.tiff")

    # 加載 Part 2 中生成的加水印圖像
    watermarked_images = {
        'random': get_image(f"{input_dir}/bridge-w1.tiff"),
        'all_zeros': get_image(f"{input_dir}/bridge-w2.tiff"),
        'all_ones': get_image(f"{input_dir}/bridge-w3.tiff")
    }

    # 加載嵌入的秘密位元流
    embedded_secrets = {
        'random': np.loadtxt(f"{input_dir}/bridge-w1_embedded_secret.txt", dtype=int),
        'all_zeros': np.loadtxt(f"{input_dir}/bridge-w2_embedded_secret.txt", dtype=int),
        'all_ones': np.loadtxt(f"{input_dir}/bridge-w3_embedded_secret.txt", dtype=int)
    }

    # 加載嵌入記錄
    embedding_records = {
        'random': np.load(f"{input_dir}/bridge-w1_embedding_record.npy"),
        'all_zeros': np.load(f"{input_dir}/bridge-w2_embedding_record.npy"),
        'all_ones': np.load(f"{input_dir}/bridge-w3_embedding_record.npy")
    }

    # 對加水印圖像進行低通濾波
    for secret_type, watermarked_image in watermarked_images.items():
        low_pass_filtered_image = low_pass_filter(watermarked_image)
        low_pass_filtered_image = np.clip(low_pass_filtered_image, 0, 255).astype(np.uint8)
        Image.fromarray(low_pass_filtered_image).save(f"{output_dir}/{secret_type}_watermarked_low_pass.png")

        # 從低通濾波後的圖像中提取秘密位元流
        watermarked_dct = perform_dct(low_pass_filtered_image)
        extracted_secret = extract_secret(watermarked_dct, embedding_records[secret_type], len(embedded_secrets[secret_type]), embedded_secrets[secret_type])
        
        watermarked_spatial = perform_idct(watermarked_dct)
        watermarked_spatial += 128  # Level shift
        watermarked_spatial = np.clip(watermarked_spatial, 0, 255).astype(np.uint8)

        # 對提取的DCT係數進行逆DCT變換，獲得加水印的空間域圖像
        watermarked_spatial = perform_idct(watermarked_dct)
        watermarked_spatial = np.clip(watermarked_spatial, 0, 255).astype(np.uint8)

        # 保存加水印的空間域圖像
        Image.fromarray(watermarked_spatial).save(f"{output_dir}/{secret_type}_watermarked_low_pass_spatial.png")

        # 計算提取準確度
        num_correct = np.sum(extracted_secret == embedded_secrets[secret_type])
        accuracy = num_correct / len(embedded_secrets[secret_type]) * 100
        print(f"Low-pass Filtered ({secret_type}):")
        print(f"Extraction Accuracy: {accuracy:.2f}%")
        print("")

    # 對加水印圖像進行中值濾波
    for secret_type, watermarked_image in watermarked_images.items():
        median_filtered_image = median_filter(watermarked_image)
        median_filtered_image = np.clip(median_filtered_image, 0, 255).astype(np.uint8)
        Image.fromarray(median_filtered_image).save(f"{output_dir}/{secret_type}_watermarked_median.png")

        # 從中值濾波後的圖像中提取秘密位元流
        watermarked_dct = perform_dct(median_filtered_image)
        extracted_secret = extract_secret(watermarked_dct, embedding_records[secret_type], len(embedded_secrets[secret_type]), embedded_secrets[secret_type])
        
        watermarked_spatial = perform_idct(watermarked_dct)
        watermarked_spatial += 128  # Level shift
        watermarked_spatial = np.clip(watermarked_spatial, 0, 255).astype(np.uint8)

        # 對提取的DCT係數進行逆DCT變換，獲得加水印的空間域圖像
        watermarked_spatial = perform_idct(watermarked_dct)
        watermarked_spatial = np.clip(watermarked_spatial, 0, 255).astype(np.uint8)

        # 保存加水印的空間域圖像
        Image.fromarray(watermarked_spatial).save(f"{output_dir}/{secret_type}_watermarked_median_spatial.png")

        # 計算提取準確度
        num_correct = np.sum(extracted_secret == embedded_secrets[secret_type])
        accuracy = num_correct / len(embedded_secrets[secret_type]) * 100
        print(f"Median Filtered ({secret_type}):")
        print(f"Extraction Accuracy: {accuracy:.2f}%")
        print("")