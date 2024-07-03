import numpy as np
import cv2
import torch
import os
from CNN import AdaptiveCNNPredictor, preprocess_image

def load_cnn_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CNN model file not found: {model_path}")
    model = AdaptiveCNNPredictor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def cnn_predict(model, image, device):
    with torch.no_grad():
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
        prediction = model(image_tensor)
    return prediction.squeeze().cpu().numpy()

def pee_extracting(stego_image, modified_positions, embedded_data, cnn_model, device):
    preprocessed_image = preprocess_image(stego_image)
    predicted_image = cnn_predict(cnn_model, preprocessed_image, device)

    extracted_data = []
    restored_image = stego_image.copy()

    for i in range(stego_image.shape[0]):
        for j in range(stego_image.shape[1]):
            if modified_positions[i, j]:
                error = int(stego_image[i, j]) - int(round(predicted_image[i, j]))
                
                if error >= 0:
                    extracted_bit = error % 2
                    restored_value = predicted_image[i, j] + (error // 2)
                else:
                    extracted_bit = (-error) % 2
                    restored_value = predicted_image[i, j] + (error // 2)
                
                extracted_data.append(extracted_bit)
                restored_image[i, j] = int(round(restored_value))
            else:
                restored_image[i, j] = int(round(predicted_image[i, j]))

    return np.array(extracted_data), restored_image

def calculate_psnr(original, restored):
    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, restored):
    from skimage.metrics import structural_similarity as ssim
    return ssim(original, restored, data_range=original.max() - original.min())

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image

def load_numpy_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Numpy file not found: {path}")
    return np.load(path)

if __name__ == "__main__":
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加載 CNN 模型
        model_path = './CNN_PEE/model/adaptive_cnn_predictor.pth'
        cnn_model = load_cnn_model(model_path, device)

        # 讀取 stego 圖像
        stego_image_path = "./CNN_PEE/stego/Tank_stego.png"
        stego_image = load_image(stego_image_path)
        
        # 讀取修改位置和嵌入數據
        modified_positions = load_numpy_file("./CNN_PEE/stego/modified_positions.npy")
        embedded_data = load_numpy_file("./CNN_PEE/stego/embedded_data.npy")

        # 計算嵌入數據的總位元數
        total_embedded_bits = np.sum(embedded_data)

        # 提取數據和還原圖像
        extracted_data, restored_image = pee_extracting(stego_image, modified_positions, embedded_data, cnn_model, device)

        # 讀取原始圖像和原始秘密數據（用於比較）
        original_image_path = "./CNN_PEE/origin/Tank.png"
        original_image = load_image(original_image_path)
        original_secret_data = load_numpy_file("./CNN_PEE/stego/original_secret_data.npy")

        # 計算 PSNR 和 SSIM
        psnr = calculate_psnr(original_image, restored_image)
        ssim = calculate_ssim(original_image, restored_image)

        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM: {ssim:.6f}")

        # 驗證提取的數據
        print(f"嵌入數據長度: {len(original_secret_data)}, 提取的數據長度: {len(extracted_data)}")

        if len(extracted_data) == len(original_secret_data):
            data_match = np.array_equal(extracted_data, original_secret_data)
            print(f"提取的數據與原始數據完全匹配: {data_match}")
            
            if not data_match:
                mismatched_count = np.sum(extracted_data != original_secret_data)
                print(f"不匹配的位元數: {mismatched_count}")
                print(f"不匹配率: {mismatched_count / len(original_secret_data):.4f}")
                
                # 顯示前幾個不匹配的位置
                mismatch_indices = np.where(extracted_data != original_secret_data)[0]
                print("前10個不匹配的位置及其值：")
                for i in range(min(10, len(mismatch_indices))):
                    idx = mismatch_indices[i]
                    print(f"位置 {idx}: 提取值 {extracted_data[idx]}, 原始值 {original_secret_data[idx]}")
        else:
            print("提取的數據長度與原始數據長度不匹配")
            print(f"提取數據的前20個值: {extracted_data[:20]}")

        # 驗證還原的圖像
        image_match = np.array_equal(restored_image, original_image)
        print(f"還原的圖像與原始圖像完全匹配: {image_match}")

        if not image_match:
            diff = np.abs(restored_image.astype(int) - original_image.astype(int))
            print(f"不匹配的像素數: {np.sum(diff != 0)}")
            print(f"最大像素差異: {np.max(diff)}")
            print(f"平均像素差異: {np.mean(diff):.4f}")
            
            # 顯示一些不匹配像素的詳細資訊
            mismatch_indices = np.where(diff != 0)
            for i in range(min(10, len(mismatch_indices[0]))):  # 顯示前10個不匹配的像素
                x, y = mismatch_indices[0][i], mismatch_indices[1][i]
                print(f"位置 ({x}, {y}): 原始值 {original_image[x, y]}, 還原值 {restored_image[x, y]}")

        # 保存還原的圖像
        cv2.imwrite("./CNN_PEE/restored/Tank_restored.png", restored_image)

        print("解碼完成。還原的圖像已保存。")

    except Exception as e:
        print(f"發生未預期的錯誤：{e}")
        import traceback
        traceback.print_exc()