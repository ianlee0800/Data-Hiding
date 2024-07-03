import torch
import torch.nn as nn
import cv2
import numpy as np
from scipy.signal import convolve2d
from CNN import AdaptiveCNNPredictor, preprocess_image
import heapq

# 加載訓練好的 CNN 模型
def load_cnn_model(model_path, device):
    model = AdaptiveCNNPredictor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# CNN 預測函數
def cnn_predict(model, image, device):
    with torch.no_grad():
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
        prediction = model(image_tensor)
    return prediction.squeeze().cpu().numpy()

def get_edge_map(image, threshold=30):
    edges = ndimage.sobel(image)
    return edges > threshold

# PEE 嵌入函數
def pee_embedding(cover_image, secret_data, cnn_model, device):
    preprocessed_image = preprocess_image(cover_image)
    predicted_image = cnn_predict(cnn_model, preprocessed_image, device)
    prediction_errors = cover_image.astype(int) - predicted_image.astype(int)

    modified_positions = np.zeros_like(cover_image, dtype=bool)
    embedded_data = np.zeros_like(cover_image, dtype=int)

    stego_image = cover_image.copy()
    data_index = 0
    
    abs_errors = np.abs(prediction_errors)
    sorted_indices = np.argsort(abs_errors.flatten())
    
    for idx in sorted_indices:
        if data_index >= len(secret_data):
            break
        
        i, j = np.unravel_index(idx, cover_image.shape)
        error = prediction_errors[i, j]
        
        bit_to_embed = secret_data[data_index]
        if error >= 0:
            stego_image[i, j] = predicted_image[i, j] + 2 * error + bit_to_embed
        else:
            stego_image[i, j] = predicted_image[i, j] + 2 * error - bit_to_embed
        
        modified_positions[i, j] = True
        embedded_data[i, j] = 1
        data_index += 1

    # 保存原始的秘密數據
    np.save("./CNN_PEE/stego/original_secret_data.npy", secret_data)

    return stego_image, modified_positions, embedded_data, data_index

# 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加載 CNN 模型
    cnn_model = load_cnn_model('./CNN_PEE/model/adaptive_cnn_predictor.pth', device)

    # 讀取封面圖像
    cover_image_path = "./CNN_PEE/origin/Tank.png"  # 請替換為實際的圖像路徑
    cover_image = cv2.imread(cover_image_path, cv2.IMREAD_GRAYSCALE)

    # 生成秘密數據（示例）
    payload_size = 80000
    secret_data = np.random.randint(0, 2, payload_size)

    # 執行 PEE 嵌入
    stego_image, modified_positions, embedded_data, embedded_bits = pee_embedding(cover_image, secret_data, cnn_model, device)
    
    print(f"實際嵌入的位元數: {embedded_bits}")
    print(f"嵌入率: {embedded_bits / (cover_image.shape[0] * cover_image.shape[1]):.4f} bpp")
    
    # 保存 stego 圖像
    cv2.imwrite("./CNN_PEE/stego/Tank_stego.png", stego_image)

    # 保存修改位置和嵌入數據（用於提取）
    np.save("./CNN_PEE/stego/modified_positions.npy", modified_positions)
    np.save("./CNN_PEE/stego/embedded_data.npy", embedded_data)

    print("嵌入完成。Stego 圖像已保存。")

    # 計算 PSNR
    mse = np.mean((cover_image - stego_image) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    print(f"PSNR: {psnr:.2f} dB")
