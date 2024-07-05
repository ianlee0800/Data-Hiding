import torch
import torch.nn as nn
import cv2
import numpy as np
from scipy.signal import convolve2d
from CNN import preprocess_image, predict, AdaptiveCNNPredictor
import heapq
import scipy.ndimage as ndimage

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

def pee_embedding(cover_image, secret_data, model_path, device):
    print("Original secret data:", secret_data)
    secret_data = secret_data.astype(int)
    print("Secret data after type conversion:", secret_data)
    
    # 加载模型
    model = AdaptiveCNNPredictor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    preprocessed_image = preprocess_image(cover_image)
    predicted_image = predict(model, preprocessed_image, device)

    embedding_info = []
    stego_image = cover_image.copy()
    data_index = 0

    for i in range(cover_image.shape[0]):
        for j in range(cover_image.shape[1]):
            if data_index >= len(secret_data):
                break
            
            original_value = int(cover_image[i, j])
            predicted_value = int(round(predicted_image[i, j]))
            error = original_value - predicted_value
            bit_to_embed = secret_data[data_index]

            if error >= 0:
                stego_value = predicted_value + 2 * error + bit_to_embed
            else:
                stego_value = predicted_value + 2 * error - bit_to_embed

            stego_image[i, j] = stego_value
            
            embedding_info.append({
                'position': (i, j),
                'original_value': original_value,
                'predicted_value': predicted_value,
                'error': error,
                'embedded_bit': bit_to_embed,
                'stego_value': stego_value
            })

            print(f"位置 ({i}, {j}): 原始值 = {original_value}, 预测值 = {predicted_value}, 错误 = {error}")
            print(f"待嵌入位 = {bit_to_embed}, 计算得到的stego值 = {stego_value}")
            print(f"验证: {stego_value - predicted_value - 2 * error if error >= 0 else -(stego_value - predicted_value - 2 * error)}")

            data_index += 1

    # 验证嵌入的数据
    embedded_bits = [info['embedded_bit'] for info in embedding_info]
    if not np.array_equal(embedded_bits, secret_data[:len(embedded_bits)]):
        print("警告：嵌入的位与原始秘密数据不匹配")
        for i, (embedded, original) in enumerate(zip(embedded_bits, secret_data[:len(embedded_bits)])):
            if embedded != original:
                print(f"位置 {i}: 嵌入 {embedded}, 原始 {original}")
                print(f"详细信息: {embedding_info[i]}")

    # 保存并验证嵌入信息
    np.save("./CNN_PEE/stego/embedding_info.npy", embedding_info)
    loaded_info = np.load("./CNN_PEE/stego/embedding_info.npy", allow_pickle=True)
    if not all(a == b for a, b in zip(embedding_info, loaded_info)):
        print("警告：保存和加载的嵌入信息不匹配")

    # 保存 stego 图像的像素值
    np.save("./CNN_PEE/stego/stego_values.npy", stego_image)

    print(f"嵌入位置数量: {len(embedding_info)}")
    print(f"嵌入错误范围: {min([info['error'] for info in embedding_info])} 到 {max([info['error'] for info in embedding_info])}")
    print(f"原始秘密数据: {secret_data}")
    print(f"嵌入的秘密数据: {embedded_bits}")
    
    np.save("./CNN_PEE/stego/original_secret_data.npy", secret_data)

    return stego_image, embedding_info, data_index

# 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加載 CNN 模型
    cnn_model = load_cnn_model('./CNN_PEE/model/adaptive_cnn_predictor.pth', device)

    # 讀取封面圖像
    cover_image_path = "./CNN_PEE/origin/Tank.png"  # 請替換為實際的圖像路徑
    cover_image = cv2.imread(cover_image_path, cv2.IMREAD_GRAYSCALE)

    # 生成秘密數據（示例）
    payload_size = 10000
    secret_data = np.random.randint(0, 2, payload_size)

    # 執行 PEE 嵌入
    model_path = './CNN_PEE/model/adaptive_cnn_predictor.pth'
    stego_image, embedding_info, embedded_bits = pee_embedding(cover_image, secret_data, model_path, device)
    
    print(f"實際嵌入的位元數: {embedded_bits}")
    print(f"嵌入率: {embedded_bits / (cover_image.shape[0] * cover_image.shape[1]):.4f} bpp")
    
    # 保存 stego 圖像
    cv2.imwrite("./CNN_PEE/stego/Tank_stego.png", stego_image)
    np.save("./CNN_PEE/stego/stego_values.npy", stego_image)

    # 保存嵌入信息
    np.save("./CNN_PEE/stego/embedding_info.npy", embedding_info)

    print("嵌入完成。Stego 圖像已保存。")

    # 計算 PSNR
    mse = np.mean((cover_image - stego_image) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    print(f"PSNR: {psnr:.2f} dB")
