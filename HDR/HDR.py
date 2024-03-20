import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 导入自定义的图像质量评估函数
# 注意：如果这些函数位于PU21.py中，确保从该文件导入
from PU21 import pu21_quality_assessment

def adaptive_tonemap(hdr_image, gamma=2.2, saturation=1.0, bias=0.85):
    # 將HDR影像轉換為浮點數格式
    hdr_image = hdr_image.astype(np.float32)

    # 使用Drago tone mapping算法
    tonemap = cv2.createTonemapDrago(gamma=gamma, saturation=saturation, bias=bias)

    # 進行tone mapping
    ldr_image = tonemap.process(hdr_image)

    return ldr_image

def adjust_exposure(ldr_image, exposure=0):
    # 調整LDR影像的曝光
    ldr_image = cv2.pow(ldr_image, 2.0 ** exposure)
    return ldr_image

def float_to_rgbe(hdr_image):
    rgbe_image = np.zeros((*hdr_image.shape[:2], 4), dtype=np.uint8)
    max_rgb = np.max(hdr_image[:, :, :3], axis=2)
    valid_mask = max_rgb >= 1e-32  # 避免除以0

    exp = np.floor(np.log2(max_rgb[valid_mask])) + 128
    exp = np.clip(exp, 128 - 128, 128 + 127).astype(np.uint8)

    scale_factor = (255.0 / max_rgb[valid_mask]) * (2.0 ** (exp - 128))
    scaled_rgb = hdr_image[:, :, :3][valid_mask] * np.expand_dims(scale_factor, axis=-1)
    scaled_rgb = np.clip(scaled_rgb, 0, 255).astype(np.uint8)

    rgbe_image[:, :, :3][valid_mask] = scaled_rgb
    rgbe_image[:, :, 3][valid_mask] = exp

    return rgbe_image

def rgbe_to_float(rgbe_image):
    """
    将RGBE格式的图像转换回浮点数格式的HDR图像。
    """
    # 分离RGB和E通道
    rgb = rgbe_image[:, :, :3].astype(np.float32)
    e = rgbe_image[:, :, 3].astype(np.float32)

    # 将指数E转换回浮点数的比例因子
    scale = 2.0 ** (e - 128.0)
    scale = np.expand_dims(scale, axis=-1)  # 使其形状与rgb数组匹配

    # 逆向计算原始的浮点数RGB值
    hdr_image = rgb * scale / 255.0

    return hdr_image

def read_and_convert_hdr_to_rgbe(image_path):
    if not os.path.exists(image_path):
        print("檔案不存在，請確認檔案名稱和路徑是否正確。")
        return None
    hdr_image = cv2.imread(image_path, flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    rgbe_image = float_to_rgbe(hdr_image)
    return rgbe_image

def plot_histogram_of_e_values(rgbe_image):
    E_values = rgbe_image[:, :, 3].flatten()
    plt.hist(E_values, bins=256, range=(0, 255), color='blue', alpha=0.7)
    plt.title('Histogram of E values in the Image')
    plt.xlabel('E value')
    plt.ylabel('Frequency')
    plt.show()
    
def predict_e_values(rgbe_image):
    # Create a copy of the E channel
    E = rgbe_image[:, :, 3].astype(np.int32)
    
    # Initialize the predicted E values array with the same shape as the E channel
    predicted_E = np.zeros_like(E)
    
    # Iterate over the image starting from the second row and second column
    # because the first row and first column do not have left and top neighbors
    for i in range(1, E.shape[0]):
        for j in range(1, E.shape[1]):
            # Neighboring values
            left = E[i, j-1]
            top = E[i-1, j]

            # Apply the prediction rules based on the neighbor values
            if left <= E[i, j] and top <= E[i, j]:
                predicted_E[i, j] = max(left, top)
            elif left >= E[i, j] and top >= E[i, j]:
                predicted_E[i, j] = min(left, top)
            else:
                predicted_E[i, j] = (left + top) // 2

    return predicted_E

def plot_e_value_histogram(rgbe_image, image_name, directory="./HDR/histogram"):
    # 從RGBE圖像中提取E值
    e_values = rgbe_image[:, :, 3].flatten()
    
    # 創建一個新的圖形和軸
    fig, ax = plt.subplots()
    
    # 繪製E值的直方圖
    ax.hist(e_values, bins=256, range=(0, 255), color='blue', alpha=0.7)
    
    # 設置圖形的標題和軸標籤
    ax.set_title('Histogram of E Values')
    ax.set_xlabel('E Value')
    ax.set_ylabel('Frequency')
    
    # 從影像名稱中移除副檔名
    image_name_without_extension = os.path.splitext(image_name)[0]
    
    # 生成直方圖的檔名
    histogram_filename = f"{image_name_without_extension}_e_value_histogram.png"
    
    # 創建目錄(如果不存在)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 構建完整的檔案路徑
    file_path = os.path.join(directory, histogram_filename)
    
    # 儲存直方圖為圖片檔
    plt.savefig(file_path)
    print(f"E值直方圖已儲存至 {file_path}")
    
    # 關閉圖形以釋放資源
    plt.close(fig)

def generate_label_map(rgbe_image, predicted_E):
    # 計算預測誤差
    prediction_error = rgbe_image[:, :, 3].astype(np.int32) - predicted_E
    
    # 初始化label map和embedding capacity陣列
    label_map = np.zeros_like(prediction_error)
    embedding_capacity = np.zeros_like(prediction_error)
    
    # 迭代预测误差
    for i in range(prediction_error.shape[0]):
        for j in range(prediction_error.shape[1]):
            # 计算预测误差绝对值的二进制表示中前导0的个数
            binary_representation = format(abs(prediction_error[i, j]), '08b')
            num_leading_zeros = len(binary_representation.split('1')[0])
            
            # 根据预测误差和原始E值的大小关系确定label值
            if prediction_error[i, j] >= 0:
                if rgbe_image[i, j, 3] >= predicted_E[i, j]:
                    label_map[i, j] = num_leading_zeros
                else:
                    label_map[i, j] = -num_leading_zeros
            else:
                if rgbe_image[i, j, 3] < predicted_E[i, j]:
                    label_map[i, j] = -num_leading_zeros
                else:
                    label_map[i, j] = num_leading_zeros
            
            # 计算embedding capacity
            embedding_capacity[i, j] = min(num_leading_zeros + 1, 8)
    
    return label_map, embedding_capacity

def save_label_map(label_map, image_name, directory="./HDR/label map"):
    # 從影像名稱中移除副檔名
    image_name_without_extension = os.path.splitext(image_name)[0]
    
    # 生成label map的檔名
    label_map_filename = f"{image_name_without_extension}_label_map.npy"
    
    # 創建目錄(如果不存在)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 構建完整的檔案路徑
    file_path = os.path.join(directory, label_map_filename)
    
    # 將label map儲存為.npy檔案
    np.save(file_path, label_map)
    print(f"Label map已儲存至 {file_path}")

def main():
    image_name = input("請輸入HDR圖像的名稱（預設副檔名為.hdr）: ")
    
    # 檢查使用者輸入的圖像名稱是否包含副檔名
    if '.' not in image_name:
        # 如果使用者沒有輸入副檔名,則預設為.hdr
        image_name += '.hdr'
    
    image_path = os.path.join('./HDR/HDR images', image_name)
    
    if not os.path.exists(image_path):
        print("檔案不存在,請確認檔案名稱和路徑是否正確。")
        return

    # 使用OpenCV的cv2.imread()函數讀取HDR影像
    hdr_image = cv2.imread(image_path, flags=cv2.IMREAD_ANYDEPTH)
    
    if hdr_image is None:
        print("無法讀取HDR圖像。")
        return
    
    print("HDR圖像讀取完成。")

    # 使用Drago tone mapping算法進行tone mapping
    ldr_image = adaptive_tonemap(hdr_image)

    # 初始曝光值
    exposure = 0

    while True:
        # 調整LDR影像的曝光
        adjusted_ldr_image = adjust_exposure(ldr_image, exposure)

        # 將adjusted_ldr_image的值限制在[0, 1]範圍內
        adjusted_ldr_image = np.clip(adjusted_ldr_image, 0, 1)

        # 將LDR影像轉換為8位整數格式
        adjusted_ldr_image_8bit = (adjusted_ldr_image * 255).astype(np.uint8)

        # 顯示調整後的LDR影像
        cv2.imshow('Adjusted LDR Image', adjusted_ldr_image_8bit)

        # 等待按鍵事件
        key = cv2.waitKey(1) & 0xFF

        # 如果按下 'q' 鍵,則退出迴圈
        if key == ord('q'):
            break
        # 如果按下 '+' 鍵,增加曝光值
        elif key == ord('+'):
            exposure += 0.1
        # 如果按下 '-' 鍵,減少曝光值
        elif key == ord('-'):
            exposure -= 0.1

    cv2.destroyAllWindows()

    # 將HDR影像轉換為RGBE格式
    rgbe_image = float_to_rgbe(hdr_image)
    
    # 繪製E值的直方圖並儲存
    plot_e_value_histogram(rgbe_image, image_name)
    
    # 預測RGBE格式影像中的E值
    predicted_E = predict_e_values(rgbe_image)
    
    # 計算預測誤差
    prediction_error = rgbe_image[:, :, 3].astype(np.int32) - predicted_E
    
    # 將預測誤差合併到RGBE影像中
    prediction_error_image = rgbe_image.copy()
    prediction_error_image[:, :, 3] = prediction_error
    
    # 生成label map和embedding capacity
    label_map, embedding_capacity = generate_label_map(rgbe_image, predicted_E)
    
    # 儲存label map
    save_label_map(label_map, image_name)

if __name__ == "__main__":
    main()