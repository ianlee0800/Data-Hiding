import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# 导入自定义的图像质量评估函数
# 注意：如果这些函数位于PU21.py中，确保从该文件导入
from PU21 import pu21_quality_assessment

def adaptive_tonemap(hdr_image):
    # 將HDR影像轉換為浮點數格式
    hdr_image = hdr_image.astype(np.float32)

    # 計算HDR影像的亮度值
    luminance = 0.2126 * hdr_image[:, :, 2] + 0.7152 * hdr_image[:, :, 1] + 0.0722 * hdr_image[:, :, 0]

    # 計算亮度值的對數平均值
    log_mean_luminance = np.exp(np.mean(np.log(luminance + 1e-6)))

    # 根據對數平均亮度值自適應調整gamma值
    gamma = 1.0 / (1.0 + np.log10(log_mean_luminance))

    # 使用自適應gamma值進行tone mapping
    tonemap = cv2.createTonemap(gamma=gamma)
    ldr_image = tonemap.process(hdr_image)

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

def generate_label_map(prediction_error_image):
    # Get the E channel prediction errors
    prediction_errors = prediction_error_image[:, :, 3]

    # Initialize the label map and embedding capacity arrays
    label_map = np.zeros_like(prediction_errors)
    embedding_capacity = np.zeros_like(prediction_errors)
    
    # Iterate through the prediction errors
    for i in range(prediction_errors.shape[0]):
        for j in range(prediction_errors.shape[1]):
            # Calculate the number of leading zeros in the binary representation
            # of the absolute prediction error value
            binary_representation = format(abs(prediction_errors[i, j]), '08b')
            leading_zeros = len(binary_representation) - len(binary_representation.lstrip('0'))
            
            # Determine the label based on the sign of the prediction error
            if prediction_errors[i, j] >= 0:
                label_map[i, j] = leading_zeros
            else:
                label_map[i, j] = -leading_zeros
            
            # Calculate the embedding capacity
            embedding_capacity[i, j] = min(leading_zeros + 1, 8)
    
    return label_map, embedding_capacity

def save_label_map(label_map, directory="./HDR/label map", filename="label_map.npy"):
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Construct the full file path
    file_path = os.path.join(directory, filename)
    
    # Save the label map as a .npy file
    np.save(file_path, label_map)
    print(f"Label map saved to {file_path}")

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

    # 使用自適應tone mapping優化顯示效果
    ldr_image = adaptive_tonemap(hdr_image)

    # 將LDR影像轉換為8位整數格式
    ldr_image_8bit = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)

    # 顯示自適應tone mapping後的LDR影像
    cv2.imshow('Adaptive Tone Mapped HDR Image', ldr_image_8bit)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()