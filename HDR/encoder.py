import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt

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

# 讓用戶輸入圖像名稱
image_name = input("請輸入HDR圖像的名稱（包含檔案擴展名，例如：image.hdr）: ")
image_path = os.path.join('./HDR/HDR images', image_name)

# 讀取並轉換HDR影像
rgbe_image = read_and_convert_hdr_to_rgbe(image_path)

if rgbe_image is not None:
    print("HDR圖像讀取和轉換為RGBE格式完成。")
    
    # Calculate the predicted E values for the image
    predicted_E = predict_e_values(rgbe_image)
    
    # Calculate the prediction error for each pixel (absolute difference)
    prediction_error = np.abs(rgbe_image[:, :, 3] - predicted_E)
    
    # Create an intermediate prediction error image (except for the first row and column)
    prediction_error_image = np.copy(rgbe_image)
    prediction_error_image[1:, 1:, 3] = prediction_error[1:, 1:]
    
    # Now you have the prediction error image that can be used for further processing
    
    plot_histogram_of_e_values(rgbe_image)
else:
    print("未能進行轉換。")
