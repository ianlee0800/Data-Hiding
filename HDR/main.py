import cv2
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from utils import float_to_rgbe, rgbe_to_float
from PU21 import *

sys.path.append('./DCT')  # 將 DCT2 目錄添加到 Python 模組搜尋路徑
from image2RLE import *

def read_hdr_image(image_name):
    image_path = os.path.join("./HDR/HDR images", f"{image_name}.hdr")
    if not os.path.exists(image_path):
        print("檔案不存在,請確認檔案名稱和路徑是否正確。")
        return None
    hdr_image = cv2.imread(image_path, flags=cv2.IMREAD_ANYDEPTH)
    return hdr_image

def save_hdr_image(rgbe_image, filename):
    # 將RGBE圖像轉換回浮點數格式
    hdr_image = rgbe_to_float(rgbe_image)
    
    # 將浮點數格式的HDR圖像儲存為HDR檔案
    cv2.imwrite(filename, hdr_image)

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

def read_hdr_image(image_name):
    image_path = os.path.join("./HDR/HDR images", f"{image_name}.hdr")
    if not os.path.exists(image_path):
        print("檔案不存在,請確認檔案名稱和路徑是否正確。")
        return None
    hdr_image = cv2.imread(image_path, flags=cv2.IMREAD_ANYDEPTH)
    return hdr_image

def save_hdr_image(image, filename):
    cv2.imwrite(filename, image)

def main():
    # 詢問使用者要讀取的圖像名稱
    image_name = input("請輸入要讀取的圖像名稱(不包含副檔名): ")
    
    # 讀取HDR圖像
    hdr_image = read_hdr_image(image_name)
    
    if hdr_image is None:
        return
    
    # 將HDR圖像轉換為RGBE格式
    rgbe_image = float_to_rgbe(hdr_image)
    
    # 從RGBE圖像中提取RGB通道的數值
    r_channel = rgbe_image[:, :, 0].astype(np.float32)
    g_channel = rgbe_image[:, :, 1].astype(np.float32)
    b_channel = rgbe_image[:, :, 2].astype(np.float32)
    
    # 對RGB通道進行DCT轉換
    dct_r = cv2.dct(r_channel)
    dct_g = cv2.dct(g_channel)
    dct_b = cv2.dct(b_channel)
    
    # 將DCT係數組合成一個三通道圖像
    dct_image = cv2.merge((dct_r, dct_g, dct_b))
    
    # 將DCT係數轉換為RGBE格式
    dct_rgbe = float_to_rgbe(dct_image)
    
    # 將RGBE格式的DCT係數轉換回浮點數格式
    dct_float = rgbe_to_float(dct_rgbe)
    
    # 建立 ./HDR/dct 目錄(如果不存在)
    os.makedirs("./HDR/dct", exist_ok=True)
    
    # 儲存DCT係數為HDR格式
    dct_filename = os.path.join("./HDR/dct", f"dct_{image_name}.hdr")
    save_hdr_image(dct_float, dct_filename)
    
    # 顯示DCT係數圖像
    cv2.imshow("DCT Image", dct_image / np.max(dct_image))
    
    # 對DCT係數進行Inverse DCT轉換
    idct_r = cv2.idct(dct_r)
    idct_g = cv2.idct(dct_g)
    idct_b = cv2.idct(dct_b)
    
    # 將Inverse DCT結果組合成一個三通道圖像
    idct_image = cv2.merge((idct_r, idct_g, idct_b))
    
    # 將Inverse DCT結果轉換為RGBE格式
    idct_rgbe = float_to_rgbe(idct_image)
    
    # 將RGBE格式的Inverse DCT結果轉換回浮點數格式
    idct_float = rgbe_to_float(idct_rgbe)
    
    # 建立 ./HDR/idct 目錄(如果不存在)
    os.makedirs("./HDR/idct", exist_ok=True)
    
    # 儲存Inverse DCT結果為HDR格式
    idct_filename = os.path.join("./HDR/idct", f"idct_{image_name}.hdr")
    save_hdr_image(idct_float, idct_filename)
    
    # 顯示Inverse DCT結果圖像
    cv2.imshow("IDCT Image", idct_image / np.max(idct_image))
    
    print("DCT係數已儲存為:", dct_filename)
    print("Inverse DCT結果已儲存為:", idct_filename)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()