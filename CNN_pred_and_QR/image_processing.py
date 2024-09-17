import numpy as np
import cv2
from scipy.signal import convolve2d

def preprocess_image(image):
    """
    預處理圖像,包括分割和過濾
    """
    h, w = image.shape
    parts = [
        image[0:h:2, 0:w:2],
        image[0:h:2, 1:w:2],
        image[1:h:2, 0:w:2],
        image[1:h:2, 1:w:2]
    ]
    
    filter_kernel = np.array([
        [1, 2, 3, 2, 1],
        [2, 4, 6, 4, 2],
        [3, 6, 9, 6, 3],
        [2, 4, 6, 4, 2],
        [1, 2, 3, 2, 1]
    ]) / 81  # 預計算規範化因子

    preprocessed_image = np.zeros_like(image)
    for i, part in enumerate(parts):
        filtered_part = convolve2d(part, filter_kernel, mode='same', boundary='symmetric')
        preprocessed_image[i//2::2, i%2::2] = filtered_part

    return preprocessed_image

def generate_histogram(array2D):
    """
    生成二維陣列的直方圖
    """
    num = [0] * 256
    height, width = array2D.shape
    for y in range(height):
        for x in range(width):
            value = int(array2D[y,x])
            num[value] += 1
    return num

def generate_different_histogram_without_frame(array2d, histId, histNum):
    """
    累算二維陣列上的數值並生成直方圖，不包括邊框
    """
    height, width = array2d.shape
    for y in range(1, height-1):
        for x in range(1, width-1):
            value = int(array2d[y,x])
            for i, id_value in enumerate(histId):
                if value == id_value:
                    histNum[i] += 1
    return histId, histNum

def two_array2D_add_or_subtract(array2D_1, array2D_2, sign):
    """
    兩個二維陣列的數值相加或相減
    """
    return array2D_1 + sign * array2D_2

def find_max(array1D):
    """
    找出一維陣列中最大值的索引
    """
    return np.argmax(array1D)

def find_w(image):
    """
    找出模塊的大小
    """
    height, width = image.shape
    RLArray = [0] * height
    for y in range(height):
        RunLength = 0
        for x in range(width-1):
            if image[y,x] == image[y,x+1]:
                RunLength += 1
            else:
                RLArray[RunLength+1] += 1
                RunLength = 0
    w = find_max(RLArray)
    return w

def array2D_transfer_to_array1D(array2D):
    """
    二維陣列轉換為一維陣列
    """
    return (array2D >= 128).flatten().astype(int)

def array1D_transfer_to_array2D(array1D):
    """
    一維陣列轉換為二維陣列
    """
    side = int(np.sqrt(len(array1D)))
    array2D = np.array(array1D).reshape(side, side)
    return array2D * 255

def same_array1D(array1, array2):
    """
    檢查兩個一維陣列是否相同
    """
    return np.array_equal(array1, array2)

def same_array2D(array1, array2):
    """
    檢查兩個二維陣列是否相同
    """
    return np.array_equal(array1, array2)

def image_rotation(image, times):
    """
    將圖像旋轉指定的次數（每次旋轉90度）
    
    參數:
    image: 輸入圖像
    times: 旋轉次數（整數）
    
    返回:
    旋轉後的圖像
    """
    return np.rot90(image, times)

def image_rerotation(image, times):
    """
    將圖像轉回原方向
    
    參數:
    image: 輸入圖像
    times: 原圖像已被90度旋轉的次數
    
    返回:
    轉回原方向的圖像
    """
    return np.rot90(image, -times % 4)

# 如果需要直接運行此文件進行測試
if __name__ == "__main__":
    # 這裡可以添加一些測試代碼
    # 例如:
    test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    processed_image = preprocess_image(test_image)
    histogram = generate_histogram(processed_image)
    print("Histogram sample:", histogram[:10])  # 打印直方圖的前10個值

    # 測試新添加的 generate_different_histogram_without_frame 函數
    histId = list(range(-10, 11))  # 從 -10 到 10 的範圍
    histNum = [0] * len(histId)
    _, diff_histogram = generate_different_histogram_without_frame(processed_image, histId, histNum)
    print("Different histogram sample:", diff_histogram[:10])  # 打印差異直方圖的前10個值