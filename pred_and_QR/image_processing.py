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

def two_array2D_add_or_subtract(array2D_1, array2D_2, sign):
    """
    兩個二維陣列的數值相加或相減
    """
    row, column = array2D_1.shape
    diff = np.zeros((row, column))
    for j in range(row):
        for i in range(column):
            diff[j,i] = int(array2D_1[j,i]) + sign*int(array2D_2[j,i])
    return diff

def find_max(array1D):
    """
    找出一維陣列中最大值的索引
    """
    max_index = 0 
    for i in range(1, len(array1D)):
        if array1D[i] > array1D[max_index]:
            max_index = i
    return max_index

def find_w(image):
    """
    找出模塊的大小
    """
    height, width = image.shape
    RLArray = [0] * height
    for y in range(height):
        RunLength = 0
        for x in range(width-1):
            color1 = image[y,x]
            color2 = image[y,x+1]
            if color1 == color2:
                RunLength += 1
            elif color1 != color2:
                RLArray[RunLength+1] += 1
                RunLength = 0
    w = find_max(RLArray)
    return w

def array2D_transfer_to_array1D(array2D):
    """
    二維陣列轉換為一維陣列
    """
    array1D = []
    row, column = array2D.shape
    for y in range(row):
        for x in range(column):
            value = array2D[y,x]
            if value < 128:
                array1D.append(0)
            elif value >= 128:
                array1D.append(1)
    return array1D

def array1D_transfer_to_array2D(array1D):
    """
    一維陣列轉換為二維陣列
    """
    length = len(array1D)
    side = int(length**0.5)
    array2D = np.zeros((side, side))
    i = 0
    for y in range(side):
        for x in range(side):
            value = array1D[i]
            if value == 1:
                value = 255
            array2D[y,x] = value
            i += 1
    return array2D

def same_array1D(array1, array2):
    """
    檢查兩個一維陣列是否相同
    """
    if len(array1) != len(array2):
        return False
    return all(a == b for a, b in zip(array1, array2))

def same_array2D(array1, array2):
    """
    檢查兩個二維陣列是否相同
    """
    if array1.shape != array2.shape:
        return False
    return np.all(array1 == array2)

# 如果需要直接運行此文件進行測試
if __name__ == "__main__":
    # 這裡可以添加一些測試代碼
    # 例如:
    test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    processed_image = preprocess_image(test_image)
    histogram = generate_histogram(processed_image)
    print("Histogram sample:", histogram[:10])  # 打印直方圖的前10個值