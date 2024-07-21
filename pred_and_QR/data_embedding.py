import numpy as np
import math

def calculate_payload(array2D, a):
    """計算陣列中正負a之間的累加值"""
    row, column = array2D.shape
    payload = 0
    r = np.floor(a/2)
    func = a % 2
    for j in range(1, row-1):
        for i in range(1, column-1):
            if func == 0 or r == 0:
                if -r < array2D[j,i] <= r:
                    payload += 1
            if func == 1:
                if -r <= array2D[j,i] <= r:
                    payload += 1
    return payload

def image_difference_shift(array2D, a):
    """比限制值大的加1，小的減1"""
    row, column = array2D.shape
    array2D_s = array2D.copy()
    func = a % 2
    r = np.floor(a/2)
    for j in range(1, row-1):
        for i in range(1, column-1):
            value = array2D[j,i]
            shift = value
            if func == 1 or a == 1:
                if value > r:
                    shift = value + r
                elif value < -r:
                    shift = value - r - 1
            elif func == 0:
                if value > r:
                    shift = value + r
                elif value < (-r+1):
                    shift = value - r
            array2D_s[j,i] = shift
    return array2D_s

def image_difference_embeding(array2D, array1D, a, flag):
    row, column = array2D.shape
    array2D_e = array2D.copy()
    inf = []
    r = int(a/2)
    func = a % 2
    h = 0
    for i in range(r, -1, -1):
        if i == 0:
            bit01 = [0, -1]
        else:
            bit01 = [2*i-1, 2*i, -2*i, -2*i-1]
        for y in range(1, row-1):
            for x in range(1, column-1):
                embed = array2D[y,x]
                value = embed
                if embed in bit01:
                    w = 0 if flag == 0 else (array1D[h] if h < len(array1D) else 0)
                    b = bit01.index(embed) % 2
                    if w != b:
                        value = bit01[bit01.index(embed) ^ 1]
                    if flag == 1:
                        h += 1
                    inf.append(w)
                array2D_e[y,x] = value
    print(f"差值直方圖嵌入的前20位信息: {inf[:20]}")
    return array2D_e, inf

def decode_image_difference_embeding(array2D, a, original_length):
    row, column = array2D.shape
    deArray = array2D.copy()
    inf = []
    r = int(a/2)
    func = a % 2
    for i in range(r, -1, -1):
        if i == 0:
            bit01 = [0, -1]
        else:
            bit01 = [2*i-1, 2*i, -2*i, -2*i-1]
        for y in range(1, row-1):
            for x in range(1, column-1):
                embed = array2D[y,x]
                if embed in bit01:
                    w = bit01.index(embed) % 2
                    inf.append(w)
                    if len(inf) >= original_length:
                        break
            if len(inf) >= original_length:
                break
        if len(inf) >= original_length:
            break
    
    inf = inf[:original_length]
    print(f"差值直方圖解碼的前20位信息: {inf[:20]}")
    return deArray, inf

def decode_image_different_shift(array2D, a):
    """復原已平移直方圖"""
    row, column = array2D.shape
    deArray = array2D.copy()
    r = np.floor(a/2)
    func = a % 2
    for y in range(row):
        for x in range(column):
            shift = array2D[y,x]
            value = shift
            if func == 1:
                if shift > r:
                    value = shift - r
                elif shift < -r:
                    value = shift + r + 1
            elif func == 0:
                if shift > r:
                    value = shift - r
                elif shift < -r:
                    value = shift + r
            deArray[y,x] = value
    return deArray

def two_value_difference_expansion(left, right, hide):
    """兩像素作差值嵌入"""
    l = np.floor((left + right) / 2)
    if left >= right:
        h = left - right
        h_e = 2 * h + hide
        left_e = l + np.floor((h_e + 1) / 2)
        right_e = l - np.floor(h_e / 2)
    elif left < right:
        h = right - left
        h_e = 2 * h + hide
        left_e = l - np.floor(h_e / 2)
        right_e = l + np.floor((h_e + 1) / 2)
    return left_e, right_e

def different_expansion_embeding_array1D(img, array1D):
    """將一維陣列用差值嵌入法進影像中"""
    height, width = img.shape
    markedImg = img.copy()
    hidemap = np.zeros((height, width))
    length = len(array1D)
    locationmap = []
    payload_DE = 0
    i = 0
    sum = 0
    for y in range(height):
        for x in range(0, width-1, 2):
            if i < length:
                b = array1D[i]
                flag = 1
            elif i >= length:
                b = 0
                flag = 0
            left = int(img[y,x])
            right = int(img[y,x+1])
            left_e, right_e = two_value_difference_expansion(left, right, b)
            if 0 <= left_e <= 255 and 0 <= right_e <= 255:
                payload_DE += 1
                if flag == 1:
                    markedImg[y,x] = left_e
                    markedImg[y,x+1] = right_e
                    hidemap[y,x] = 255
                    hidemap[y,x+1] = 255
                    i += 1
                elif flag == 0:
                    continue
            else:
                if flag == 1:
                    locationmap.append(y)
                    locationmap.append(x)
                    sum += 1
    return markedImg, hidemap, locationmap, payload_DE

def two_value_decode_different_expansion(left_e, right_e):
    """兩像素作差值解碼"""
    if left_e >= right_e:
        sign = 1
    elif left_e < right_e:
        sign = -1
    l_e = np.floor((left_e + right_e) / 2)
    h_e = sign * np.floor(left_e - right_e)
    if h_e % 2 == 0:
        b = 0
    elif h_e % 2 == 1:
        b = 1
    h = np.floor(h_e / 2)
    if left_e >= right_e:
        left = l_e + np.floor((h + 1) / 2)
        right = l_e - np.floor(h / 2)
    elif left_e < right_e:
        right = l_e + np.floor((h + 1) / 2)
        left = l_e - np.floor(h / 2)
    return b, left, right

def decode_different_expansion(img, hidemap):
    """解碼差值擴展嵌入"""
    height, width = img.shape
    decodeImg = img.copy()
    exInf = []
    for y in range(height):
        x = 0
        while x <= width - 2:
            if hidemap[y,x] != 255:
                x += 1
                continue
            left_e = int(img[y,x])
            right_e = int(img[y,x+1])
            b, left, right = two_value_decode_different_expansion(left_e, right_e)
            decodeImg[y,x] = left
            decodeImg[y,x+1] = right
            exInf.append(b)
            x += 2
    return decodeImg, exInf

def histogram_data_hiding(img, flag, array1D):
    """直方圖平移嵌入"""
    h_img, w_img = img.shape
    markedImg = img.copy()
    hist = [0] * 256
    for y in range(h_img):
        for x in range(w_img):
            hist[int(img[y,x])] += 1  # 將像素值轉換為整數
    peak = hist.index(max(hist))
    payload_h = hist[peak]
    i = 0
    length = len(array1D)
    zero = next((i for i, x in enumerate(hist) if x == 0), None)
    for y in range(h_img):
        for x in range(w_img):
            if i < length:
                b = array1D[i]
            else:
                b = 0
            value = int(img[y,x])  # 將像素值轉換為整數
            if flag == 0:
                if value < peak:
                    value -= 1
                elif value == peak:
                    value -= b
            elif flag == 1:
                if peak < value < zero:
                    value += 1
                elif value == peak:
                    value += b
            markedImg[y,x] = value
            if value != int(img[y,x]):
                i += 1
    return markedImg, peak, payload_h

# 如果需要直接運行此文件進行測試
if __name__ == "__main__":
    # 這裡可以添加一些測試代碼
    # 例如:
    test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    test_data = np.random.randint(0, 2, 1000)
    embedded_image, hidemap, locationmap, payload = different_expansion_embeding_array1D(test_image, test_data)
    print(f"Embedded payload: {payload}")
    print(f"Location map size: {len(locationmap)}")