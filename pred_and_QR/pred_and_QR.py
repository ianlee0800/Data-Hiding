import numpy as np
import cv2
import math
import time
import json
import matplotlib.pyplot as plt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

def check_quality_after_stage(stage_name, original_img, embedded_img):
    psnr = calculate_psnr(original_img, embedded_img)
    ssim = calculate_ssim(original_img, embedded_img)
    hist_orig, _, _, _ = generate_histogram(original_img)
    hist_emb, _, _, _ = generate_histogram(embedded_img)
    corr = histogram_correlation(hist_orig, hist_emb)
    print(f"{stage_name}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, Histogram Correlation={corr:.4f}")

def plot_histogram(data, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(data)), data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()
    
def plot_histogram_comparison(hist1, hist2, title1, title2, xlabel, ylabel, filename):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(hist1)), hist1)
    plt.title(title1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.subplot(1, 2, 2)
    plt.bar(range(len(hist2)), hist2)
    plt.title(title2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def calculate_correlation(img1, img2):
    # 將圖像轉換為一維數組
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    # 計算相關係數
    correlation = np.corrcoef(img1_flat, img2_flat)[0, 1]
    
    return round(correlation, 4)

def histogram_correlation(hist1, hist2):
    # 確保兩個直方圖長度相同
    max_length = max(len(hist1), len(hist2))
    hist1_padded = hist1 + [0] * (max_length - len(hist1))
    hist2_padded = hist2 + [0] * (max_length - len(hist2))
    
    # 計算相關係數
    correlation = np.corrcoef(hist1_padded, hist2_padded)[0, 1]
    
    return round(correlation, 4)

#計算峰值訊噪比PSNR
def calculate_psnr(img1, img2):
    height, width = img1.shape
    size_img = height*width
    max = 255
    mse = 0
    for y in range(height):
        for x in range(width):
            bgr1 = int(img1[y,x])
            bgr2 = int(img2[y,x])
            # print(img1, img2)
            diff = bgr1 - bgr2
            mse += diff**2
    mse = mse / size_img
    if mse == 0:
        return 0
    else:
        psnr = 10*math.log10(max**2/mse)
        psnr = round(psnr, 2)
        return psnr
    
#計算結構相似性SSIM
def calculate_ssim(img1, img2):
    height, width = img1.shape
    size_img = height*width
    sum = 0
    sub = 0
    sd1 = 0
    sd2 = 0
    cov12 = 0.0 #共變異數
    c1 = (255*0.01)**2 #其他參數
    c2 = (255*0.03)**2
    c3 = c2/2
    for y in range(height):
        for x in range(width):
            bgr1 = int(img1[y,x])
            bgr2 = int(img2[y,x])
            sum += bgr1
            sub += bgr2
    mean1 = sum / size_img
    mean2 = sub / size_img
    for y in range(height):
        for x in range(width):
            bgr1 = int(img1[y,x])
            bgr2 = int(img2[y,x])
            diff1 = bgr1 - mean1
            diff2 = bgr2 - mean2
            sd1 += diff1**2
            sd2 += diff2**2
            cov12 += (diff1*diff2)
    sd1 = math.pow(sd1/(size_img-1), 0.5)
    sd2 = math.pow(sd2/(size_img-1), 0.5)
    cov12 = cov12/(size_img-1)
    light = (2*mean1*mean2+c1)/(mean1**2+mean2**2+c1) #比較亮度
    contrast = (2*sd1*sd2+c2)/(sd1**2+sd2**2+c2) #比較對比度
    structure = (cov12+c3)/(sd1*sd2+c3) #比較結構
    ssim = light*contrast*structure #結構相似性
    ssim = round(ssim, 4)
    return ssim

#找出最小公因數
def find_least_common_multiple(array):
    x = array[0]
    for i in range(1,4):
        lcm = 1
        y = array[i]
        mini = x
        if(y < x):
            mini = y
        for c in range(2, mini+1):
            if (x % c == 0) and (y % c == 0):
                lcm = c
        x = lcm
    return x

#生成預測影像
def generate_perdict_image(img, weight):
    #輸入：二維陣列(影像)，4個元素字串
    #輸出：二維陣列(預測影像)
    height, width = img.shape
    temp = img.copy()
    for y in range(1, height-1):
        for x in range(1, width-1):
            ul = int(img[y-1,x-1])
            up = int(img[y-1,x])
            ur = int(img[y-1,x+1])
            left = int(img[y,x-1])
            p = (weight[0]*up+weight[1]*ul+weight[2]*ur+weight[3]*left)/(weight[0]+weight[1]+weight[2]+weight[3])
            temp[y,x] = round(p)
    return temp

def reversible_perdict_image(img, weight):
    #輸入：二維陣列(影像)，4個元素字串
    #輸出：二維陣列(預測影像)
    height, width = img.shape
    temp = np.zeros(height, width)
    for y in range(0, height, height-1):
        for x in range(0, width, width-1):
            temp[y,x] = img[y,x]
    for y in range(1, height-1):
        for x in range(1, width-1):
            ul = int(temp[y-1,x-1])
            up = int(temp[y-1,x])
            ur = int(temp[y-1,x+1])
            left = int(temp[y,x-1])
            p = (weight[0]*up+weight[1]*ul+weight[2]*ur+weight[3]*left)/(weight[0]+weight[1]+weight[2]+weight[3])
            temp[y,x] = round(p)
    return temp

#影像轉回原方向
def image_rerotation(image, times):
    #輸入：二維陣列(影像)，數字(原影像已被90度旋轉次數)
    #輸出：二維陣列(轉回原方向影像)
    if times % 4 == 3:
        image = np.rot90(image, 1)
    elif times % 4 == 2:
        image = np.rot90(image, 2)
    elif times % 4 == 1:
        image = np.rot90(image, 3)
    return image

#找出能產生最佳PSNR預測影像的加權值
def find_best_psnr_from_different_weigth_in_predict_image(img, limit):
    for n1 in range(1, limit+1):
        for n2 in range(1, limit+1):
            for n3 in range(1, limit+1):
                for n4 in range(1, limit+1):
                    #找出最小公因數並跳過來節省運行時間
                    weight = [n1,n2,n3,n4]
                    lcm = find_least_common_multiple(weight)
                    if lcm >= 2:
                        continue
                    #生成預測影像
                    predImg = img.copy()
                    # for t in range(times):
                    temp = generate_perdict_image(predImg, weight)
                    predImg = temp.copy()
                    # img_p = np.rot90(img_p, 1)
                    #回復成原影像方向
                    # img_p = image_rerotation(img_p, times)
                    psnr = calculate_psnr(img, predImg)
                    ssim = calculate_ssim(img, predImg)
                    print(weight, psnr, ssim)
                    #找PSNR最大的加權值
                    if weight == [1,1,1,1]:
                        max_w = weight
                        max_psnr = psnr
                        max_ssim = ssim
                    if max_psnr < psnr:
                        max_w = weight
                        max_psnr = psnr
                        max_ssim = ssim
    return predImg, max_w, max_psnr, max_ssim

#累算二維陣列上的數值並生成直方圖
def generate_different_histogram_without_frame(array2d, histId, histNum):
    #輸入：二維陣列，一維陣列(序號)，一維陣列(累計數)
    #輸出：一維陣列(數列)，一維陣列(累計數)
    height, width = array2d.shape
    for y in range(1, height-1):
        for x in range(1, width-1):
            value = int(array2d[y,x])
            for i in range(len(histNum)):
                if value == histId[i]:
                    histNum[i] += 1
    return histId, histNum

def generate_histogram(array2D):
    height, width = array2D.shape
    values = array2D.flatten()
    min_val = int(np.min(values))
    max_val = int(np.max(values))
    range_size = max_val - min_val + 1
    num = [0] * range_size
    for value in values:
        num[int(value) - min_val] += 1
    return num, min_val, max_val, range_size  # 确保这里返回的是列表，而不是元组

#兩個二維陣列的數值相加或相減
def two_array2D_add_or_subtract(array2D_1, array2D_2, sign):
    #輸出：二維陣列1，二維陣列2
    #輸入：二維陣列
    row, column = array2D_1.shape
    diff = np.zeros((row, column))
    for j in range(row):
        for i in range(column):
            diff[j,i] = int(array2D_1[j,i]) + sign*int(array2D_2[j,i])
    return diff

#計算陣列中正負a之間的累加值
def calculate_payload(array2D, a):
    #輸入：二維陣列、a
    #輸出：整數(累加值)
    row, column = array2D.shape
    payload = 0
    r = np.floor(a/2)
    func = a % 2
    for j in range(1, row-1):
        for i in range(1, column-1):
            if func == 0 or r == 0:
                if -r < array2D[j,i] and array2D[j,i] <= r :
                    payload += 1
            if func == 1:
                if  -r <= array2D[j,i] and array2D[j,i] <= r :
                    payload += 1
    return payload

#比限制值大的加1，小的減1
def image_difference_shift(array2D, a):
    #輸入：二維陣列(差值陣列)，數字(限制值)
    #輸出：二維陣列(偏移後陣列)
    row, column = array2D.shape
    array2D_s = array2D.copy()
    func = a%2
    r = np.floor(a/2)
    for j in range(1, row-1):
        for i in range(1, column-1):
            value = array2D[j,i]
            shift = value
            if func == 1 or a == 1:
                if value > r:
                    shift = value+r
                elif value < -r:
                    shift = value-r-1
            elif func == 0:
                if value > r:
                    shift = value+r
                elif value < (-r+1):
                    shift = value-r
            array2D_s[j,i] = shift
    return array2D_s

#差值直方圖平移嵌入
def image_difference_embeding(array2D, array1D, a, flag):
    row, column = array2D.shape
    array2D_e = array2D.copy()
    inf = []
    r = int(a/2)
    func = a % 2
    h = 0
    data_length = len(array1D)

    for y in range(1, row-1):
        for x in range(1, column-1):
            value = array2D[y,x]
            embed = value  # 默认不改变像素值

            if h < data_length:
                w = array1D[h]
            else:
                break  # 如果所有数据都已嵌入，则退出循环

            if func == 0:
                if r == 0 and value == 0:
                    embed = value - w
                elif abs(value) == r:
                    embed = value + (1 if w == 1 else -1) * (1 if value > 0 else -1)
            elif func == 1:
                if abs(value) == r:
                    embed = value + (1 if w == 1 else -1) * (1 if value > 0 else -1)
                elif r == 0 and value == 0:
                    embed = value - w

            if embed != value:
                array2D_e[y,x] = embed
                inf.append(w)
                h += 1
                if h >= data_length:
                    break  # 如果所有数据都已嵌入，则退出内层循环

        if h >= data_length:
            break  # 如果所有数据都已嵌入，则退出外层循环

    return array2D_e, inf

#正數字串轉換為二進位制，並分開列成list
def int_transfer_binary_single_intlist(array1D, set):
    intA = []
    lenA = len(array1D)
    for i in range(lenA):
        bin = set.format((array1D[i]), "b")
        bin = list(map(int, bin))
        lenB = len(bin)
        for j in range(lenB):
            intA.append(bin[j])
    return intA

def binary_list_to_int(binary_list, bits_per_int):
    """
    将二进制列表转换为整数列表
    
    参数:
    binary_list -- 二进制数字的列表 (0 和 1)
    bits_per_int -- 每个整数使用的位数
    
    返回:
    整数列表
    """
    int_list = []
    for i in range(0, len(binary_list), bits_per_int):
        binary_chunk = binary_list[i:i+bits_per_int]
        # 如果最后一个块不完整,用0填充
        if len(binary_chunk) < bits_per_int:
            binary_chunk = binary_chunk + [0] * (bits_per_int - len(binary_chunk))
        
        # 将二进制块转换为整数
        int_value = 0
        for bit in binary_chunk:
            int_value = (int_value << 1) | bit
        
        int_list.append(int_value)
    
    return int_list

# 使用示例
# 假设我们有一个二进制列表和每个整数使用9位
binary_data = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1]
restored_integers = binary_list_to_int(binary_data, 9)
print(restored_integers)

#固定長度串接重複數字陣列
def repeat_int_array(array, lenNewA):
    newArray = [0]*lenNewA
    lenArray = len(array)
    timesInf = int(lenNewA/lenArray)
    for i in range(timesInf):
        for j in range(lenArray):
            newArray[i*lenArray+j] = array[j]
    return newArray

#二維陣列轉換為一維陣列
def array2D_transfer_to_array1D(array2D):
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

#一維陣列轉換為二維陣列
def array1D_transfer_to_array2D(array1D):
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

#是否為相同陣列
def same_array1D(array1, array2):
    if len(array1) == len(array2):
        for x in range(len(array1)):
            if array1[x] != array2[x]:
                return 0
        return 1
    else:
        return 0

def same_array2D(array1, array2):
    row1, column1 = array1.shape
    row2, column2 = array2.shape
    if row1 == row2 and column1 == column2:
        for y in range(row1):
            for x in range(row2):
                if array1[y,x] != array2[y,x]:
                    return 0
        return 1
    else:
        return 0

#簡化QRcode
def simplified_qrcode(qrcode):
    height, width = qrcode.shape
    same = 0
    simQrcode = -1
    bits = 0
    for y in range(height-1):
        if same_array1D(qrcode[y], qrcode[y+1]):
            same += 1
        else:
            same = 0
        if same == 1:
            bits += 1
            if simQrcode == -1:
                simQrcode = [qrcode[y]]
            else:
                simQrcode.append(qrcode[y])
    simQrcode = np.array(simQrcode)
    outQrcode = np.zeros((bits, bits), np.uint8)
    i = 0
    same = 0
    for x in range(width-1):
        if same_array1D(qrcode[x], qrcode[x+1]):
            same += 1
        else:
            same = 0
        if same == 1:
            for j in range(bits):
                outQrcode[j,i] = simQrcode[j,x]
            i += 1
    return bits, outQrcode

#找出所有方格的左上角座標
def find_locaion(qrcode, bits):
    pixOfBits = int(qrcode.shape[0]/bits)
    locArray = np.zeros((2, bits))
    for y in range(bits):
        locArray[0,y] = y*pixOfBits
    for x in range(bits):
        locArray[1,x] = x*pixOfBits
    return locArray

#計算可嵌入位元數
def calculate_embedded_bits(qrcode, mode):
    bits = qrcode.shape[0]
    payload = 0
    if mode == 1:
        for i in range(bits-1):
            for j in range(bits-1):
                if j == 0 and qrcode[j,i] != qrcode[j,i+1]:
                    payload += 1
                if qrcode[j,i] == qrcode[j,i+1] and qrcode[j+1,i] != qrcode[j+1,i+1]:
                    payload += 1
    elif mode == 2:
        for j in range(bits-1):
            for i in range(bits-1):
                if i == 0 and qrcode[j,i] != qrcode[j+1,i]:
                    payload += 1
                if qrcode[j,i] == qrcode[j+1,i] and qrcode[j,i+1] != qrcode[j+1,i+1]:
                    payload += 1
    return payload

#計算可嵌入位元數
def calculate_embedded_bits(qrcode, mode):
    bits = qrcode.shape[0]
    r = 0
    if mode == 1:
        for i in range(bits-1):
            for j in range(bits-1):
                if j == 0 and qrcode[j,i] != qrcode[j,i+1]:
                    r += 1
                if qrcode[j,i] == qrcode[j,i+1] and qrcode[j+1,i] != qrcode[j+1,i+1]:
                    r += 1
    elif mode == 2:
        for j in range(bits-1):
            for i in range(bits-1):
                if i == 0 and qrcode[j,i] != qrcode[j+1,i]:
                    r += 1
                if qrcode[j,i] == qrcode[j+1,i] and qrcode[j,i+1] != qrcode[j+1,i+1]:
                    r += 1
    return r

#模塊組的類別
def MB_classification(m1, m2, m3, m4):
    if m1 == m2 and m3 == m4:
        sort = 1
    elif m1 != m2 and m3 == m4:
        sort = 2
    elif m1 == m2 and m3 != m4:
        sort = 3
    elif m1 != m2 and m3 != m4:
        sort = 4
    return sort

def split_qr_code(qr_data, num_parts=5):
    """將QR碼數據分割為指定數量的部分"""
    height, width = qr_data.shape
    part_height = height // num_parts
    parts = []
    for i in range(num_parts):
        start = i * part_height
        end = start + part_height if i < num_parts - 1 else height
        parts.append(qr_data[start:end, :])
    return parts

def reconstruct_qr_code(parts, original_shape):
    """從分割的部分重建QR碼"""
    flat_qr = np.concatenate(parts)
    return flat_qr.reshape(original_shape)

def improved_predict_image(img, block_size=8):
    height, width = img.shape
    pred_img = np.zeros_like(img)
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = img[i:i+block_size, j:j+block_size]
            if i == 0 and j == 0:
                pred_img[i:i+block_size, j:j+block_size] = np.mean(block)
            elif i == 0:
                pred_img[i:i+block_size, j:j+block_size] = block[:, 0][:, np.newaxis]
            elif j == 0:
                pred_img[i:i+block_size, j:j+block_size] = block[0, :][np.newaxis, :]
            else:
                left = img[i:i+block_size, j-1][:, np.newaxis]
                top = img[i-1, j:j+block_size][np.newaxis, :]
                pred_img[i:i+block_size, j:j+block_size] = (left + top) / 2

    return pred_img

def adaptive_embedding(diff, data, threshold):
    embedded = np.zeros_like(diff)
    embedded_data = []
    data_index = 0
    
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if data_index >= len(data):
                embedded[i, j] = diff[i, j]
                continue
            
            if abs(diff[i, j]) < threshold:
                bit = data[data_index]
                if diff[i, j] >= 0:
                    embedded[i, j] = 2 * diff[i, j] + bit
                else:
                    embedded[i, j] = 2 * diff[i, j] - bit
                embedded_data.append(bit)
                data_index += 1
            else:
                embedded[i, j] = diff[i, j]
    
    return embedded, embedded_data

def improved_pee_embedding_split(origImg, qr_data, block_size=8, threshold=3):
    current_img = origImg.copy()
    total_payload = 0
    all_embedded_data = []
    qr_parts = split_qr_code(qr_data, 5)  # Assuming 5 rotations as before
    
    for i, qr_part in enumerate(qr_parts):
        print(f"\nProcessing rotation {i}")
        pred_img = improved_predict_image(current_img, block_size)
        diff = current_img - pred_img
        
        qr_bits = qr_part.flatten()
        embedded_diff, embedded_data = adaptive_embedding(diff, qr_bits, threshold)
        
        current_img = pred_img + embedded_diff
        
        total_payload += len(embedded_data)
        all_embedded_data.extend(embedded_data)
        
        print(f"Payload for this rotation: {len(embedded_data)}")
        print(f"Total payload so far: {total_payload}")
        
        if i < 4:  # Don't rotate after the last iteration
            current_img = np.rot90(current_img)
    
    print(f"\nFinal total payload: {total_payload}")
    print(f"Total embedded data length: {len(all_embedded_data)}")
    
    return current_img, all_embedded_data, total_payload

def generate_metadata(qr_data):
    return {
        "content_type": "QR Code",
        "version": "1.0",
        "timestamp": int(time.time()),
        "description": "Secure QR code with enhanced PEE and splitting",
        "qr_size": qr_data.shape[0]  # 假設QR碼是正方形的
    }

def sign_data(data, private_key):
    return private_key.sign(
        data,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

def improved_pee_extraction_split(embedded_image, weight, EL, num_rotations=4):
    current_img = embedded_image.copy()
    qr_parts = []
    
    for i in range(num_rotations, -1, -1):
        img_p = generate_perdict_image(current_img, weight)
        diffA_e = two_array2D_add_or_subtract(current_img, img_p, -1)
        diffA_s, extracted_info = decode_image_difference_embedding(diffA_e, EL)
        diffA = decode_image_different_shift(diffA_s, EL)
        current_img = two_array2D_add_or_subtract(img_p, diffA, 1)
        
        qr_parts.insert(0, extracted_info)
        
        if i > 0:
            current_img = np.rot90(current_img, -1)
    
    # 重建QR碼
    # 注意：我們需要一種方法來確定原始QR碼的大小，這可能需要在嵌入過程中額外存儲
    qr_size = int(np.sqrt(sum(len(part) for part in qr_parts)))
    reconstructed_qr = reconstruct_qr_code(qr_parts, (qr_size, qr_size))
    
    return reconstructed_qr

#影藏數據嵌入
def embedding(image, locArray, j, i, b, k, mode):
    height = image.shape[0]
    width = image.shape[1]
    bits = locArray.shape[1]
    if mode == 1:
        if b == 1:
            p_y1 = int(locArray[0,j])
            p_x1 = int(locArray[1,i])
            if j == bits-1:
                p_y2 = height
            else:
                p_y2 = int(locArray[0,j+1])
            p_x2 = int(locArray[1,i]+k)
            color = image[p_y1,p_x1-1]
            for y in range(p_y1, p_y2):
                for x in range(p_x1, p_x2):
                    image[y,x] = color
    elif mode == 2:
        if b == 1:
            p_y1 = int(locArray[0,j])
            p_x1 = int(locArray[1,i])
            p_y2 = int(locArray[0,j]+k)
            if i == bits-1:
                p_x2 = width
            else:
                p_x2 = int(locArray[1,i+1])
            color = image[p_y1-1,p_x1]
            for y in range(p_y1, p_y2):
                for x in range(p_x1, p_x2):
                    image[y,x] = color

#提出嵌入數值
def find_embedding_bits(image, locArray, j, i, mode):
    height = image.shape[0]
    bits = locArray.shape[1]
    pixOfBit = int(height/bits)
    sub = pixOfBit - 1
    if mode == 1:
        p_y1 = int(locArray[0,j]+sub)
        p_x1 = int(locArray[1,i])
        p_y2 = int(locArray[0,j]+sub)
        p_x2 = int(locArray[1,i]-1)
    elif mode == 2:
        p_y1 = int(locArray[0,j])
        p_x1 = int(locArray[1,i]+sub)
        p_y2 = int(locArray[0,j]-1)
        p_x2 = int(locArray[1,i]+sub)
    if image[p_y1,p_x1] == image[p_y2,p_x2]:
        return 1
    elif image[p_y1,p_x1] != image[p_y2,p_x2]:
        return 0

#調整MP，消除鋸齒狀邊界
def adjustment(image, locArray, j, i, b, k, mode):
    height = image.shape[0]
    width = image.shape[1]
    bits = locArray.shape[1]
    if mode == 1:
        if b == 1:
            p_y1 = int(locArray[0,j])
            p_x1 = int(locArray[1,i])
            if j == bits-1:
                p_y2 = height
            else:
                p_y2 = int(locArray[0,j+1])
            p_x2 = int(locArray[1,i]+k)
            color = image[p_y1,p_x1-1]
            for y in range(p_y1, p_y2):
                for x in range(p_x1, p_x2):
                    image[y,x] = color
    elif mode == 2:
        if b == 1:
            p_y1 = int(locArray[0,j])
            p_x1 = int(locArray[1,i])
            p_y2 = int(locArray[0,j]+k)
            if i == bits-1:
                p_x2 = width
            else:
                p_x2 = int(locArray[1,i+1])
            color = image[p_y1-1,p_x1]
            for y in range(p_y1, p_y2):
                for x in range(p_x1, p_x2):
                    image[y,x] = color

#水平嵌入（垂直掃描）
def horizontal_embedding(qrcode, simQrcode, locArray, insertArray, k):
    i_b = 0
    length = len(insertArray)
    bits = simQrcode.shape[0]
    stegoImg = qrcode.copy()
    for i in range(bits-1):
        for j in range(bits-1):
            m11 = simQrcode[j,i]
            m12 = simQrcode[j,i+1]
            m21 = simQrcode[j+1,i]
            m22 = simQrcode[j+1,i+1]
            sort = MB_classification(m11, m12, m21, m22)
            if j == 0 and m11 != m12:
                if i_b < length:
                    b = insertArray[i_b]
                else:
                    b = 0
                i_b += 1
                embedding(stegoImg, locArray, j, i+1, b, k, 1)
            if sort == 3:
                if i_b < length:
                    b = insertArray[i_b]
                else:
                    b = 0
                i_b += 1
                embedding(stegoImg, locArray, j+1, i+1, b, k, 1)
            elif sort == 4:
                b = find_embedding_bits(stegoImg, locArray, j, i+1, 1)
                adjustment(stegoImg, locArray, j+1, i+1, b, k, 1)
    return stegoImg

#垂直遷入（水平掃描）
def vertical_embedding(qrcode, simQrcode, locArray, insertArray, k):
    i_b = 0
    length = len(insertArray)
    bits = simQrcode.shape[0]
    stegoImg = qrcode.copy()
    for j in range(bits-1):
        for i in range(bits-1):
            m11 = simQrcode[j,i]
            m12 = simQrcode[j,i+1]
            m21 = simQrcode[j+1,i]
            m22 = simQrcode[j+1,i+1]
            sort = MB_classification(m11, m21, m12, m22)
            if i == 0 and m11 != m21:
                if i_b < length:
                    b = insertArray[i_b]
                else:
                    b = 0
                i_b += 1
                embedding(stegoImg, locArray, j+1, i, b, k, 2)
                insertArray.append(b)
            if sort == 3:
                if i_b < length:
                    b = insertArray[i_b]
                else:
                    b = 0
                i_b += 1
                embedding(stegoImg, locArray, j+1, i+1, b, k, 2)
            elif sort == 4:
                b = find_embedding_bits(stegoImg, locArray, j+1, i, 2)
                adjustment(stegoImg, locArray, j+1, i+1, b, k, 2)
    return stegoImg

#兩像素作差值嵌入
def two_value_difference_expansion(left, right, hide):
    l = np.floor((left+right)/2)
    if left >= right:
        h = left-right
        h_e = 2*h+hide
        left_e = l+np.floor((h_e+1)/2)
        right_e = l-np.floor(h_e/2)
    elif left < right:
        h = right-left
        h_e = 2*h+hide
        left_e = l-np.floor(h_e/2)
        right_e = l+np.floor((h_e+1)/2)
    return left_e, right_e

def enhanced_de_embedding_split(img_pee, qr_data, num_rotations, EL, weight):
    current_img = img_pee.copy()
    all_hidemap = []
    all_locationmap = []
    all_payload = []

    # 生成元數據
    metadata = generate_metadata(qr_data)
    metadata_bytes = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
    
    # 生成數字簽名
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    signature = np.frombuffer(sign_data(qr_data.tobytes(), private_key), dtype=np.uint8)

    print(f"Metadata bytes length: {len(metadata_bytes)}")
    print(f"Signature length: {len(signature)}")

    # 合併元數據和簽名
    metadata_and_signature = np.concatenate([metadata_bytes, signature])

    for i in range(num_rotations + 1):
        img_p = generate_perdict_image(current_img, weight)
        diffA = two_array2D_add_or_subtract(current_img, img_p, -1)
        diffA_s = image_difference_shift(diffA, EL)
        
        if i == 0:
            # 在第一次迭代中嵌入元數據和簽名
            combined_data = metadata_and_signature
        else:
            # 其他迭代可以用於嵌入額外的錯誤校正數據或留空
            combined_data = np.array([], dtype=np.uint8)
        
        print(f"Combined data shape: {combined_data.shape}, dtype: {combined_data.dtype}")
        
        diffA_e, inInf = image_difference_embeding(diffA_s, combined_data, EL, 1)
        current_img = two_array2D_add_or_subtract(img_p, diffA_e, 1)
        
        payload = len(inInf)  # 使用實際嵌入的位元數
        
        all_hidemap.append(inInf)
        all_locationmap.append(diffA_s)
        all_payload.append(payload)

        if i < num_rotations:
            current_img = np.rot90(current_img)

    total_payload = sum(all_payload)
    print(f"X2 Total payload: {total_payload}")

    return current_img, all_hidemap, all_locationmap, all_payload, private_key.public_key()

def enhanced_de_extraction_split(embedded_image, public_key, num_rotations, EL, weight):
    current_img = embedded_image.copy()
    metadata = None
    signature = None
    
    for i in range(num_rotations, -1, -1):
        img_p = generate_perdict_image(current_img, weight)
        diffA_e = two_array2D_add_or_subtract(current_img, img_p, -1)
        diffA_s, extracted_data = decode_image_difference_embedding(diffA_e, EL)
        diffA = decode_image_different_shift(diffA_s, EL)
        current_img = two_array2D_add_or_subtract(img_p, diffA, 1)
        
        if i == 0:
            # 處理第一部分（包含元數據和簽名）
            metadata_end = extracted_data.index(b'}') + 1
            metadata = json.loads(extracted_data[:metadata_end].decode())
            signature_length = 256  # 假設簽名長度為256字節
            signature = extracted_data[metadata_end:metadata_end+signature_length]
        
        if i > 0:
            current_img = np.rot90(current_img, -1)
    
    # 驗證簽名
    try:
        public_key.verify(
            signature,
            metadata['qr_size'].to_bytes(4, byteorder='big'),  # 假設我們只驗證QR碼大小
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        signature_valid = True
    except:
        signature_valid = False
    
    return metadata, signature_valid

def two_value_decode_different_expansion(left_e, right_e):
    if left_e >= right_e:
        sign = 1
    elif left_e < right_e:
        sign = -1
    l_e = np.floor((left_e+right_e)/2)
    h_e = sign*np.floor(left_e - right_e)
    if h_e%2 == 0:
        b = 0
    elif h_e%2 == 1:
        b = 1
    h = np.floor(h_e/2)
    if left_e >= right_e:
        left = l_e + np.floor((h+1)/2)
        right = l_e - np.floor(h/2)
    elif left_e < right_e:
        right = l_e + np.floor((h+1)/2)
        left = l_e - np.floor(h/2)
    return b, left, right

#找出一維陣列中最大值
def find_max(array1D):
    if not array1D:
        return None
    if isinstance(array1D[0], (list, np.ndarray)):
        return max(range(len(array1D)), key=lambda i: max(array1D[i]) if len(array1D[i]) > 0 else float('-inf'))
    else:
        return max(range(len(array1D)), key=lambda i: array1D[i])

#找出模塊的大小
def find_w(image):
    height = image.shape[0]
    width = image.shape[1]
    RLArray = [0]*height
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

#簡化隱寫QRcode
def simplified_stego(qrcode, bits, w):
    simQrcode = np.zeros((bits, bits), np.uint8)
    for j in range(bits):
        for i in range(bits):
            simQrcode[j-1,i-1] = qrcode[j*w-1,i*w-1]
    return simQrcode

#找出隱藏資訊
def extract_message(image, simQrcode, locArray, mode):
    bits = simQrcode.shape[0]
    insertArray = []
    b_i = 0
    if mode == 1:
        for i in range(bits-1):
            for j in range(bits-1):
                m11 = simQrcode[j,i]
                m12 = simQrcode[j,i+1]
                m21 = simQrcode[j+1,i]
                m22 = simQrcode[j+1,i+1]
                sort = MB_classification(m11, m12, m21, m22)
                if j == 0 and m11 != m12:
                    b = find_embedding_bits(image, locArray, j, i+1, mode)
                    insertArray.append(b)
                if sort == 3:
                    b = find_embedding_bits(image, locArray, j+1, i+1, mode)
                    insertArray.append(b)
    elif mode == 2:
        for j in range(bits-1):
            for i in range(bits-1):
                m11 = simQrcode[j,i]
                m12 = simQrcode[j,i+1]
                m21 = simQrcode[j+1,i]
                m22 = simQrcode[j+1,i+1]
                sort = MB_classification(m11, m21, m12, m22)
                if i == 0 and m11 != m21:
                    b = find_embedding_bits(image, locArray, j+1, i, mode)
                    insertArray.append(b)
                    b_i += 1
                if sort == 3:
                    b = find_embedding_bits(image, locArray, j+1, i+1, mode)
                    insertArray.append(b)
                    b_i += 1
    return insertArray

#從二進制中提取資訊
def get_infor_from_array1D(array1D, num, digit):
    out = [0]*num
    for i in range(num):
        decimal = 0
        for b in range(digit):
            decimal += array1D[i*digit+b]*2**(7-b)
        out[i] = decimal
    return out

#已嵌入直方圖解碼，取出嵌入值
def decode_image_difference_embedding(array2D, a):
    row, column = array2D.shape
    deArray = array2D.copy()
    inf = []
    r = int(a/2)
    func = a % 2
    
    def process_pixel(embed, i):
        if i == 0:
            if embed in [0, -1]:
                return 0, abs(embed)
        else:
            if embed in [2*i-1, 2*i, -2*i, -2*i-1]:
                w = embed % 2
                value = i if embed > 0 else -i
                if func % 2 == 0 and embed < 0 and r == i:
                    return None, None
                return value, w
        return None, None

    for i in range(r, -1, -1):
        for y in range(1, row-1):
            for x in range(1, column-1):
                value, w = process_pixel(array2D[y,x], i)
                if value is not None:
                    deArray[y,x] = value
                    inf.append(w)
    
    return deArray, inf

#復原已平移直方圖
def decode_image_different_shift(array2D, a):
    row, column = array2D.shape
    deArray = array2D.copy()
    r = np.floor(a/2)
    func = a%2
    for y in range(row):
        for x in range(column):
            shift = array2D[y,x]
            value = shift
            if func == 1:
                if shift > r:
                    value = shift-r
                elif shift < -r:
                    value = shift+r+1
            elif func == 0:
                if shift > r:
                    value = shift-r
                elif shift < -r:
                    value = shift+r
            deArray[y,x] = value
    return deArray

#計算嵌入值與取出值得正確率
def calculate_correct_ratio(true, extra):
    length = len(true)
    sum_t = 0
    for b in range(length):
        if true[b] == extra[b]:
            sum_t += 1
    Rc = round(sum_t/length, 6)
    return Rc

#直方圖平移嵌入
def histogram_data_hiding(img, flag, array1D):
    h_img, w_img = img.shape
    markedImg = img.copy()
    times = 0
    hist, _, _, _ = generate_histogram(img)  # 只使用頻率計數部分
    
    print("Debug: Histogram in histogram_data_hiding")
    print(f"Histogram type: {type(hist)}")
    print(f"Histogram length: {len(hist)}")
    print(f"Histogram sample: {hist[:10]}")
    
    peak = find_max(hist)
    print(f"Peak found in histogram_data_hiding: {peak}")
    
    payload_h = hist[peak]
    i = 0
    length = len(array1D)
    
    # 初始化 zero
    zero = 255  # 假設最大像素值為 255
    # 找出直方圖為零的像素值，以避免右移溢位
    for h in range(len(hist)):
        if hist[h] == 0:
            zero = h
            break
    
    # 長條圖右移且嵌入隱藏值
    for y in range(h_img):
        for x in range(w_img):
            if i < length:
                b = array1D[i]
            else:
                b = 0
            value = img[y,x]
            if flag == 0:
                if value < peak:
                    value -= 1
                elif value == peak:
                    value -= b
            elif flag == 1:
                if value > peak and value < zero:
                    value += 1
                elif value == peak:
                    value += b
            markedImg[y,x] = value
            i += 1  # 增加 i 確保它在循環中遞增
    
    return markedImg, peak, payload_h

def histogram_data_extraction(img, peak, flag):
    h_img, w_img = img.shape
    extractedImg = img.copy()
    extracted_data = []
    
    # 寻找零点（与嵌入时相同的逻辑）
    hist = generate_histogram(img)
    zero = 255  # 假设最大像素值为 255
    for h in range(len(hist)):
        if hist[h] == 0:
            zero = h
            break
    
    # 提取隐藏数据并恢复图像
    for y in range(h_img):
        for x in range(w_img):
            value = img[y,x]
            if flag == 0:
                if value < peak - 1:
                    extractedImg[y,x] = value + 1
                elif value == peak - 1 or value == peak:
                    extracted_data.append(peak - value)
                    extractedImg[y,x] = peak
            elif flag == 1:
                if value > peak + 1 and value < zero:
                    extractedImg[y,x] = value - 1
                elif value == peak or value == peak + 1:
                    extracted_data.append(value - peak)
                    extractedImg[y,x] = peak
    
    return extractedImg, extracted_data


def int_transfer_binary_single_intlist(array1D, width):
    # 如果输入是列表，转换为 numpy 数组
    if isinstance(array1D, list):
        array1D = np.array(array1D)
    
    # 确保输入是一维数组
    array1D_flat = array1D.flatten()
    
    # 创建一个函数来处理单个整数
    def int_to_binary(x):
        return np.binary_repr(int(x), width=width)
    
    # 使用 numpy 的 vectorize 来应用这个函数到整个数组
    vfunc = np.vectorize(int_to_binary)
    binary_strings = vfunc(array1D_flat)
    
    # 将所有二进制字符串连接起来，然后转换为整数列表
    all_bits = ''.join(binary_strings)
    return [int(bit) for bit in all_bits]

#影像讀取
imgName = "airplane"
qrcodeName = "nuk_L"
filetype = "png"
range_x = 10 #預估是嵌入過程直方圖x範圍
k = 2 #QRcode模塊形狀調整寬度
method = "h" #模塊形狀調整方法：h水平、v垂直
EL = 3 #嵌入限制(大於1)
weight = [1,2,11,12] #加權值

origImg = cv2.imread("./pred_and_QR/image/%s.%s"%(imgName, filetype), cv2.IMREAD_GRAYSCALE) #原影像讀取位置
QRCImg = cv2.imread("./pred_and_QR/qrcode/%s.%s"%(qrcodeName, filetype), cv2.IMREAD_GRAYSCALE) #二維碼影像讀取位置

# 編碼過程
print("開始編碼過程...")

num_rotations = 4

# 生成預測圖像
img_p = generate_perdict_image(origImg, weight)

# X1: 改進的預測誤差擴展（PEE）嵌入法
print("\nX1: 改進的預測誤差擴展（PEE）嵌入法")
img_pee, embedded_data, payload_pee = improved_pee_embedding_split(origImg, QRCImg)

psnr_pee = calculate_psnr(origImg, img_pee)
ssim_pee = calculate_ssim(origImg, img_pee)
hist_orig, _, _, _ = generate_histogram(origImg)
hist_pee, _, _, _ = generate_histogram(img_pee)
corr_pee = histogram_correlation(hist_orig, hist_pee)
bpp_pee = round(payload_pee / origImg.size, 4)

print(f"X1 Final: PSNR={psnr_pee:.2f}, SSIM={ssim_pee:.4f}, Correlation={corr_pee:.4f}")
print(f"X1: Total payload={payload_pee}, bpp={bpp_pee:.4f}")

check_quality_after_stage("X1", origImg, img_pee)

# X2: 增強的DE嵌入法
print("\nX2: 增強的DE嵌入法")
img_de, hidemap_parts, locationmap_parts, payload_de_parts, public_key = enhanced_de_embedding_split(
    img_pee, QRCImg, num_rotations, EL, weight
)

total_payload_de = sum(payload_de_parts)
psnr_de = calculate_psnr(origImg, img_de)
ssim_de = calculate_ssim(origImg, img_de)
hist_de, _, _, _ = generate_histogram(img_de)
corr_de = histogram_correlation(hist_orig, hist_de)

print(f"X2: Total DE payload = {total_payload_de}")
print(f"X2: PSNR = {psnr_de:.2f}, SSIM = {ssim_de:.4f}")
print(f"X2: Histogram Correlation = {corr_de:.4f}")

check_quality_after_stage("X2", origImg, img_de)

# X3: 直方圖平移嵌入法
print("\nX3: 直方圖平移嵌入法")
combined_locationmap = np.concatenate(locationmap_parts)
combined_locationmap_int = np.round(combined_locationmap).astype(int)
max_abs_value = np.max(np.abs(combined_locationmap_int))
required_bits = int(np.ceil(np.log2(max_abs_value + 1))) + 1
bin_map = int_transfer_binary_single_intlist(combined_locationmap_int, required_bits)

hist_de, _, _, _ = generate_histogram(img_de)
peak = find_max(hist_de)
img_h, peak, payload_h = histogram_data_hiding(img_de, 1, bin_map)

psnr_h = calculate_psnr(origImg, img_h)
ssim_h = calculate_ssim(origImg, img_h)
hist_h, _, _, _ = generate_histogram(img_h)
corr_h = histogram_correlation(hist_orig, hist_h)

check_quality_after_stage("X3", origImg, img_h)

print(f"X3: Peak = {peak}, Payload = {payload_h}")
print(f"X3: PSNR = {psnr_h:.2f}, SSIM = {ssim_h:.4f}")
print(f"X3: Histogram Correlation = {corr_h:.4f}")

# 計算總的payload和bpp
total_payload = payload_pee + total_payload_de + payload_h
total_bpp = total_payload / origImg.size

# 處理QR碼
bits, simQrcode = simplified_qrcode(QRCImg)
loc = find_locaion(QRCImg, bits)

bin_weight = [1, 2, 11, 12]  # Replace with the desired values for bin_weight

if method == "h":
    payload_qr = calculate_embedded_bits(simQrcode, 1)
    message = repeat_int_array(bin_weight, payload_qr)
    QRCImg_m = horizontal_embedding(QRCImg, simQrcode, loc, message, k)
elif method == "v":
    payload_qr = calculate_embedded_bits(simQrcode, 2)
    message = repeat_int_array(bin_weight, payload_qr)
    QRCImg_m = vertical_embedding(QRCImg, simQrcode, loc, message, k)

# 計算QR碼的SSIM和正確率
ssim_q = calculate_ssim(QRCImg, QRCImg_m)
binQRC = array2D_transfer_to_array1D(QRCImg)
binQRC_m = array2D_transfer_to_array1D(QRCImg_m)
ratio_qr = calculate_correct_ratio(binQRC, binQRC_m)

# 儲存結果圖像
cv2.imwrite(f"./pred_and_QR/outcome/image/{imgName}/{imgName}_pred.{filetype}", img_p)
cv2.imwrite(f"./pred_and_QR/outcome/qrcode/{qrcodeName}_{method}.{filetype}", QRCImg_m)
cv2.imwrite(f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1.{filetype}", img_pee)
cv2.imwrite(f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X2.{filetype}", img_de)
cv2.imwrite(f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X3.{filetype}", img_h)

# 輸出最終結果
print("\n編碼過程總結:")
print(f"原影像: {imgName}, 原二維碼: {qrcodeName}")
print(f"加權值={weight}, EL={EL}")
print(f"QR碼: SSIM={ssim_q:.4f}, 正確率={ratio_qr:.4f}")
print(f"X1 (PEE): payload={payload_pee}, bpp={bpp_pee:.4f}")
print(f"X1 (PEE): PSNR={psnr_pee:.2f}, SSIM={ssim_pee:.4f}, 直方圖相關性={corr_pee:.4f}")
print(f"X2 (增強DE): 最大payload={total_payload_de}, location map={len(combined_locationmap)} bits")
print(f"X2 (增強DE): PSNR={psnr_de:.2f}, SSIM={ssim_de:.4f}, 直方圖相關性={corr_de:.4f}")
print(f"X3 (直方圖平移): peak={peak}, payload={payload_h}")
print(f"X3 (直方圖平移): PSNR={psnr_h:.2f}, SSIM={ssim_h:.4f}, 直方圖相關性={corr_h:.4f}")
print(f"總payload={total_payload}, 總bpp={total_bpp:.4f}")
print("...編碼過程結束...")
print()

# 解碼過程
print("開始解碼過程...")

# 初始化
markedImg = img_h.copy()  # 使用X3階段的輸出作為起點

# X3: 直方圖平移解碼
print("X3: 直方圖平移解碼")
img_de_extracted = histogram_data_extraction(markedImg, peak, bin_map)  # 需要實現這個函數

# X2: 增強DE提取
print("X2: 增強DE提取")
reconstructed_qr, extracted_metadata, is_signature_valid = enhanced_de_extraction_split(
    img_de_extracted, public_key, num_rotations, EL
)

print(f"提取的元數據: {extracted_metadata}")
print(f"簽名驗證: {'成功' if is_signature_valid else '失敗'}")

# X1: 改進的PEE提取
print("X1: 改進的PEE提取")
extracted_qr_parts = []
current_img = reconstructed_qr.copy()

for i in range(num_rotations, -1, -1):
    img_p = generate_perdict_image(current_img, weight)
    diffA_e = two_array2D_add_or_subtract(current_img, img_p, -1)
    diffA_s, extracted_data = decode_image_difference_embedding(diffA_e, EL)
    diffA = decode_image_different_shift(diffA_s, EL)
    current_img = two_array2D_add_or_subtract(img_p, diffA, 1)
    
    extracted_qr_parts.insert(0, extracted_data)
    
    if i > 0:
        current_img = np.rot90(current_img, -1)

# 重建完整的QR碼
final_reconstructed_qr = reconstruct_qr_code(extracted_qr_parts, QRCImg.shape)

# 提取和验证QR码
print("提取和验证QR码")
extracted_qr, extracted_metadata, is_signature_valid = enhanced_de_extraction_split(
    img_de, private_key.public_key(), num_rotations, EL
)

print(f"提取的元数据: {extracted_metadata}")
print(f"签名验证: {'成功' if is_signature_valid else '失败'}")
print(f"QR码成功重建: {np.array_equal(extracted_qr, QRCImg)}")

# 提取權重信息
w = find_w(final_reconstructed_qr)
bits = int(final_reconstructed_qr.shape[0]/w)
location = find_locaion(final_reconstructed_qr, bits)
simStego = simplified_stego(final_reconstructed_qr, bits, w)

if method == "h":
    exBin = extract_message(final_reconstructed_qr, simStego, location, 1)
elif method == "v":
    exBin = extract_message(final_reconstructed_qr, simStego, location, 2)

exWeight = get_infor_from_array1D(exBin, 4, 8)
print(f"提取的權重值: {exWeight}")

# 生成並保存差值直方圖
for i, (diffA, stage) in enumerate([
    (diffA, "diff"),
    (diffA_s, "diffshift"),
    (diffA_e, "diffembed")
]):
    diffId, diffNum = generate_different_histogram_without_frame(diffA, list(range(-range_x,range_x+1)), [0]*(range_x*2+1))
    plt.figure()
    plt.bar(diffId, diffNum)
    plt.ylim(0, max(diffNum) * 1.3)
    plt.savefig(f"./pred_and_QR/outcome/histogram/{imgName}/difference/de_{imgName}_{stage}.{filetype}")
    plt.close()

# 最終驗證
if qr_reconstruction_success and is_signature_valid:
    print("解密成功！QR碼重建正確，簽名驗證通過。")
elif qr_reconstruction_success and not is_signature_valid:
    print("警告：QR碼重建正確，但簽名驗證失敗。")
elif not qr_reconstruction_success and is_signature_valid:
    print("警告：QR碼重建失敗，但簽名驗證通過。")
else:
    print("解密失敗：QR碼重建失敗，簽名驗證也失敗。")

print("...解碼過程結束...")

# 影像展示
plt.subplot(2,2,1)
plt.imshow(origImg, cmap="gray")
plt.title(f"{imgName}")
plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(img_pee, cmap="gray")
plt.title("Improved PEE")
plt.axis('off')
plt.subplot(2,2,3)
plt.imshow(img_de, cmap="gray")
plt.title("Difference Expansion")
plt.axis('off')
plt.subplot(2,2,4)
plt.imshow(img_h, cmap="gray")
plt.title("Histogram Shift")
plt.axis('off')
plt.show()
plt.close()