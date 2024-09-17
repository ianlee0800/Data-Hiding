import numpy as np
import json
from cryptography.hazmat.primitives.asymmetric import rsa
from image_processing import (
    generate_perdict_image, 
    image_difference_shift, 
    two_array2D_add_or_subtract,
    generate_histogram,
    find_w,
    simplified_qrcode,
    find_locaion,
    improved_predict_image,
    calculate_psnr,
    calculate_ssim
)
from utils import (
    find_max,
    generate_metadata,
    sign_data,
    int_transfer_binary_single_intlist,
    MB_classification,
    find_least_common_multiple
)

def pee_embedding_adaptive(img, data, weight, block_size=8, threshold=3):
    """改进的自适应预测误差扩展(PEE)嵌入"""
    height, width = img.shape
    pred_img = improved_predict_image(img, weight, block_size)
    diff = img - pred_img
    
    embedded = np.zeros_like(diff)
    embedded_data = []
    data_index = 0
    
    for i in range(height):
        for j in range(width):
            if data_index >= len(data):
                embedded[i, j] = diff[i, j]
                continue
            
            # 自适应阈值：使用局部方差来调整阈值
            local_var = np.var(img[max(0, i-1):min(height, i+2), max(0, j-1):min(width, j+2)])
            adaptive_threshold = threshold * (1 + local_var / 1000)  # 可以根据需要调整这个公式
            
            if abs(diff[i, j]) < adaptive_threshold:
                bit = data[data_index]
                if diff[i, j] >= 0:
                    embedded[i, j] = 2 * diff[i, j] + bit
                else:
                    embedded[i, j] = 2 * diff[i, j] - bit
                embedded_data.append(bit)
                data_index += 1
            else:
                embedded[i, j] = diff[i, j]
    
    embedded_img = pred_img + embedded
    return embedded_img, embedded_data, data_index

def adaptive_embedding(diff, data, threshold):
    """自适应嵌入"""
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

def de_embedding_split(img_pee, qr_data, num_rotations, EL, weight):
    """增强的差值扩展(DE)嵌入"""
    current_img = img_pee.copy()
    all_hidemap = []
    all_locationmap = []
    all_payload = []

    # 生成元数据
    metadata = generate_metadata(qr_data)
    metadata_bytes = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
    
    # 生成数字签名
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    signature = np.frombuffer(sign_data(qr_data.tobytes(), private_key), dtype=np.uint8)

    print(f"Metadata bytes length: {len(metadata_bytes)}")
    print(f"Signature length: {len(signature)}")

    # 合并元数据和签名
    metadata_and_signature = np.concatenate([metadata_bytes, signature])

    for i in range(num_rotations + 1):
        img_p = generate_perdict_image(current_img, weight)
        diffA = two_array2D_add_or_subtract(current_img, img_p, -1)
        diffA_s = image_difference_shift(diffA, EL)
        
        if i == 0:
            # 在第一次迭代中嵌入元数据和签名
            combined_data = metadata_and_signature
        else:
            # 其他迭代可以用于嵌入额外的错误校正数据或留空
            combined_data = np.array([], dtype=np.uint8)
        
        print(f"Combined data shape: {combined_data.shape}, dtype: {combined_data.dtype}")
        
        diffA_e, inInf = image_difference_embeding(diffA_s, combined_data, EL, 1)
        current_img = two_array2D_add_or_subtract(img_p, diffA_e, 1)
        
        payload = len(inInf)  # 使用实际嵌入的位元数
        
        all_hidemap.append(inInf)
        all_locationmap.append(diffA_s)
        all_payload.append(payload)

        if i < num_rotations:
            current_img = np.rot90(current_img)

    total_payload = sum(all_payload)
    print(f"DE Total payload: {total_payload}")

    return current_img, all_hidemap, all_locationmap, all_payload, private_key.public_key()

def image_difference_embeding(array2D, array1D, a, flag):
    """差值直方图平移嵌入"""
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

def histogram_data_hiding(img, flag, embedding_info):
    """直方图平移嵌入"""
    h_img, w_img = img.shape
    markedImg = img.copy()
    hist, _, _, _ = generate_histogram(img)
    
    print("Debug: Histogram in histogram_data_hiding")
    print(f"Histogram type: {type(hist)}")
    print(f"Histogram length: {len(hist)}")
    print(f"Histogram sample: {hist[:10]}")
    
    peak = find_max(hist)
    print(f"Peak found in histogram_data_hiding: {peak}")
    
    payload_h = hist[peak]
    i = 0
    
    # 将embedding_info转换为二进制列表
    array1D = []
    for key, value in embedding_info.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, list):
                    array1D.extend([bin(x)[2:].zfill(8) for x in item])
                else:
                    array1D.append(bin(item)[2:].zfill(32))
        else:
            array1D.append(bin(value)[2:].zfill(32))
    array1D = ''.join(array1D)
    length = len(array1D)
    
    # 初始化 zero
    zero = 255  # 假设最大像素值为 255
    # 找出直方图为零的像素值，以避免右移溢位
    for h in range(len(hist)):
        if hist[h] == 0:
            zero = h
            break
    
    # 长条图右移且嵌入隐藏值
    for y in range(h_img):
        for x in range(w_img):
            if i < length:
                b = int(array1D[i])
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
            i += 1
    
    return markedImg, peak, i

def horizontal_embedding(qrcode, simQrcode, locArray, insertArray, k):
    """水平嵌入（垂直掃描）"""
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

def vertical_embedding(qrcode, simQrcode, locArray, insertArray, k):
    """垂直嵌入（水平掃描）"""
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

def embedding(image, locArray, j, i, b, k, mode):
    """影藏数据嵌入"""
    height, width = image.shape
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

def find_embedding_bits(image, locArray, j, i, mode):
    """提出嵌入数值"""
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

def adjustment(image, locArray, j, i, b, k, mode):
    """调整MP，消除锯齿状边界"""
    height, width = image.shape
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

def two_value_difference_expansion(left, right, hide):
    """两像素作差值嵌入"""
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

def calculate_embedded_bits(qrcode, mode):
    """计算可嵌入位元数"""
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

def find_best_psnr_from_different_weigth_in_predict_image(img, limit):
    """找出能产生最佳PSNR预测影像的加权值"""
    max_psnr = float('-inf')
    max_ssim = float('-inf')
    max_w = None
    
    for n1 in range(1, limit+1):
        for n2 in range(1, limit+1):
            for n3 in range(1, limit+1):
                for n4 in range(1, limit+1):
                    weight = [n1,n2,n3,n4]
                    lcm = find_least_common_multiple(weight)
                    if lcm >= 2:
                        continue
                    predImg = generate_perdict_image(img, weight)
                    psnr = calculate_psnr(img, predImg)
                    ssim = calculate_ssim(img, predImg)
                    print(weight, psnr, ssim)
                    if psnr > max_psnr:
                        max_w = weight
                        max_psnr = psnr
                        max_ssim = ssim
    
    return generate_perdict_image(img, max_w), max_w, max_psnr, max_ssim

