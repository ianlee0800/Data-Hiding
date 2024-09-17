import numpy as np
import json
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from image_processing import (
    generate_perdict_image,
    image_difference_shift,
    two_array2D_add_or_subtract,
    generate_histogram,
    find_w,
    simplified_qrcode,
    find_locaion,
    improved_predict_image
)
from utils import (
    find_max,
    verify_signature,
    MB_classification
)

def pee_extraction_adaptive(embedded_img, payload, weight, block_size=8, threshold=3):
    """改进的自适应预测误差扩展(PEE)提取"""
    height, width = embedded_img.shape
    pred_img = improved_predict_image(embedded_img, weight, block_size)
    diff = embedded_img - pred_img
    
    extracted_data = []
    
    for i in range(height):
        for j in range(width):
            if len(extracted_data) >= payload:
                break
            
            local_var = np.var(embedded_img[max(0, i-1):min(height, i+2), max(0, j-1):min(width, j+2)])
            adaptive_threshold = threshold * (1 + local_var / 1000)
            
            if abs(diff[i, j]) < adaptive_threshold:
                if diff[i, j] % 2 == 0:
                    extracted_data.append(0)
                else:
                    extracted_data.append(1)
    
    return extracted_data

def adaptive_extraction(diff, threshold):
    """自适应提取"""
    extracted_data = []
    
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if abs(diff[i, j]) < threshold:
                if diff[i, j] % 2 == 0:
                    extracted_data.append(0)
                else:
                    extracted_data.append(1)
    
    return extracted_data

def de_extraction_split(embedded_image, public_key, num_rotations, EL, weight):
    """增强的差值扩展(DE)提取"""
    current_img = embedded_image.copy()
    all_extracted_data = []
    metadata = None
    signature = None
    
    for i in range(num_rotations, -1, -1):
        img_p = generate_perdict_image(current_img, weight)
        diffA_e = two_array2D_add_or_subtract(current_img, img_p, -1)
        diffA_s, extracted_data = decode_image_difference_embedding(diffA_e, EL)
        diffA = decode_image_different_shift(diffA_s, EL)
        current_img = two_array2D_add_or_subtract(img_p, diffA, 1)
        
        print(f"Rotation {i}: Extracted data length: {len(extracted_data)}")
        
        if i == 0:
            # 处理第一部分（包含元数据和签名）
            print(f"Extracted data (first 100 bytes): {extracted_data[:100]}")
            try:
                metadata_end = extracted_data.index(ord('}')) + 1
                metadata_str = bytes(extracted_data[:metadata_end]).decode('utf-8')
                metadata = json.loads(metadata_str)
                print(f"Extracted metadata: {metadata}")
                
                signature_length = 256  # 假设签名长度为256字节
                signature = bytes(extracted_data[metadata_end:metadata_end+signature_length])
                print(f"Extracted signature length: {len(signature)}")
            except ValueError as e:
                print(f"Error parsing metadata: {e}")
                print(f"Full extracted data: {extracted_data}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Extracted metadata string: {metadata_str}")
        else:
            all_extracted_data.insert(0, extracted_data)
        
        if i > 0:
            current_img = np.rot90(current_img, -1)
    
    # 验证签名
    is_signature_valid = False
    if metadata and signature:
        try:
            is_signature_valid = verify_signature(public_key, signature, metadata['qr_size'].to_bytes(4, byteorder='big'))
        except Exception as e:
            print(f"Error verifying signature: {e}")
    
    # 重建QR码
    qr_size = metadata['qr_size'] if metadata else 0
    reconstructed_qr = None
    if qr_size > 0 and all_extracted_data:
        try:
            reconstructed_qr = np.concatenate(all_extracted_data).reshape((qr_size, qr_size))
            print(f"Successfully reconstructed QR code with shape: {reconstructed_qr.shape}")
        except ValueError as e:
            print(f"Error reconstructing QR code: {e}")
            print(f"Total extracted data length: {sum(len(data) for data in all_extracted_data)}")
    else:
        print("Unable to reconstruct QR code: insufficient data or invalid QR size")
    
    return current_img, reconstructed_qr, metadata, is_signature_valid

def two_value_decode_different_expansion(left_e, right_e):
    """两值差值扩展解码"""
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

def decode_image_difference_embedding(array2D, a):
    """已嵌入直方图解码，取出嵌入值"""
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

def decode_image_different_shift(array2D, a):
    """复原已平移直方图"""
    row, column = array2D.shape
    deArray = array2D.copy()
    r = int(a/2)
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

def histogram_data_extraction(img, peak, flag):
    """直方图平移数据提取"""
    h_img, w_img = img.shape
    extractedImg = img.copy()
    extracted_data = []
    
    # 寻找零点（与嵌入时相同的逻辑）
    hist, _, _, _ = generate_histogram(img)
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

def extract_message(image, simQrcode, locArray, mode):
    """从QR码中提取消息"""
    bits = simQrcode.shape[0]
    insertArray = []
    
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
                if sort == 3:
                    b = find_embedding_bits(image, locArray, j+1, i+1, mode)
                    insertArray.append(b)
    
    return insertArray

def find_embedding_bits(image, locArray, j, i, mode):
    """提取嵌入的比特"""
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

# 可能需要的其他提取相关函数...