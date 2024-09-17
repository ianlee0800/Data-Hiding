import numpy as np
import json
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import time
from common import calculate_psnr
from image_processing import improved_predict_image

def generate_random_binary_array(size, ratio_of_ones=0.5):
    """生成指定大小的随机二进制数组，可调整1的比例"""
    return np.random.choice([0, 1], size=size, p=[1-ratio_of_ones, ratio_of_ones])

def find_best_weights(img, block_size=8):
    """找出能产生最佳PSNR预测影像的加权值"""
    best_psnr = float('-inf')
    best_weights = None
    
    for w1 in range(1, 5):
        for w2 in range(1, 5):
            for w3 in range(1, 5):
                for w4 in range(1, 5):
                    weights = [w1, w2, w3, w4]
                    pred_img = improved_predict_image(img, weights, block_size)
                    psnr = calculate_psnr(img, pred_img)
                    if psnr > best_psnr:
                        best_psnr = psnr
                        best_weights = weights
    
    return best_weights

def find_least_common_multiple(array):
    """找出最小公因數"""
    x = array[0]
    for i in range(1, 4):
        lcm = 1
        y = array[i]
        mini = min(x, y)
        for c in range(2, mini + 1):
            if (x % c == 0) and (y % c == 0):
                lcm = c
        x = lcm
    return x

def find_max(array1D):
    """找出一維陣列中最大值"""
    if not array1D:
        return None
    if isinstance(array1D[0], (list, np.ndarray)):
        return max(range(len(array1D)), key=lambda i: max(array1D[i]) if len(array1D[i]) > 0 else float('-inf'))
    else:
        return max(range(len(array1D)), key=lambda i: array1D[i])

def int_transfer_binary_single_intlist(array1D, width):
    """将整数或整数数组转换为固定宽度的二进制列表"""
    # 确保输入是一个 NumPy 数组
    array1D = np.asarray(array1D).flatten()
    
    # 创建一个二进制表示的数组
    binary_array = ((np.abs(array1D)[:, np.newaxis] & (1 << np.arange(width)[::-1])) > 0).astype(int)
    
    # 处理负数
    negative_mask = array1D < 0
    binary_array[negative_mask] = 1 - binary_array[negative_mask]
    
    # 将结果展平为一维列表
    return binary_array.flatten().tolist()

def binary_list_to_int(binary_list, bits_per_int):
    """將二進制列表轉換為整數列表"""
    int_list = []
    for i in range(0, len(binary_list), bits_per_int):
        binary_chunk = binary_list[i:i+bits_per_int]
        if len(binary_chunk) < bits_per_int:
            binary_chunk = binary_chunk + [0] * (bits_per_int - len(binary_chunk))
        int_value = 0
        for bit in binary_chunk:
            int_value = (int_value << 1) | bit
        int_list.append(int_value)
    return int_list

def repeat_int_array(array, lenNewA):
    """固定長度串接重複數字陣列"""
    newArray = [0]*lenNewA
    lenArray = len(array)
    timesInf = int(lenNewA/lenArray)
    for i in range(timesInf):
        for j in range(lenArray):
            newArray[i*lenArray+j] = array[j]
    return newArray

def same_array1D(array1, array2):
    """檢查兩個一維陣列是否相同"""
    if len(array1) == len(array2):
        return all(a == b for a, b in zip(array1, array2))
    return False

def same_array2D(array1, array2):
    """檢查兩個二維陣列是否相同"""
    return np.array_equal(array1, array2)

def generate_metadata(qr_data):
    """生成元數據"""
    return {
        "content_type": "QR Code",
        "version": "1.0",
        "timestamp": int(time.time()),
        "description": "Secure QR code with enhanced PEE and splitting",
        "qr_size": qr_data.shape[0]
    }

def sign_data(data, private_key):
    """對數據進行簽名"""
    return private_key.sign(
        data,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

def verify_signature(public_key, signature, data):
    """驗證簽名"""
    try:
        public_key.verify(
            signature,
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except:
        return False

def get_infor_from_array1D(array1D, num, digit):
    """從二進制中提取資訊"""
    out = [0] * num
    for i in range(num):
        decimal = 0
        for b in range(digit):
            decimal += array1D[i*digit+b] * 2**(7-b)
        out[i] = decimal
    return out

def calculate_correct_ratio(true, extra):
    """計算嵌入值與取出值的正確率"""
    length = len(true)
    sum_t = sum(1 for t, e in zip(true, extra) if t == e)
    return round(sum_t / length, 6)

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

def calculate_payload(array2D, a):
    """計算陣列中正負a之間的累加值"""
    row, column = array2D.shape
    payload = 0
    r = np.floor(a/2)
    func = a % 2
    for j in range(1, row-1):
        for i in range(1, column-1):
            value = array2D[j,i]
            if func == 0 or r == 0:
                if -r < value <= r:
                    payload += 1
            elif func == 1:
                if -r <= value <= r:
                    payload += 1
    return payload

def MB_classification(m1, m2, m3, m4):
    """模塊組的類別"""
    if m1 == m2 and m3 == m4:
        return 1
    elif m1 != m2 and m3 == m4:
        return 2
    elif m1 == m2 and m3 != m4:
        return 3
    elif m1 != m2 and m3 != m4:
        return 4

def find_locaion(qrcode, bits):
    """找出所有方格的左上角座標"""
    pixOfBits = qrcode.shape[0] // bits
    return np.array([np.arange(bits) * pixOfBits, np.arange(bits) * pixOfBits])
