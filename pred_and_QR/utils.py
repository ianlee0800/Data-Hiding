import numpy as np
import struct
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
import time
import cupy as cp
import random
import math
from numba import cuda
from image_processing import improved_predict_image_cuda
from common import calculate_psnr
from deap import base, creator, tools, algorithms

@cuda.jit
def pee_kernel(img, pred_img, data, embedded, payload, base_EL):
    x, y = cuda.grid(2)
    if 1 <= x < 511 and 1 <= y < 511:
        # 計算局部標準差
        local_sum = 0.0
        local_sum_sq = 0.0
        count = 0
        for i in range(max(0, x-1), min(512, x+2)):
            for j in range(max(0, y-1), min(512, y+2)):
                pixel = img[i, j]
                local_sum += pixel
                local_sum_sq += pixel * pixel
                count += 1
        local_mean = local_sum / count
        local_var = (local_sum_sq / count) - (local_mean * local_mean)
        local_std = math.sqrt(max(0, local_var))

        # 自適應 EL
        adaptive_EL = max(3, min(base_EL, int(base_EL * (1 - local_std / 255))))

        diff = int(img[x, y]) - int(pred_img[x, y])
        if abs(diff) < adaptive_EL and payload[0] < len(data):
            bit = data[payload[0]]
            cuda.atomic.add(payload, 0, 1)
            if diff >= 0:
                embedded[x, y] = img[x, y] + bit
            else:
                embedded[x, y] = img[x, y] - bit
        else:
            embedded[x, y] = img[x, y]

from image_processing import improved_predict_image_cuda

def pee_embedding_adaptive_cuda(img, data, weight, base_EL):
    if not isinstance(img, cp.ndarray):
        img = cp.asarray(img)
    if not isinstance(data, cp.ndarray):
        data = cp.asarray(data)

    # 使用 image_processing.py 中的函數進行預測
    pred_img = improved_predict_image_cuda(img, weight)

    embedded_img = cp.zeros_like(img)
    d_payload = cp.zeros(1, dtype=int)

    threads_per_block = (16, 16)
    blocks_per_grid = (32, 32)  # 針對 512x512 圖像優化

    pee_kernel[blocks_per_grid, threads_per_block](img, pred_img, data, embedded_img, d_payload, base_EL)

    payload = int(d_payload[0])
    embedded_data = data[:payload]

    return embedded_img, payload, embedded_data
    
def choose_el(img, rotation, current_payload):
    target_payload = 480000
    remaining_rotations = 5 - rotation
    if current_payload < target_payload:
        if rotation == 0:
            return 7  # 第一次旋轉使用較大的 EL
        elif rotation == 1:
            return 5
        else:
            return 3
    else:
        return 3  # 如果已達到目標，使用較小的 EL 以維持圖像質量

def generate_random_binary_array(size, ratio_of_ones=0.5):
    """生成指定大小的随机二进制数组，可调整1的比例"""
    return np.random.choice([0, 1], size=size, p=[1-ratio_of_ones, ratio_of_ones])

def evaluate_weights(weights, img, data, EL):
    pred_img = improved_predict_image_cuda(img, weights)
    embedded_img, payload, _ = pee_embedding_adaptive_cuda(img, data, weights, EL)
    psnr = calculate_psnr(cp.asnumpy(img), cp.asnumpy(embedded_img))
    return payload, psnr

# 全局變量來追蹤是否已經創建了類
_created_classes = False

def ensure_deap_classes():
    global _created_classes
    if not _created_classes:
        if 'FitnessMulti' not in creator.__dict__:
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
        if 'Individual' not in creator.__dict__:
            creator.create("Individual", list, fitness=creator.FitnessMulti)
        _created_classes = True

def find_best_weights_ga(img, data, EL, population_size=50, generations=20, max_weight=15):
    ensure_deap_classes()
    toolbox = base.Toolbox()

    toolbox.register("attr_int", random.randint, 1, max_weight)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_wrapper(individual):
        return evaluate_weights(individual, img, data, EL)

    toolbox.register("evaluate", eval_wrapper)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=max_weight, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    population = toolbox.population(n=population_size)
    algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, 
                              lambda_=population_size, cxpb=0.7, mutpb=0.2, 
                              ngen=generations, verbose=False)

    best_ind = tools.selBest(population, 1)[0]
    return best_ind, best_ind.fitness.values

def encode_pee_info(total_rotations, weights, payloads, ELs):
    print(f"Debug: Encoding PEE info")
    print(f"Total rotations: {total_rotations}")
    print(f"Weights: {weights}")
    print(f"Payloads: {payloads}")
    print(f"ELs: {ELs}")
    
    if not isinstance(total_rotations, int):
        raise TypeError(f"Expected total_rotations to be an integer, got {type(total_rotations)}")
    if not isinstance(weights, list) or len(weights) != total_rotations:
        raise ValueError(f"Expected weights to be a list of {total_rotations} elements, got {weights}")
    if not isinstance(payloads, list) or len(payloads) != total_rotations:
        raise ValueError(f"Expected payloads to be a list of {total_rotations} elements, got {payloads}")
    if not isinstance(ELs, list) or len(ELs) != total_rotations:
        raise ValueError(f"Expected ELs to be a list of {total_rotations} elements, got {ELs}")
    
    encoded = struct.pack('B', total_rotations)
    for rotation in range(total_rotations):
        if not isinstance(weights[rotation], (list, tuple)) or len(weights[rotation]) != 4:
            raise ValueError(f"Expected weights[{rotation}] to be a list or tuple of 4 integers, got {weights[rotation]}")
        if not isinstance(ELs[rotation], int):
            raise ValueError(f"Expected ELs[{rotation}] to be an integer, got {ELs[rotation]}")
        if not isinstance(payloads[rotation], int):
            raise ValueError(f"Expected payloads[{rotation}] to be an integer, got {payloads[rotation]}")
        
        try:
            encoded += struct.pack('BBBBB', *weights[rotation], ELs[rotation])
            encoded += struct.pack('I', payloads[rotation])  # 使用 'I' 代替 'H' 來存儲更大的整數
        except struct.error as e:
            print(f"Error packing data for rotation {rotation}: {e}")
            print(f"Weights: {weights[rotation]}, EL: {ELs[rotation]}, Payload: {payloads[rotation]}")
            raise
    
    return encoded

def decode_pee_info(encoded_data):
    if not isinstance(encoded_data, (bytes, bytearray)):
        raise TypeError(f"Expected bytes or bytearray, got {type(encoded_data)}")
    
    if len(encoded_data) < 1:
        raise ValueError("Encoded data is too short")

    total_rotations = struct.unpack('B', encoded_data[:1])[0]
    weights = []
    payloads = []
    ELs = []
    offset = 1
    try:
        for _ in range(total_rotations):
            if offset + 9 > len(encoded_data):  # 改為 9，因為我們現在用 4 bytes 存儲 payload
                raise ValueError("Encoded data is incomplete")
            weight_and_el = struct.unpack('BBBBB', encoded_data[offset:offset+5])
            weights.append(list(weight_and_el[:4]))
            ELs.append(weight_and_el[4])
            offset += 5
            payloads.append(struct.unpack('I', encoded_data[offset:offset+4])[0])  # 使用 'I' 來解包
            offset += 4
    except struct.error as e:
        raise ValueError(f"Error decoding data: {e}")

    return total_rotations, weights, payloads, ELs

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
