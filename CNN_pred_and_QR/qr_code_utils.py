import numpy as np

def simplified_qrcode(qrcode):
    """簡化QRcode"""
    height, width = qrcode.shape
    same = 0
    simQrcode = None
    bits = 0
    for y in range(height-1):
        if np.array_equal(qrcode[y], qrcode[y+1]):
            same += 1
        else:
            same = 0
        if same == 1:
            bits += 1
            if simQrcode is None:
                simQrcode = [qrcode[y]]
            else:
                simQrcode.append(qrcode[y])
    
    if bits == 0:
        return 0, np.array([])  # 返回空數組而不是 -1
    
    simQrcode = np.array(simQrcode)
    outQrcode = np.zeros((bits, bits), np.uint8)
    i = 0
    same = 0
    for x in range(width-1):
        if np.array_equal(qrcode[x], qrcode[x+1]):
            same += 1
        else:
            same = 0
        if same == 1:
            for j in range(bits):
                outQrcode[j,i] = simQrcode[j,x]
            i += 1
    return bits, outQrcode

def find_locaion(qrcode, bits):
    """找出所有方格的左上角座標"""
    if bits == 0:
        return np.array([])  # 如果 bits 為 0，返回空數組
    pixOfBits = int(qrcode.shape[0]/bits)
    locArray = np.zeros((2, bits))
    for y in range(bits):
        locArray[0,y] = y*pixOfBits
    for x in range(bits):
        locArray[1,x] = x*pixOfBits
    return locArray


def calculate_embedded_bits(qrcode, mode):
    """計算可嵌入位元數"""
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

def MB_classification(m1, m2, m3, m4):
    """模塊組的類別"""
    if m1 == m2 and m3 == m4:
        sort = 1
    elif m1 != m2 and m3 == m4:
        sort = 2
    elif m1 == m2 and m3 != m4:
        sort = 3
    elif m1 != m2 and m3 != m4:
        sort = 4
    return sort

def embedding(image, locArray, j, i, b, k, mode):
    """影藏數據嵌入"""
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
            image[p_y1:p_y2, p_x1:p_x2] = color
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
            image[p_y1:p_y2, p_x1:p_x2] = color

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
    return 1 if abs(int(image[p_y1,p_x1]) - int(image[p_y2,p_x2])) > 0 else 0

def adjustment(image, locArray, j, i, b, k, mode):
    """調整MP，消除鋸齒狀邊界"""
    embedding(image, locArray, j, i, b, k, mode)

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
                b = insertArray[i_b] if i_b < length else 0
                i_b += 1
                embedding(stegoImg, locArray, j, i+1, b, k, 1)
            if sort == 3:
                b = insertArray[i_b] if i_b < length else 0
                i_b += 1
                embedding(stegoImg, locArray, j+1, i+1, b, k, 1)
            elif sort == 4:
                b = find_embedding_bits(stegoImg, locArray, j, i+1, 1)
                adjustment(stegoImg, locArray, j+1, i+1, b, k, 1)
    print(f"水平嵌入的信息: {insertArray}")
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
                b = insertArray[i_b] if i_b < length else 0
                i_b += 1
                embedding(stegoImg, locArray, j+1, i, b, k, 2)
            if sort == 3:
                b = insertArray[i_b] if i_b < length else 0
                i_b += 1
                embedding(stegoImg, locArray, j+1, i+1, b, k, 2)
            elif sort == 4:
                b = find_embedding_bits(stegoImg, locArray, j+1, i, 2)
                adjustment(stegoImg, locArray, j+1, i+1, b, k, 2)
    print(f"垂直嵌入的信息: {insertArray}")
    return stegoImg

def simplified_stego(qrcode, bits, w):
    """簡化隱寫QRcode"""
    simQrcode = np.zeros((bits, bits), np.uint8)
    for j in range(bits):
        for i in range(bits):
            simQrcode[j,i] = qrcode[j*w,i*w]
    return simQrcode

def extract_message(image, simQrcode, locArray, mode):
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
                if len(insertArray) >= 40:  # 只提取前40位
                    break
            if len(insertArray) >= 40:
                break
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
                if len(insertArray) >= 40:  # 只提取前40位
                    break
            if len(insertArray) >= 40:
                break
    print(f"從QR碼中提取的原始信息: {insertArray}")
    return insertArray[:40]  # 确保只返回40位

# 如果需要直接運行此文件進行測試
if __name__ == "__main__":
    # 創建一個模擬 QR 碼的測試數組
    def create_test_qr(size=21, module_size=4):
        qr = np.zeros((size * module_size, size * module_size), dtype=np.uint8)
        # 添加定位圖案
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                qr[i*module_size:(i+1)*module_size, j*module_size:(j+1)*module_size] = 255
                qr[i*module_size:(i+1)*module_size, (size-3+j)*module_size:(size-2+j)*module_size] = 255
                qr[(size-3+i)*module_size:(size-2+i)*module_size, j*module_size:(j+1)*module_size] = 255
        
        # 添加一些隨機數據
        np.random.seed(42)  # 使用固定的隨機種子以獲得可重複的結果
        random_data = np.random.randint(0, 2, (size, size), dtype=np.uint8) * 255
        for i in range(size):
            for j in range(size):
                if qr[i*module_size, j*module_size] == 0:  # 只在未被定位圖案佔用的地方添加隨機數據
                    qr[i*module_size:(i+1)*module_size, j*module_size:(j+1)*module_size] = random_data[i, j]
        
        return qr

    # 創建測試 QR 碼
    test_qr = create_test_qr()
    
    # 測試 simplified_qrcode 函數
    bits, simplified = simplified_qrcode(test_qr)
    print(f"Simplified QR code size: {bits}x{bits}")
    
    if bits > 0:
        # 測試 find_locaion 函數
        loc_array = find_locaion(test_qr, bits)
        print(f"Location array shape: {loc_array.shape}")
        
        # 測試 calculate_embedded_bits 函數
        embed_bits_h = calculate_embedded_bits(simplified, 1)
        embed_bits_v = calculate_embedded_bits(simplified, 2)
        print(f"Embeddable bits (horizontal): {embed_bits_h}")
        print(f"Embeddable bits (vertical): {embed_bits_v}")
        
        # 測試嵌入和提取
        test_data = np.random.randint(0, 2, embed_bits_h)
        embedded_qr = horizontal_embedding(test_qr, simplified, loc_array, test_data, k=2)
        extracted_data = extract_message(embedded_qr, simplified, loc_array, mode=1)
        
        print(f"Embedded data length: {len(test_data)}")
        print(f"Extracted data length: {len(extracted_data)}")
        print(f"Data correctly extracted: {np.array_equal(test_data, extracted_data)}")
    else:
        print("QR code simplification failed or resulted in an empty array.")