import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def calculate_correlation(img1, img2):
    # 將圖像轉換為一維數組
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    # 計算相關係數
    correlation = np.corrcoef(img1_flat, img2_flat)[0, 1]
    
    return round(correlation, 4)

def histogram_correlation(hist1, hist2):
    # 確保兩個直方圖長度相同
    assert len(hist1) == len(hist2), "Histograms must have the same length"
    
    # 計算相關係數
    correlation = np.corrcoef(hist1, hist2)[0, 1]
    
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
    num = [0]*256
    for y in range(height):
        for x in range(width):
            value = int(array2D[y,x])
            num[value] += 1
    return num

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
    func = r%2
    h = 0
    for i in range(r, -1, -1):
        temp = array2D_e.copy()
        for y in range(1, row-1):
            for x in range(1, column-1):
                value = temp[y,x]
                embed = "none"
                if flag == 0:
                    w = np.random.choice([0,1], p=[0.5,0.5])
                elif flag == 1:
                    if h < len(array1D):
                        w = array1D[h]
                    else:
                        w = 0
                if func == 0:
                    if i == 0 and value == 0:
                        embed = value-w
                    elif value == i:
                        embed = value+i-1+w
                    elif value == -i:
                        embed = value-i-w
                elif func == 1:
                    if i >= 1:
                        if value == i:
                            embed = value+i-1+w
                        elif value == -i:
                            embed = value-i-w
                    elif i == 0:
                        if value == i:
                            embed = value-w
                if embed == "none":
                    embed = value
                else:
                    h += 1
                    inf.append(w)
                array2D_e[y,x] = embed
    if flag == 0:
        return array2D_e, inf
    elif flag == 1:
        return array2D_e

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

def improved_pee_embedding(origImg, weight, EL, range_x, num_rotations=4):
    current_img = origImg.copy()
    total_payload = 0
    all_inInf = []
    
    # 初始化用於存儲最後一次旋轉的直方圖數據
    final_diffId, final_diffNum = None, None
    final_diffId_s, final_diffNum_s = None, None
    final_diffId_e, final_diffNum_e = None, None
    
    for i in range(num_rotations + 1):
        img_p = generate_perdict_image(current_img, weight)
        
        # a. 差值直方圖處理
        diffA = two_array2D_add_or_subtract(current_img, img_p, -1)
        payload_diff = calculate_payload(diffA, EL)
        diffId, diffNum = generate_different_histogram_without_frame(diffA, list(range(-range_x,range_x+1)), [0]*(range_x*2+1))
        
        # b. 直方圖左右偏移，移出嵌入空間
        diffA_s = image_difference_shift(diffA, EL)
        diffId_s, diffNum_s = generate_different_histogram_without_frame(diffA_s, list(range(-range_x,range_x+1)), [0]*(range_x*2+1))
        
        # c. 差值直方圖平移，嵌入隨機值
        diffA_e, inInf = image_difference_embeding(diffA_s, 0, EL, 0)
        diffId_e, diffNum_e = generate_different_histogram_without_frame(diffA_e, list(range(-range_x,range_x+1)), [0]*(range_x*2+1))
        
        # 差值直方圖與預測影像還原成已嵌入影像
        current_img = two_array2D_add_or_subtract(img_p, diffA_e, 1)
        
        total_payload += payload_diff
        all_inInf.extend(inInf)
        
        # 保存最後一次旋轉的直方圖數據
        final_diffId, final_diffNum = diffId, diffNum
        final_diffId_s, final_diffNum_s = diffId_s, diffNum_s
        final_diffId_e, final_diffNum_e = diffId_e, diffNum_e
        
        psnr = calculate_psnr(origImg, current_img)
        ssim = calculate_ssim(origImg, current_img)
        corr = calculate_correlation(origImg, current_img)
        print(f"Rotation {i}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, Correlation={corr:.4f}, Payload={payload_diff}")
        
        if i < num_rotations:
            current_img = np.rot90(current_img)
    
    final_img_p = generate_perdict_image(current_img, weight)
    return current_img, all_inInf, total_payload, (final_diffId, final_diffNum), (final_diffId_s, final_diffNum_s), (final_diffId_e, final_diffNum_e), final_img_p

def improved_pee_extraction(embedded_image, weight, EL, num_rotations=4):
    current_img = embedded_image.copy()
    all_extracted_info = []
    
    for i in range(num_rotations, -1, -1):
        img_p = generate_perdict_image(current_img, weight)
        diffA_e = two_array2D_add_or_subtract(current_img, img_p, -1)
        diffA_s, extracted_info = decode_image_difference_embeding(diffA_e, EL)
        diffA = decode_image_different_shift(diffA_s, EL)
        current_img = two_array2D_add_or_subtract(img_p, diffA, 1)
        
        all_extracted_info = extracted_info + all_extracted_info
        
        if i > 0:
            current_img = np.rot90(current_img, -1)
    
    return current_img, all_extracted_info

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

#將一維陣列用差值嵌入法進影像中
def different_expansion_embeding_array1D(img, array1D):
    height, width = img.shape
    markedImg = img.copy()
    hidemap = np.zeros((height, width))
    length = len(array1D)
    locationmap = []
    payload_DE = 0
    i = 0
    sum = 0
    #差值嵌入法，嵌入二維碼
    for y in range(height):
        for x in range(0, width-1, 2):
            if i < length:
                b = array1D[i]
                flag = 1
            elif i >= length:
                b = 0
                flag = 0
            left = int(img[y,x]) #左像素值
            right = int(img[y,x+1]) #右像素值
            left_e, right_e = two_value_difference_expansion(left, right, b) #變更後的左右像素值
            #挑選可隱藏像素避免溢位
            if left_e >= 0 and left_e <= 255 and right_e >= 0 and right_e <= 255:
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
                    sum+=1
    return markedImg, hidemap, locationmap, payload_DE

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

def decode_different_expansion(img):
    height, width = img.shape
    decodeImg = img.copy()
    exInf = []
    i = 0
    for y in range(height):
        x = 0
        while x <= width-2:
            if hidemap[y,x] != 255:#未嵌入位置跳過
                x += 1
                continue
            left_e = int(img_de[y,x]) #左像素值
            right_e = int(img_de[y,x+1]) #右像素值
            b, left, right = two_value_decode_different_expansion(left_e, right_e)
            decodeImg[y,x] = left
            decodeImg[y,x+1] = right
            exInf.append(b)
            i += 1
            x += 2
    return decodeImg, exInf

#找出一維陣列中最大值
def find_max(array1D):
    max = 0 
    for i in range(1, len(array1D)):
        if array1D[i] > array1D[max]:
            max = i
    return max

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
def decode_image_difference_embeding(array2D, a):
    row, column = array2D.shape
    deArray = array2D.copy()
    inf = []
    r = int(a/2)
    func = a%2
    for i in range(r, -1, -1):
        if i == 0:
            bit01 = [0, -1]
        else:
            bit01 = [2*i-1, 2*i, -2*i, -2*i-1]
        for y in range(1, row-1):
            for x in range(1, column-1):
                embed = array2D[y,x]
                value = "no"
                for b in range(len(bit01)):
                    if embed == bit01[b]:
                        if b%2  == 0:
                            w = 0
                        elif b%2 == 1:
                            w = 1
                        if i == 0:
                            if embed == -1:
                                value = 0
                            elif embed == 0:
                                value = 0
                        elif i > 0:
                            if func%2 == 1:
                                if embed > 0:
                                    value = i
                                elif embed < 0:
                                    value = -i
                            elif func%2 == 0:
                                if embed > 0:
                                    value = i
                                elif embed < 0 and r != i:
                                    value = -i
                        if value != "no":
                            inf.append(w)
                            deArray[y,x] = value
    # for y in range(1, row-1):
    #     for x in range(1, column-1):
    #         embed = array2D[y,x]
    #         value = embed
    #         if func == 1 and (2*(-r)-1 <= embed and embed <= 2*r):
    #             if embed > 0:
    #                 if embed % 2 == 0:
    #                     w = 1
    #                     value = int(embed/2)
    #                 elif embed % 2 == 1:
    #                     w = 0
    #                     value = int((embed+1)/2)
    #             elif embed <= 0:
    #                 if embed % 2 == 0:
    #                     w = 0
    #                     value = int(embed/2)
    #                 elif embed % 2 == 1:
    #                     w = 1
    #                     value = int((embed+1)/2)
    #             inf.append(w)
    #         elif func == 0 and (2*(-r) < embed and embed <= 2*r):
    #             if embed > 0:
    #                 if embed % 2 == 0:
    #                     w = 1
    #                     value = int(embed/2)
    #                 elif embed % 2 == 1:
    #                     w = 0
    #                     value = int((embed+1)/2)
    #             elif embed <= 0:
    #                 if embed % 2 == 0:
    #                     w = 0
    #                     value = int(embed/2)
    #                 elif embed % 2 == 1:
    #                     w = 1
    #                     value = int((embed+1)/2)
    #             inf.append(w)
    #         deArray[y,x] = value
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
    hist = generate_histogram(img)
    peak = find_max(hist)
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

#影像讀取
imgName = "airplane"
qrcodeName = "nuk_L"
filetype = "png"
range_x = 10 #預估是嵌入過程直方圖x範圍
k = 2 #QRcode模塊形狀調整寬度
method = "h" #模塊形狀調整方法：h水平、v垂直
EL = 7 #嵌入限制(大於1)
weight = [1,2,11,12] #加權值

origImg = cv2.imread("./pred_and_QR/image/%s.%s"%(imgName, filetype), cv2.IMREAD_GRAYSCALE) #原影像讀取位置
QRCImg = cv2.imread("./pred_and_QR/qrcode/%s.%s"%(qrcodeName, filetype), cv2.IMREAD_GRAYSCALE) #二維碼影像讀取位置

#encode

# 計算原始圖像的直方圖
hist_orig = generate_histogram(origImg)

# X1: 改進的預測誤差擴展（PEE）嵌入法
print("X1: 改進的預測誤差擴展（PEE）嵌入法")
img_pee, inInf, payload_pee, (diffId, diffNum), (diffId_s, diffNum_s), (diffId_e, diffNum_e), img_p = improved_pee_embedding(origImg, weight, EL, range_x, num_rotations=4)
hist_pee = generate_histogram(img_pee)
psnr_pee = calculate_psnr(origImg, img_pee)
ssim_pee = calculate_ssim(origImg, img_pee)
corr_pee = histogram_correlation(hist_orig, hist_pee)
bpp_pee = round(payload_pee/origImg.size, 2)

print(f"X1 Final: PSNR={psnr_pee:.2f}, SSIM={ssim_pee:.4f}, Correlation={corr_pee:.4f}")
print(f"X1: Total payload={payload_pee}, bpp={bpp_pee}")

# 二維碼處理部分保持不變
data_p = weight.copy()
data_p.append(EL)
bin_weight = int_transfer_binary_single_intlist(data_p, "{0:08b}")
bits, simQrcode = simplified_qrcode(QRCImg)
loc = find_locaion(QRCImg, bits)

if method == "h":
    payload_qr = calculate_embedded_bits(simQrcode, 1)
    message = repeat_int_array(bin_weight, payload_qr)
    QRCImg_m = horizontal_embedding(QRCImg, simQrcode, loc, message, k)
elif method == "v":
    payload_qr = calculate_embedded_bits(simQrcode, 2)
    message = repeat_int_array(bin_weight, payload_qr)
    QRCImg_m = vertical_embedding(QRCImg, simQrcode, loc, message, k)

# X2：DE嵌入法，嵌入二進制二維碼
binQRC_m = array2D_transfer_to_array1D(QRCImg_m)
img_de, hidemap, locationmap, payload_de = different_expansion_embeding_array1D(img_pee, binQRC_m)
hist_de = generate_histogram(img_de)
psnr_de = calculate_psnr(origImg, img_de)
ssim_de = calculate_ssim(origImg, img_de)
corr_de = histogram_correlation(hist_orig, hist_de)
binQRC = array2D_transfer_to_array1D(QRCImg)  # 原QRcode轉換為二進制
ratio_qr = calculate_correct_ratio(binQRC, binQRC_m)
ssim_q = calculate_ssim(QRCImg, QRCImg_m)

#X3：直方圖平移嵌入法，嵌入可嵌入地圖
bin_map = int_transfer_binary_single_intlist(locationmap, "{0:09b}")
img_h, peak, payload_h = histogram_data_hiding(img_de, 1, bin_map)
hist_h = generate_histogram(img_h)
psnr_h = calculate_psnr(origImg, img_h)
ssim_h = calculate_ssim(origImg, img_h)
corr_h = histogram_correlation(hist_orig, hist_h)

# 相關影像儲存
cv2.imwrite(f"./pred_and_QR/outcome/image/{imgName}/{imgName}_pred.{filetype}", img_p)  # 預測影像
cv2.imwrite(f"./pred_and_QR/outcome/qrcode/{qrcodeName}_{method}.{filetype}", QRCImg_m)  # 嵌入後二維碼
cv2.imwrite(f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1.{filetype}", img_pee)  # X1 (改進的PEE)
cv2.imwrite(f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X2.{filetype}", img_de)  # X2 (DE)
cv2.imwrite(f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X3.{filetype}", img_h)  # X3 (直方圖平移)

# 差值直方圖影像的輸出
plt.bar(diffId, diffNum)
plt.ylim(0, max(diffNum) * 1.3)
plt.savefig(f"./pred_and_QR/outcome/histogram/{imgName}/difference/{imgName}_diff.{filetype}")
plt.close()

plt.bar(diffId_s, diffNum_s)
plt.ylim(0, max(diffNum_s) * 1.3)
plt.savefig(f"./pred_and_QR/outcome/histogram/{imgName}/difference/{imgName}_diffshift.{filetype}")
plt.close()

plt.bar(diffId_e, diffNum_e)
plt.ylim(0, max(diffNum_e) * 1.3)
plt.savefig(f"./pred_and_QR/outcome/histogram/{imgName}/difference/{imgName}_diffembed.{filetype}")
plt.close()

#原影像直方圖
hist_orig = generate_histogram(origImg)
plt.bar(range(256), hist_orig)
plt.savefig("./pred_and_QR/outcome/histogram/%s/%s_orighist.%s"%(imgName, imgName, filetype))
plt.close()  

# X1直方圖
plt.bar(range(256), hist_pee)
plt.savefig(f"./pred_and_QR/outcome/histogram/{imgName}/{imgName}_X1_hist.{filetype}")
plt.close()

#X2直方圖  
plt.bar(range(256), hist_de)
plt.savefig("./pred_and_QR/outcome/histogram/%s/%s_X2_hist.%s"%(imgName, imgName, filetype)) 
plt.close()

#X3直方圖
plt.bar(range(256), hist_h)  
plt.savefig("./pred_and_QR/outcome/histogram/%s/%s_X3_hist.%s"%(imgName, imgName, filetype))
plt.close()

# 打印結果
print(f"原影像: {imgName}, 原二維碼: {qrcodeName}")
print(f"加權值={weight}, EL={EL}")
print(f"qrcode: ssim={ssim_q:.4f}, correct ratio={ratio_qr:.4f}")
print(f"X1: payload={payload_pee}, bpp={bpp_pee:.4f}")
print(f"X1: PSNR={psnr_pee:.2f}, SSIM={ssim_pee:.4f}, Histogram Correlation={corr_pee:.4f}")
print(f"X2: maximum payload={payload_de}, location map={len(locationmap)} bits")
print(f"X2: PSNR={psnr_de:.2f}, SSIM={ssim_de:.4f}, Histogram Correlation={corr_de:.4f}")
print(f"X3: peak={peak}, payload={payload_h}")
print(f"X3: PSNR={psnr_h:.2f}, SSIM={ssim_h:.4f}, Histogram Correlation={corr_h:.4f}")
print("...加密完成...")
print()

#decode
markedImg = img_de.copy()

# 使用改進的PEE提取方法
exImg_pee, exInf = improved_pee_extraction(markedImg, weight, EL, num_rotations=4)

# DE解碼，取出已嵌入二維碼二進制
exImg_de, exInf_qr = decode_different_expansion(markedImg)
flag_img = same_array2D(exImg_de, img_de) and calculate_psnr(exImg_de, img_de) == 0 and calculate_ssim(exImg_de, img_de)
flag_inf = same_array1D(binQRC_m, exInf_qr)

#二維碼二進制轉換為二維碼圖像
exQRc = array1D_transfer_to_array2D(exInf_qr)

#還原QRcode與提取資訊轉換為加權值
w = find_w(exQRc)
bits = int(exQRc.shape[0]/w)
location = find_locaion(exQRc, bits)
simStego = simplified_stego(exQRc, bits, w)

#水平隱寫影像資訊提取
if method == "h":
    exBin = extract_message(exQRc, simStego, location, 1)

#垂直隱寫影像資訊提取
elif method == "v":
    exBin = extract_message(exQRc, simStego, location, 2)

#從二維碼提取的資訊中得到加權值
exWeight = get_infor_from_array1D(exBin, 4, 8)

# exWeight = [1,2,3,4]

# 生成預測影像，並與DE解碼的影像相減，得到差值直方圖
exImg_p = generate_perdict_image(exImg_pee, weight)
exdiffA_e = two_array2D_add_or_subtract(exImg_de, exImg_p, -1)
exdiffId_e, exdiffNum_e = generate_different_histogram_without_frame(exdiffA_e, list(range(-range_x,range_x+1)), [0]*(range_x*2+1))

plt.bar(exdiffId_e, exdiffNum_e)
plt.ylim(0, max(exdiffNum_e) * 1.3)
plt.savefig(f"./pred_and_QR/outcome/histogram/{imgName}/difference/de_{imgName}_diffembed.{filetype}")
plt.close()

# 解碼已嵌入差值直方圖得到隱藏資訊
exdiffA_s, exInf = decode_image_difference_embeding(exdiffA_e, EL)
exdiffId_s, exdiffNum_s = generate_different_histogram_without_frame(exdiffA_s, list(range(-range_x,range_x+1)), [0]*(range_x*2+1))

plt.bar(exdiffId_s, exdiffNum_s)
plt.ylim(0, max(exdiffNum_s) * 1.3)
plt.savefig(f"./pred_and_QR/outcome/histogram/{imgName}/difference/de_{imgName}_diffshift.{filetype}")
plt.close()

# 將已平移過的直方圖復原
exdiffA = decode_image_different_shift(exdiffA_s, EL)
exdiffId, exdiffNum = generate_different_histogram_without_frame(exdiffA, list(range(-range_x,range_x+1)), [0]*(range_x*2+1))

plt.bar(exdiffId, exdiffNum)
plt.ylim(0, max(exdiffNum) * 1.3)
plt.savefig(f"./pred_and_QR/outcome/histogram/{imgName}/difference/de_{imgName}_diff.{filetype}")
plt.close()

print("取出加權值：%s"%(exWeight))
if flag_img == 1 and flag_inf == 1:
    print("DE影像還原正確，取出二維碼資訊正確")
elif flag_img == 1 and flag_inf == 0:
    print("DE影像還原正確，取出二維碼資料錯誤")
elif flag_img == 0 and flag_inf == 1:
    print("DE影像還原錯誤，取出二維碼資訊正確")
elif flag_img == 0 and flag_inf == 0:
    print("DE影像還原錯誤，取出二維碼資料錯誤")
if same_array1D(inInf, exInf) == 1:
    print("差值直方圖解碼，隱藏資訊提取正確")
if flag_img and flag_inf and same_array1D(inInf, exInf):
    print("...解密完成...")
else:
    print("...解密失敗...")

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