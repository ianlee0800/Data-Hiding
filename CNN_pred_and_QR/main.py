import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from image_metrics import *
from image_processing import *
from data_embedding import *
from qr_code_utils import *
from cnn_model import *

def int_transfer_binary_single_intlist(array1D, bits=8):
    return np.unpackbits(np.array(array1D, dtype=np.uint8).reshape(-1, 1), axis=1)[:, -bits:].flatten()

def get_infor_from_array1D(array1D, num, digit):
    padding = digit - (len(array1D) % digit) if len(array1D) % digit != 0 else 0
    padded_array = np.pad(array1D, (0, padding), 'constant', constant_values=(0))
    reshaped = padded_array.reshape(-1, digit)
    return np.packbits(reshaped, axis=1, bitorder='big').flatten()[:num]

def calculate_correct_ratio(true, extra):
    return sum(t == e for t, e in zip(true, extra)) / len(true)

def save_histogram(data, filename, title, x_label, y_label):
    plt.figure()
    plt.bar(range(len(data)), data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.close()

def save_difference_histogram(data, filename, title, range_x):
    plt.figure()
    plt.bar(range(-range_x, range_x+1), data)
    plt.title(title)
    plt.xlabel("Difference")
    plt.ylabel("Frequency")
    plt.savefig(filename)
    plt.close()

def main():
    print("程序執行開始...")

    # 設置參數
    imgName = "airplane"  
    qrcodeName = "nuk_L"
    filetype = "png"
    range_x = 10  # 預估是嵌入過程直方圖x範圍
    k = 2  # QRcode模塊形狀調整寬度
    method = "h"  # 模塊形狀調整方法：h水平、v垂直
    EL = 7  # 嵌入限制(大於1)
    rotation_times = 1  # 旋轉次數，例如旋轉90度

    # 設置新的文件路徑
    image_path = os.path.join("./pred_and_QR/image", f"{imgName}.{filetype}")
    qrcode_path = os.path.join("./pred_and_QR/qrcode", f"{qrcodeName}.{filetype}")
    
    # 檢查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"無法找到圖像文件: {image_path}")
    if not os.path.exists(qrcode_path):
        raise FileNotFoundError(f"無法找到QR碼文件: {qrcode_path}")

    # 讀取圖像
    origImg = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    QRCImg = cv2.imread(qrcode_path, cv2.IMREAD_GRAYSCALE)

    if origImg is None:
        raise ValueError(f"無法讀取圖像文件: {image_path}")
    if QRCImg is None:
        raise ValueError(f"無法讀取QR碼文件: {qrcode_path}")

    print(f"成功讀取圖像文件: {image_path}")
    print(f"成功讀取QR碼文件: {qrcode_path}")

    print("加載CNN模型...")
    model_path = "./CNN_PEE/model/adaptive_cnn_predictor.pth"  
    model = load_model(model_path)
    
    print(f"旋轉圖像 {rotation_times * 90} 度...")
    rotated_img = image_rotation(origImg, rotation_times)

    print("生成預測圖像...")
    img_p = generate_predict_image(rotated_img, model)
    print(f"預測圖像數據類型: {img_p.dtype}")

    # 保存原始圖像直方圖
    save_histogram(generate_histogram(origImg), f"./pred_and_QR/outcome/histogram/{imgName}/{imgName}_orig_hist.{filetype}", 
                   "Original Image Histogram", "Pixel Value", "Frequency")

    # X1: 預估式嵌入法，嵌入隨機位元
    diffA = two_array2D_add_or_subtract(rotated_img, img_p, -1)
    payload_diff = np.sum((diffA >= -EL) & (diffA <= EL)) - np.sum((diffA == -EL) | (diffA == EL))
    bpp_diff = round(payload_diff / rotated_img.size, 4)

    # 1. 原始差值直方圖
    diffId, diffNum = generate_different_histogram_without_frame(diffA, list(range(-range_x, range_x+1)), [0]*(range_x*2+1))
    save_difference_histogram(diffNum, f"./pred_and_QR/outcome/histogram/{imgName}/difference/{imgName}_diff.{filetype}", 
                              "Original Difference Histogram", range_x)

    diffA_s = image_difference_shift(diffA, EL)

    # 2. 差值平移後的直方圖
    diffId_s, diffNum_s = generate_different_histogram_without_frame(diffA_s, list(range(-range_x, range_x+1)), [0]*(range_x*2+1))
    save_difference_histogram(diffNum_s, f"./pred_and_QR/outcome/histogram/{imgName}/difference/{imgName}_diffshift.{filetype}", 
                              "Difference Shift Histogram", range_x)

    diffA_e, inInf = image_difference_embeding(diffA_s, 0, EL, 0)

    # 3. 差值嵌入後的直方圖
    diffId_e, diffNum_e = generate_different_histogram_without_frame(diffA_e, list(range(-range_x, range_x+1)), [0]*(range_x*2+1))
    save_difference_histogram(diffNum_e, f"./pred_and_QR/outcome/histogram/{imgName}/difference/{imgName}_diffembed.{filetype}", 
                              "Difference Embedding Histogram", range_x)

    img_diff = two_array2D_add_or_subtract(img_p, diffA_e, 1).astype(np.uint8)
    print(f"差值嵌入後圖像數據類型: {img_diff.dtype}")
    print(f"差值嵌入信息長度: {len(inInf)}")

    hist_diff = generate_histogram(img_diff)

    # 保存X1結果直方圖
    save_histogram(hist_diff, f"./pred_and_QR/outcome/histogram/{imgName}/{imgName}_X1_hist.{filetype}", 
                   "X1: Difference Histogram", "Pixel Value", "Frequency")

    # 二維碼：模塊形狀調整，只嵌入EL
    data_p = [EL]
    bin_el = int_transfer_binary_single_intlist(data_p)
    print(f"原始 EL 值: {EL}")
    print(f"EL 的二進制表示: {bin_el}")

    bits, simQrcode = simplified_qrcode(QRCImg)
    loc = find_locaion(QRCImg, bits)

    if method == "h":
        payload_qr = calculate_embedded_bits(simQrcode, 1)
    elif method == "v":
        payload_qr = calculate_embedded_bits(simQrcode, 2)

    message = bin_el  # 只嵌入 EL 值
    print(f"QR碼嵌入信息長度: {len(message)}")

    if method == "h":
        QRCImg_m = horizontal_embedding(QRCImg, simQrcode, loc, message, k)
    elif method == "v":
        QRCImg_m = vertical_embedding(QRCImg, simQrcode, loc, message, k)

    # X2：DE嵌入法，嵌入二進制二維碼
    binQRC_m = array2D_transfer_to_array1D(QRCImg_m)
    img_de, hidemap, locationmap, payload_de = different_expansion_embeding_array1D(img_diff, binQRC_m)
    img_de = img_de.astype(np.uint8)
    print(f"DE嵌入後圖像數據類型: {img_de.dtype}")

    hist_de = generate_histogram(img_de)

    # 保存X2結果直方圖
    save_histogram(hist_de, f"./pred_and_QR/outcome/histogram/{imgName}/{imgName}_X2_hist.{filetype}", 
                   "X2: DE Histogram", "Pixel Value", "Frequency")

    binQRC = array2D_transfer_to_array1D(QRCImg)
    ratio_qr = calculate_correct_ratio(binQRC, binQRC_m)
    ssim_q = calculate_ssim(QRCImg, QRCImg_m)

    # X3：直方圖平移嵌入法，嵌入可嵌入地圖
    bin_map = int_transfer_binary_single_intlist(locationmap, 9)
    img_h, peak, payload_h = histogram_data_hiding(img_de, 1, bin_map)
    print(f"直方圖平移嵌入後圖像數據類型: {img_h.dtype}")

    hist_h = generate_histogram(img_h)

    # 保存X3結果直方圖
    save_histogram(hist_h, f"./pred_and_QR/outcome/histogram/{imgName}/{imgName}_X3_hist.{filetype}", 
                   "X3: Histogram Shift", "Pixel Value", "Frequency")

    # 所有嵌入步驟完成後,將圖像轉回原方向
    img_h = image_rerotation(img_h, rotation_times)

    # 保存中間結果
    cv2.imwrite(f"./pred_and_QR/outcome/image/{imgName}/{imgName}_pred.{filetype}", img_p)
    cv2.imwrite(f"./pred_and_QR/outcome/qrcode/{qrcodeName}_{method}.{filetype}", QRCImg_m)
    cv2.imwrite(f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X1.{filetype}", image_rerotation(img_diff, rotation_times))
    cv2.imwrite(f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X2.{filetype}", image_rerotation(img_de, rotation_times))   
    cv2.imwrite(f"./pred_and_QR/outcome/image/{imgName}/{imgName}_X3.{filetype}", img_h)

    # 計算PSNR和SSIM時使用原始未旋轉的圖像
    psnr_diff = calculate_psnr(origImg, image_rerotation(img_diff, rotation_times))
    ssim_diff = calculate_ssim(origImg, image_rerotation(img_diff, rotation_times))
    psnr_de = calculate_psnr(origImg, image_rerotation(img_de, rotation_times))
    ssim_de = calculate_ssim(origImg, image_rerotation(img_de, rotation_times))
    psnr_h = calculate_psnr(origImg, img_h)
    ssim_h = calculate_ssim(origImg, img_h)

    # 輸出結果
    print(f"原影像: {imgName}, 原二維碼: {qrcodeName}")
    print(f"EL={EL}")
    print(f"qrcode: ssim={ssim_q}, correct ratio={ratio_qr}") 
    print(f"X1: payload={payload_diff}, bpp={bpp_diff}")
    print(f"X1: PSNR={psnr_diff}, SSIM={ssim_diff}")
    print(f"X2: maximum payload={payload_de}, location map={len(bin_map)} bits")  
    print(f"X2: PSNR={psnr_de}, SSIM={ssim_de}")
    print(f"X3: peak={peak}, payload={payload_h}")
    print(f"X3: PSNR={psnr_h}, SSIM={ssim_h}")
    print("...加密完成...")

# 解密過程  
    print("\n開始解密過程...")
    markedImg = img_h.copy()

    # 將接收到的圖像旋轉到處理方向
    rotated_markedImg = image_rotation(markedImg, rotation_times)

    exImg_diff, exInf_qr = decode_different_expansion(rotated_markedImg, hidemap)  
    flag_img = np.array_equal(exImg_diff, img_diff)
    flag_inf = np.array_equal(binQRC_m, exInf_qr)
    exQRc = array1D_transfer_to_array2D(exInf_qr)

    w = find_w(exQRc)
    bits = exQRc.shape[0] // w
    location = find_locaion(exQRc, bits)
    simStego = simplified_stego(exQRc, bits, w)

    if method == "h":
        exBin = extract_message(exQRc, simStego, location, 1)
    elif method == "v":
        exBin = extract_message(exQRc, simStego, location, 2)

    print(f"提取的二進制數據長度: {len(exBin)}")
    print(f"提取的二進制數據: {exBin}")

    # 只提取 EL 值
    exEL = get_infor_from_array1D(exBin, 1, 8)[0]
    print(f"提取的 EL 值：{exEL}")

    exImg_p = generate_predict_image(exImg_diff, model)
    exdiffA_e = two_array2D_add_or_subtract(exImg_diff, exImg_p, -1)
    
    # 4. 解碼後的差值嵌入直方圖
    exdiffId_e, exdiffNum_e = generate_different_histogram_without_frame(exdiffA_e, list(range(-range_x, range_x+1)), [0]*(range_x*2+1))
    save_difference_histogram(exdiffNum_e, f"./pred_and_QR/outcome/histogram/{imgName}/difference/de_{imgName}_diffembed.{filetype}", 
                              "Decoded Difference Embedding Histogram", range_x)

    exdiffA_s, exInf = decode_image_difference_embeding(exdiffA_e, exEL, len(inInf))

    # 5. 解碼後的差值平移直方圖
    exdiffId_s, exdiffNum_s = generate_different_histogram_without_frame(exdiffA_s, list(range(-range_x, range_x+1)), [0]*(range_x*2+1))
    save_difference_histogram(exdiffNum_s, f"./pred_and_QR/outcome/histogram/{imgName}/difference/de_{imgName}_diffshift.{filetype}", 
                              "Decoded Difference Shift Histogram", range_x)

    exdiffA = decode_image_different_shift(exdiffA_s, exEL)

    # 6. 解碼後的原始差值直方圖  
    exdiffId, exdiffNum = generate_different_histogram_without_frame(exdiffA, list(range(-range_x, range_x+1)), [0]*(range_x*2+1))
    save_difference_histogram(exdiffNum, f"./pred_and_QR/outcome/histogram/{imgName}/difference/de_{imgName}_diff.{filetype}", 
                              "Decoded Original Difference Histogram", range_x)

    print(f"解碼的差值信息長度: {len(exInf)}")

    # 解密完成後，將恢復的圖像轉回原方向
    exImg_diff = image_rerotation(exImg_diff, rotation_times)
                              
    print(f"原始 EL 值：{EL}")
    print(f"提取的 EL 值：{exEL}")  
    print(f"DE 影像還原正確：{flag_img}")
    print(f"取出二維碼資訊正確：{flag_inf}")
    print(f"原始隱藏資訊長度：{len(inInf)}")  
    print(f"提取的隱藏資訊長度：{len(exInf)}")
    print(f"隱藏資訊提取正確：{np.array_equal(inInf[:len(exInf)], exInf)}")

    if flag_img and flag_inf and np.array_equal(inInf[:len(exInf)], exInf):
        print("...解密完成...") 
    else:
        print("...解密失敗...")
        if not flag_img:
            print("DE 影像還原錯誤")
        if not flag_inf:
            print("取出二維碼資訊錯誤") 
        if not np.array_equal(inInf[:len(exInf)], exInf):
            print("差值直方圖解碼，隱藏資訊提取錯誤")

    # 顯示圖像
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.imshow(origImg, cmap="gray")
    plt.title(f"{imgName}")
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(image_rerotation(img_diff, rotation_times), cmap="gray")
    plt.title("difference histogram") 
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(image_rerotation(img_de, rotation_times), cmap="gray")
    plt.title("difference expansion")
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(img_h, cmap="gray")
    plt.title("histogram shift")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()