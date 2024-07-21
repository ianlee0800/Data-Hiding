import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from image_metrics import calculate_psnr, calculate_ssim
from image_processing import *
from data_embedding import *
from qr_code_utils import *
from cnn_model import load_model, generate_predict_image

def int_transfer_binary_single_intlist(array1D, bits=8):
    return np.unpackbits(np.array(array1D, dtype=np.uint8).reshape(-1, 1), axis=1)[:, -bits:].flatten()

def repeat_int_array(array, lenNewA):
    return (array * (lenNewA // len(array) + 1))[:lenNewA]

def get_infor_from_array1D(array1D, num, digit):
    padding = digit - (len(array1D) % digit) if len(array1D) % digit != 0 else 0
    padded_array = np.pad(array1D, (0, padding), 'constant', constant_values=(0))
    reshaped = padded_array.reshape(-1, digit)
    return np.packbits(reshaped, axis=1, bitorder='big').flatten()[:num]

def calculate_correct_ratio(true, extra):
    return sum(t == e for t, e in zip(true, extra)) / len(true)

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
    weight = [1, 2, 11, 12]  # 加權值

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

    print("生成預測圖像...")
    img_p = generate_predict_image(origImg, model)
    print(f"預測圖像數據類型: {img_p.dtype}")

    # X1: 預估式嵌入法，嵌入隨機位元
    diffA = two_array2D_add_or_subtract(origImg, img_p, -1)
    payload_diff = np.sum((diffA >= -EL) & (diffA <= EL)) - np.sum((diffA == -EL) | (diffA == EL))
    bpp_diff = round(payload_diff / origImg.size, 2)

    diffA_s = image_difference_shift(diffA, EL)
    diffA_e, inInf = image_difference_embeding(diffA_s, 0, EL, 0)
    img_diff = two_array2D_add_or_subtract(img_p, diffA_e, 1).astype(np.uint8)
    print(f"差值嵌入後圖像數據類型: {img_diff.dtype}")
    print(f"差值嵌入信息長度: {len(inInf)}")

    hist_diff = generate_histogram(img_diff)
    psnr_diff = calculate_psnr(origImg, img_diff)
    ssim_diff = calculate_ssim(origImg, img_diff)

    # 二維碼：模塊形狀調整，嵌入加權值與EL
    data_p = weight + [EL]
    bin_weight = int_transfer_binary_single_intlist(data_p)
    print(f"原始加權值: {data_p}")
    print(f"原始加權值的二進制表示: {bin_weight}")

    bits, simQrcode = simplified_qrcode(QRCImg)
    loc = find_locaion(QRCImg, bits)

    message = bin_weight  # 只嵌入原始的加權值，不進行重複
    print(f"QR碼嵌入信息長度: {len(message)}")

    if method == "h":
        QRCImg_m = horizontal_embedding(QRCImg, simQrcode, loc, message, k)
    elif method == "v":
        QRCImg_m = vertical_embedding(QRCImg, simQrcode, loc, message, k)
    print(f"QR碼嵌入的原始信息: {message}")

    # X2：DE嵌入法，嵌入二進制二維碼
    binQRC_m = array2D_transfer_to_array1D(QRCImg_m)
    img_de, hidemap, locationmap, payload_de = different_expansion_embeding_array1D(img_diff, binQRC_m)
    img_de = img_de.astype(np.uint8)
    print(f"DE嵌入後圖像數據類型: {img_de.dtype}")

    hist_de = generate_histogram(img_de)
    psnr_de = calculate_psnr(origImg, img_de)
    ssim_de = calculate_ssim(origImg, img_de)

    binQRC = array2D_transfer_to_array1D(QRCImg)
    ratio_qr = calculate_correct_ratio(binQRC, binQRC_m)
    ssim_q = calculate_ssim(QRCImg, QRCImg_m)

    # X3：直方圖平移嵌入法，嵌入可嵌入地圖
    bin_map = int_transfer_binary_single_intlist(locationmap, 9)
    img_h, peak, payload_h = histogram_data_hiding(img_de, 1, bin_map)
    print(f"直方圖平移嵌入後圖像數據類型: {img_h.dtype}")

    hist_h = generate_histogram(img_h)
    psnr_h = calculate_psnr(origImg, img_h)
    ssim_h = calculate_ssim(origImg, img_h)

    # 輸出結果
    print(f"原影像: {imgName}, 原二維碼: {qrcodeName}")
    print(f"加權值={weight}, EL={EL}")
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
    exImg_diff, exInf_qr = decode_different_expansion(markedImg, hidemap)
    flag_img = np.array_equal(exImg_diff, img_diff)
    flag_inf = np.array_equal(binQRC_m[:40], exInf_qr)  # 只比較前40位
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
    print(f"原始嵌入的二進制數據: {message}")
    exWeight = get_infor_from_array1D(exBin, 5, 8)
    print(f"提取的加權值（包括 EL）：{exWeight}")

    exImg_p = generate_predict_image(exImg_diff, model)
    exdiffA_e = two_array2D_add_or_subtract(exImg_diff, exImg_p, -1)
    exdiffA_s, exInf = decode_image_difference_embeding(exdiffA_e, EL, len(inInf))
    exdiffA = decode_image_different_shift(exdiffA_s, EL)
    print(f"解碼的差值信息長度: {len(exInf)}")

    print(f"原始加權值：{data_p}")
    print(f"提取的加權值：{exWeight}")
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
    plt.imshow(img_diff, cmap="gray")
    plt.title("difference histogram")
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(img_de, cmap="gray")
    plt.title("difference expansion")
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(img_h, cmap="gray")
    plt.title("histogram shift")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()