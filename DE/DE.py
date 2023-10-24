import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import random

def number_to_bits(number, length):
    return np.array([int(bit) for bit in format(number, '0' + str(length) + 'b')])

def difference_expansion(pixel, bit, adaptive=True, threshold=128):
    l = pixel // 2
    h = pixel - l
    
    # 自適應選擇：僅在像素值低於某個閾值時進行嵌入
    if adaptive and pixel > threshold:
        return pixel
    
    # 簡單的錯誤容忍：如果高位（h）過小，則不進行嵌入
    if h < 2:
        return pixel
    
    h_e = (h << 1) + bit
    
    return np.clip(l + h_e, 0, 255).astype(np.uint8)

def extract_and_restore(stego_pixel, adaptive=True, threshold=128):
    l = stego_pixel // 2
    h_e = stego_pixel - l

    # 如果是自適應模式，且像素大於閾值，則直接返回像素並提取bit為0
    if adaptive and stego_pixel > threshold:
        return stego_pixel, 0
    
    # 從擴展後的高位（h_e）中提取bit
    extracted_bit = h_e & 1  # 取最低位

    # 還原高位
    h = h_e >> 1
    
    # 還原原始像素
    restored_pixel = np.clip(l + h, 0, 255).astype(np.uint8)

    return restored_pixel, extracted_bit

def calculate_ssim(img1, img2):
    # Set a default window size
    win_size = 7
    
    # Check if the image is grayscale or color
    if len(img1.shape) == 2 or img1.shape[2] == 1:
        # Grayscale image
        min_dim = min(img1.shape[0], img1.shape[1])
    else:
        # Color image
        min_dim = min(img1.shape[0], img1.shape[1])
        
    # Adjust the window size if it's too large for the image
    if min_dim < win_size:
        win_size = min_dim // 2 if min_dim // 2 % 2 != 0 else (min_dim // 2) - 1
        win_size = max(win_size, 1)  # Ensure it's at least 1
    
    # Calculate SSIM
    if len(img1.shape) == 2 or img1.shape[2] == 1:
        # Grayscale image
        return ssim(img1, img2, win_size=win_size)
    else:
        # Color image, we need to specify the channel_axis
        return ssim(img1, img2, win_size=win_size, channel_axis=2)

imgName = input("Image name: ")
fileType = "png"

if "_marked" in imgName:
    print("This is a stego image.")
    origImg = cv2.imread(f"./DE/outcome/{imgName}.{fileType}")
    stegoImg = origImg.copy()  # Missing in your code
    # Initialize restored image and bit array
    restoredImg = np.zeros_like(stegoImg)
    extracted_bits = []
    img_hsv = cv2.cvtColor(stegoImg, cv2.COLOR_BGR2HSV)
    saturation = img_hsv[:, :, 1].mean()
    saturation_threshold = 30

    if saturation > saturation_threshold:  # Color Image
        channels = ['B', 'G', 'R']
    
        for idx, color in enumerate(channels):
            for i in range(0, stegoImg.shape[0], 2):
                for j in range(0, stegoImg.shape[1], 2):
                    restored_pixel, extracted_bit = extract_and_restore(stegoImg[i, j, idx])
                    extracted_bits.append(extracted_bit)
                    restoredImg[i, j, idx] = restored_pixel

    else:  # Grayscale Image
        stegoImg_gray = cv2.cvtColor(stegoImg, cv2.COLOR_BGR2GRAY)
    
        for i in range(0, stegoImg_gray.shape[0], 2):
            for j in range(0, stegoImg_gray.shape[1], 2):
                restored_pixel, extracted_bit = extract_and_restore(stegoImg_gray[i, j])
                extracted_bits.append(extracted_bit)
                restoredImg[i, j] = [restored_pixel] * 3  # Update all three channels

    # Convert extracted bits to numpy array
    extracted_bits = np.array(extracted_bits).astype(np.uint8)
    # Convert bits to bytes
    extracted_bytes = np.packbits(extracted_bits)
    # Convert bytes to original data (assuming the original data is text)
    extracted_text = extracted_bytes.tobytes().decode('utf-8', errors='ignore')
    
    cv2.imwrite(f"./DE/restored/{imgName}_restored.{fileType}", restoredImg)

else:
    print("This is not a stego image.")
    origImg = cv2.imread(f"./DE/images/{imgName}.{fileType}")
    if origImg is None:
        print("Failed to load the image. Check the file path.")
    else:
        stegoImg = origImg.copy()  # Create a stego image
        img_hsv = cv2.cvtColor(stegoImg, cv2.COLOR_BGR2HSV)
        saturation = img_hsv[:, :, 1].mean()
        saturation_threshold = 30

        if saturation > saturation_threshold:  # Color Image
            print("COLOR IMAGE")
            channels = ['B', 'G', 'R']

            for idx, color in enumerate(channels):
                max_payload_size = origImg.shape[0] * origImg.shape[1] // 4
                min_payload_size = 0
                payload_size = random.randint(min_payload_size, max_payload_size)
                payload = np.random.randint(0, 2, payload_size).astype(np.uint8)

                bit_index = 0
                for i in range(0, origImg.shape[0], 2):
                    for j in range(0, origImg.shape[1], 2):
                        if bit_index < payload_size:
                            bit = payload[bit_index]
                            origImg[i, j, idx] = difference_expansion(origImg[i, j, idx], bit)
                            bit_index += 1

        else:  # Grayscale Image
            print("GRAYSCALE IMAGE")
            stegoImg = cv2.cvtColor(stegoImg, cv2.COLOR_BGR2GRAY)
            max_payload_size = origImg.shape[0] * origImg.shape[1] // 4
            min_payload_size = 0
            payload_size = random.randint(min_payload_size, max_payload_size)
            payload = np.random.randint(0, 2, payload_size).astype(np.uint8)

            bit_index = 0
            for i in range(0, stegoImg.shape[0], 2):
                for j in range(0, stegoImg.shape[1], 2):
                    if bit_index < payload_size:
                        bit = payload[bit_index]
                        stegoImg[i, j] = difference_expansion(stegoImg[i, j], bit)
                        bit_index += 1

            stegoImg = cv2.cvtColor(stegoImg, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for consistent PSNR and SSIM

        # Calculate PSNR and SSIM using the stego image
        psnr = cv2.PSNR(origImg, stegoImg)
        ssim_score = calculate_ssim(origImg, stegoImg)
        bpp = payload_size / (origImg.shape[0] * origImg.shape[1] / 4)
        
        print("Image Size:", origImg.shape[0], "x" , origImg.shape[1])
        print("Payload Size:", payload_size)
        print("Bits Per Pixel (bpp):", bpp)
        print("PSNR:", psnr)
        print("SSIM:", ssim_score)

        # Save the stego-image in ./DE/outcome
        cv2.imwrite(f"./DE/outcome/{imgName}_marked.{fileType}", stegoImg)
