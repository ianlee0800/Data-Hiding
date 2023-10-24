import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import random

def number_to_bits(number, length):
    return np.array([int(bit) for bit in format(number, '0' + str(length) + 'b')])

def difference_expansion(pixel, bit):
    l = pixel // 2
    h = pixel - l
    h_e = (h << 1) + bit
    return np.clip(l + h_e, 0, 255).astype(np.uint8)

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
origImg = cv2.imread(f"./DE/images/{imgName}.{fileType}")
origImgCopy = origImg.copy()  # Save a copy of the original image for comparison


if origImg is None:
    print("Failed to load the image. Check the file path.")
else:
    img_hsv = cv2.cvtColor(origImg, cv2.COLOR_BGR2HSV)
    saturation = img_hsv[:, :, 1].mean()
    saturation_threshold = 30

    if saturation > saturation_threshold:  # Color Image
        print("COLOR IMAGE")
        channels = ['B', 'G', 'R']

        for idx, color in enumerate(channels):
            payload_size = random.randint(100, 500000)
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
        payload_size = random.randint(100, 500000)
        payload = np.random.randint(0, 2, payload_size).astype(np.uint8)

        bit_index = 0
        grayImg = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)
        for i in range(0, grayImg.shape[0], 2):
            for j in range(0, grayImg.shape[1], 2):
                if bit_index < payload_size:
                    bit = payload[bit_index]
                    grayImg[i, j] = difference_expansion(grayImg[i, j], bit)
                    bit_index += 1

    # Calculate PSNR and SSIM using the stego image
    psnr = cv2.PSNR(origImgCopy, origImg)
    ssim_score = calculate_ssim(origImgCopy, origImg)
    bpp = payload_size / (origImg.shape[0] * origImg.shape[1] / 4)
    

    print("Bits Per Pixel (bpp):", bpp)
    print("PSNR:", psnr)
    print("SSIM:", ssim_score)

    # Save the stego-image in ./DE/outcome
    cv2.imwrite(f"./DE/outcome/{imgName}_marked.{fileType}", origImg)
