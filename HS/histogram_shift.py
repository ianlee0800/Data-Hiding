
# Importing libraries
import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os

# Vectorized function for histogram shifting
def histogram_shifting_vectorized(channel, random_number):
    hist, bin_edges = np.histogram(channel, bins=256, range=(0, 256), density=True)
    cdf = hist.cumsum()
    
    cdf_min = cdf.min()
    cdf_max = cdf.max()
    
    if cdf_max == cdf_min:
        cdf_diff = np.zeros_like(cdf)
    else:
        cdf_diff = (cdf - cdf_min) * random_number / (cdf_max - cdf_min)

    shifted_channel = np.interp(channel, bin_edges[:-1], cdf_diff)
    return shifted_channel

# Function to calculate PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 10 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# Function to calculate SSIM
def calculate_ssim(img1, img2):
    return ssim(img1, img2, win_size=3, data_range=img2.max() - img2.min(), multichannel=True)

# Load the original image
imgName = input("Image name: ")
fileType = "png"

# File existence check
if not os.path.exists(f"./HS/images/{imgName}.{fileType}"):
    print("Image file does not exist. Please check the file path.")
else:
    origImg = cv2.imread(f"./HS/images/{imgName}.{fileType}")
    
    if origImg is not None:
        # Convert to HSV
        img_hsv = cv2.cvtColor(origImg, cv2.COLOR_BGR2HSV)
        
        # Calculate mean saturation
        saturation = img_hsv[:,:,1].mean()
        
        # Saturation threshold
        saturation_threshold = 30
        
        if saturation > saturation_threshold:
            print("COLOR IMAGE")
            random_numbers = [random.randint(0, 255) for _ in range(3)]
            stegoImg = np.zeros_like(origImg)
            
            for i in range(3):
                stegoImg[:,:,i] = histogram_shifting_vectorized(origImg[:,:,i], random_numbers[i])
        else:
            print("GRAYSCALE IMAGE")
            random_number = random.randint(0, 255)
            stegoImg = np.zeros_like(origImg)
            stegoImg[:,:,0] = histogram_shifting_vectorized(origImg[:,:,0], random_number)
        
        # Metrics
        psnr_value = calculate_psnr(origImg, stegoImg)
        ssim_score = calculate_ssim(origImg, stegoImg)
        
        payload_bits = 3 * 8 if saturation > saturation_threshold else 8
        payload_size = payload_bits * origImg.shape[0] * origImg.shape[1]
        bpp = payload_size / (origImg.shape[0] * origImg.shape[1])
        
        print(f"Payload Size (bits): {payload_size}")
        print(f"Bits Per Pixel (bpp): {bpp}")
        print(f"PSNR: {psnr_value}")
        print(f"SSIM: {ssim_score}")
        
        # Save stego-image
        outcome_path = "./HS/outcome/"
        if not os.path.exists(outcome_path):
            os.makedirs(outcome_path)
        cv2.imwrite(os.path.join(outcome_path, f"{imgName}_marked_histogram.{fileType}"), stegoImg)

        # Plot and save histogram
        plt.hist(origImg.ravel(), bins=256, range=(0, 256), density=True, alpha=0.75, color='blue')
        plt.xlabel('Pixel Value')
        plt.ylabel('Normalized Frequency')
        plt.title('Histogram of Original Image')
        plt.grid(True)
        plt.savefig(f"./HS/histogram/{imgName}_histogram.png")
    else:
        print("Failed to load the image. Check the file path.")
