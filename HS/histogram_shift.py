import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os  # Import the 'os' module


# Function for histogram shifting
def histogram_shifting(channel, random_number):
    # Calculate the cumulative distribution function (CDF) of the channel
    hist, bin_edges = np.histogram(channel, bins=256, range=(0, 256), density=True)
    cdf = hist.cumsum()

    # Calculate the CDF difference for shifting
    cdf_min = cdf.min()
    cdf_max = cdf.max()
    
    if cdf_max == cdf_min:
        cdf_diff = np.zeros_like(cdf)
    else:
        cdf_diff = (cdf - cdf_min) * random_number / (cdf_max - cdf_min)

    # Interpolate the values based on the CDF difference
    shifted_channel = np.interp(channel, bin_edges[:-1], cdf_diff)

    return shifted_channel


# Calculate Peak Signal-to-Noise Ratio (PSNR)
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # Images are identical, so PSNR is infinity
    max_pixel = 255.0
    psnr = 10 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# Calculate Structural Similarity Index (SSIM)
def calculate_ssim(img1, img2):
    return ssim(img1, img2, win_size=3, data_range=img2.max() - img2.min(), multichannel=True)

# Load the original image
imgName = input("Image name: ")
fileType = "png"  # You can adjust this as needed
origImg = cv2.imread(f"./HS/images/{imgName}.{fileType}")

if origImg is not None:
    # Convert the image to HSV color space
    img_hsv = cv2.cvtColor(origImg, cv2.COLOR_BGR2HSV)
    
    # Calculate the mean saturation value
    saturation = img_hsv[:,:,1].mean()
    
    # Define a threshold to classify as color or grayscale
    saturation_threshold = 30 # You can adjust the threshold
    
    if saturation > saturation_threshold:
        print("COLOR IMAGE")

        # Generate random numbers for RGB channels
        random_number_R = random.randint(0, 255)
        random_number_G = random.randint(0, 255)
        random_number_B = random.randint(0, 255)

        # Create a stego image
        stegoImg = origImg.copy()

        for i in range(origImg.shape[0]):
            for j in range(origImg.shape[1]):
                stegoImg[i, j, 2] = histogram_shifting(origImg[i, j, 2], random_number_R)
                stegoImg[i, j, 1] = histogram_shifting(origImg[i, j, 1], random_number_G)
                stegoImg[i, j, 0] = histogram_shifting(origImg[i, j, 0], random_number_B)
    else:
        print("GRAYSCALE IMAGE")

        # Generate a random number for grayscale
        random_number = random.randint(0, 255)

        # Create a stego image
        stegoImg = origImg.copy()

        for i in range(origImg.shape[0]):
            for j in range(origImg.shape[1]):
                stegoImg[i, j, 0] = histogram_shifting(origImg[i, j, 0], random_number)

    # Calculate PSNR
    psnr_value = cv2.PSNR(origImg, stegoImg)

    # Calculate SSIM
    ssim_score = calculate_ssim(origImg, stegoImg)

    # Calculate payload size in bits
    if saturation > saturation_threshold:
        # For color images
        payload_bits = 3 * 8  # 3 random values, each with 8 bits
    else:
        # For grayscale images
        payload_bits = 8  # 1 random value with 8 bits

    payload_size = payload_bits * origImg.shape[0] * origImg.shape[1]
    bpp = payload_size / (origImg.shape[0] * origImg.shape[1])

    print("Payload Size (bits):", payload_size)
    print("Bits Per Pixel (bpp):", bpp)
    print("PSNR:", psnr_value)
    print("SSIM:", ssim_score)
    
    # Save the stego-image
    outcome_path = "./HS/outcome/"
    if not os.path.exists(outcome_path):
        os.makedirs(outcome_path)
    cv2.imwrite(os.path.join(outcome_path, f"{imgName}_marked_histogram.{fileType}"), stegoImg)

    # Plot and save the histogram
    histogram, bins = np.histogram(origImg.ravel(), bins=256, range=(0, 256))
    plt.hist(origImg.ravel(), bins=256, range=(0, 256), density=True, alpha=0.75, color='blue')
    plt.xlabel('Pixel Value')
    plt.ylabel('Normalized Frequency')
    plt.title('Histogram of Original Image')
    plt.grid(True)
    plt.savefig(f"./HS/histogram/{imgName}_histogram.png")

else:
    print("Failed to load the image. Check the file path.")
