import cv2
import numpy as np
import math
import random
from skimage.metrics import structural_similarity as ssim

# Define your custom window size
win_size = 11  # Adjust to your desired window size

# Function for difference expansion
def difference_expansion(pixel, random_number):
    l = (pixel + random_number) // 2  # Use integer division
    h = np.abs(pixel - random_number)
    h_e = 2 * h

    # Ensure that the value stays within the valid range (0-255)
    pixel_e = np.clip(l + np.floor((h_e + 1) / 2), 0, 255).astype(np.uint8)

    return pixel_e




# Calculate Peak Signal-to-Noise Ratio (PSNR)
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # Images are identical, so PSNR is infinity
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# Calculate Structural Similarity Index (SSIM) with a custom window size
def calculate_ssim(img1, img2, win_size=7):
    ssim_score = ssim(img1, img2, win_size=win_size)
    return ssim_score


# Load the original image
imgName = input("Image name: ")
fileType = "png"  # You can adjust this as needed
origImg = cv2.imread(f"./images/{imgName}.{fileType}")

if origImg is not None:
    # Convert the image to HSV color space
    img_hsv = cv2.cvtColor(origImg, cv2.COLOR_BGR2HSV)

    # Calculate the mean saturation value
    saturation = img_hsv[:, :, 1].mean()

    # Define a threshold to classify as color or grayscale
    saturation_threshold = 30  # You can adjust the threshold

    if saturation > saturation_threshold:
        print("COLOR IMAGE")
        payload_bits = 3 * 8  # 3 random values, each with 8 bits

        # Generate random numbers for RGB channels
        random_number_R = random.randint(0, 255)
        random_number_G = random.randint(0, 255)
        random_number_B = random.randint(0, 255)

        # Create a stego image
        stegoImg = origImg.copy()

        for i in range(origImg.shape[0]):
            for j in range(origImg.shape[1]):
                r = difference_expansion(origImg[i, j, 2], random_number_R)
                g = difference_expansion(origImg[i, j, 1], random_number_G)
                b = difference_expansion(origImg[i, j, 0], random_number_B)
                
                stegoImg[i, j, 2] = r
                stegoImg[i, j, 1] = g
                stegoImg[i, j, 0] = b 
    else:
        print("GRAYSCALE IMAGE")
        payload_bits = 8  # 1 random value with 8 bits

        # Generate a random number for grayscale
        random_number = random.randint(0, 255)

        # Create a stego image
        stegoImg = origImg.copy()
        
        for i in range(origImg.shape[0]):
            for j in range(origImg.shape[1]):
                stegoImg[i, j] = difference_expansion(origImg[i, j], random_number)

    # Crop the images to the specified window size
    origImg_cropped = origImg[:win_size, :win_size]
    stegoImg_cropped = stegoImg[:win_size, :win_size]
    
    payload_size = payload_bits * origImg.shape[0] * origImg.shape[1]
    bpp = payload_size / (origImg.shape[0] * origImg.shape[1])
    
    
    # Calculate PSNR and SSIM using the stego image
    psnr = calculate_psnr(origImg, stegoImg)  # Calculate PSNR using the entire images
    ssim_score = calculate_ssim(origImg, stegoImg, win_size=3)  # Set your custom window size
    

    # Define and calculate the payload based on your data embedding
    payload = origImg.size // 8  # Assuming each pixel hides 1 bit
    

    # Print results
    print("Payload Size:", payload)
    print("Bits Per Pixel (bpp):", bpp)
    print("PSNR:", psnr)
    print("SSIM:", ssim_score)

    # Save the stego-image
    cv2.imwrite(f"./outcome/{imgName}_marked.{fileType}", origImg)

else:
    print("Failed to load the image. Check the file path.")
