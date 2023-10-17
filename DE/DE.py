import cv2
import numpy as np
import math
import random
from skimage.metrics import structural_similarity as ssim

# Function for difference expansion
def difference_expansion(left, right, hide):
    l = int((left + right) / 2)
    if left >= right:
        h = left - right
        h_e = 2 * h + hide
        left_e = l + int((h_e + 1) / 2)
        right_e = l - int(h_e / 2)
    elif left < right:
        h = right - left
        h_e = 2 * h + hide
        left_e = l - int(h_e / 2)
        right_e = l + int((h_e + 1) / 2)
    return left_e, right_e

# Calculate Peak Signal-to-Noise Ratio (PSNR)
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # Images are identical, so PSNR is infinity
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# Calculate Structural Similarity Index (SSIM)
def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=img2.max() - img2.min(), multichannel=True)


# Load the original image
imgName = input("Image name: ")
fileType = "png"  # You can adjust this as needed
origImg = cv2.imread(f"./images/{imgName}.{fileType}")

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
        
        # Embed these random numbers in the corresponding channels
        origImg[:, :, 2] = random_number_R
        origImg[:, :, 1] = random_number_G
        origImg[:, :, 0] = random_number_B
    else:
        print("GRAYSCALE IMAGE")
        
        # Generate a random number for grayscale
        random_number = random.randint(0, 255)
        
        # Embed this random number in the grayscale channel
        origImg = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)
        origImg = origImg.astype(np.uint8)
        origImg[:, :] = random_number
    
    markedImg = origImg.copy()
    height, width, channels = origImg.shape  # Get the number of channels
    
    # Continue with the code for grayscale images
    # Continue with the code for both color and grayscale images
    # Calculate SSIM with a specified win_size (e.g., 3x3)
    psnr = calculate_psnr(origImg, markedImg)
    ssim_score = ssim(origImg, markedImg, win_size=3)  # Adjust win_size
    
    # Define and calculate the payload based on your data embedding
    
    payload = height * width // 8  # Assuming each pixel hides 1 bit

    # Print results
    print("Payload Size:", payload)
    print("PSNR:", psnr)
    print("SSIM:", ssim_score)

    # Save the stego-image
    cv2.imwrite(f"./outcome/{imgName}_marked.{fileType}", markedImg)

else:
    print("Failed to load the image. Check the file path.")
