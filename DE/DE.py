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
origImg = cv2.imread(f"./DE/images/{imgName}.{fileType}")

markedImg = origImg.copy()

if origImg is not None:
    height, width, channels = origImg.shape  # Get the number of channels

    if channels == 3:
        # Color image
        B, G, R = cv2.split(origImg)

        # Generate random numbers for each channel
        random_number_R = random.randint(0, 255)
        random_number_G = random.randint(0, 255)
        random_number_B = random.randint(0, 255)
        print("Random Numbers (R, G, B):", random_number_R, random_number_G, random_number_B)

    elif channels == 1:
        # Grayscale image

        # Define 'jump_gray' list for grayscale
        jump_gray = []
        
         # Populate the 'jump_gray' list with coordinates for hiding data
        for _ in range(8):
            y = random.randint(0, height - 1)
            x = random.randint(0, width - 2)  # Adjusted to avoid out-of-bounds error
            jump_gray.append((y, x))

        # Generate a random number
        random_number = random.randint(0, 255)
        print("Random Number:", random_number)

        for i in range(8):
            bit = (random_number >> i) & 1
            y, x = jump_gray[i]
            left = int(markedImg[y, x])
            right = int(markedImg[y, x + 1])
            left_e, right_e = difference_expansion(left, right, bit)
            markedImg[y, x] = left_e
            markedImg[y, x + 1] = right_e

        # Continue with the code for grayscale images

    else:
        print("Unsupported image format. It should be either grayscale or color (3-channel).")

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
