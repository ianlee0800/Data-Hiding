import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os

# Function to calculate SSIM
def calculate_ssim(img1, img2):
    return ssim(img1, img2, win_size=3, data_range=img2.max() - img2.min(), multichannel=True)

# New function for histogram shifting and data embedding
def embed_data_with_histogram_shifting(orig_img, payload_bits):
    height, width = orig_img.shape[:2]  # Safeguard for both grayscale and color
    levels = 256

    # Calculate the histogram and find the peak
    hist = cv2.calcHist([orig_img], [0], None, [levels], [0, levels]).flatten()
    peak = int(np.argmax(hist))

    # Decide shift direction
    shift = -1 if peak == 255 else 1

    # Prepare the stego image
    stego_img = orig_img.copy()

    # Embed the payload
    payload_index = 0
    max_payload_size = hist[peak]  # The max number of bits we can embed is the count of the peak level
    map_shifted = np.zeros_like(orig_img, dtype=np.uint8)  # Map of shifted pixels

    # Make sure the payload does not exceed the number of pixels at the peak level
    if len(payload_bits) > hist[peak]:
        raise ValueError("Payload exceeds the maximum size that can be embedded.")

    for y in range(height):
        for x in range(width):
            if stego_img[y, x] == peak:
                if payload_index < len(payload_bits) and payload_bits[payload_index] == 1:
                    stego_img[y, x] += shift
                    map_shifted[y, x] = 1  # Mark the pixel as shifted
                payload_index += 1
            elif (shift == 1 and stego_img[y, x] > peak and stego_img[y, x] < 255) or \
                 (shift == -1 and stego_img[y, x] < peak and stego_img[y, x] > 0):
                stego_img[y, x] += shift

    # Check if all bits were embedded
    if payload_index < len(payload_bits):
        raise ValueError("Not all payload bits were embedded.")

    return stego_img, map_shifted

# Load the original image
imgName = input("Image name: ")
fileType = "png"

# File existence check
if not os.path.exists(f"./HS/images/{imgName}.{fileType}"):
    print("Image file does not exist. Please check the file path.")
else:
    origImg = cv2.imread(f"./HS/images/{imgName}.{fileType}")
    
    if origImg is not None:
        # Convert to HSV to check saturation
        img_hsv = cv2.cvtColor(origImg, cv2.COLOR_BGR2HSV)
        saturation = img_hsv[:,:,1].mean()
        saturation_threshold = 30  # Define a threshold for saturation
        
        # Initialize stegoImg to ensure it's the correct shape
        stegoImg = np.zeros_like(origImg)
        
        if saturation > saturation_threshold:
            print("COLOR IMAGE")
            # Determine the minimum peak pixel count across all channels
            min_peak_pixels = min(
                cv2.calcHist([origImg[:, :, i]], [0], None, [256], [0, 256]).flatten().max()
                for i in range(3)
            )

            # Generate payload bits based on the minimum peak pixel count
            payload_bits = [random.randint(0, 1) for _ in range(int(min_peak_pixels))]

            # Embed the payload into each channel and plot histograms
            for i, color in enumerate(['blue', 'green', 'red']):  # OpenCV uses BGR format
                # Embed data
                stegoImg[:, :, i], map_shifted = embed_data_with_histogram_shifting(origImg[:, :, i], payload_bits)
                
                # Calculate histogram for current channel
                hist = cv2.calcHist([stegoImg[:, :, i]], [0], None, [256], [0, 256])
                
                # Plot histogram
                plt.figure(figsize=(10, 5))
                plt.plot(hist, color=color)
                plt.xlim([0, 256])
                plt.title(f'Histogram for {color.capitalize()} Channel')
                plt.xlabel('Pixel Value')
                plt.ylabel('Frequency')
                plt.grid(True)
                
                # Save histogram image
                plt.savefig(f"./HS/histogram/{imgName}_{color}_histogram.png")
                plt.close()  # Close the plot
        
        else:
            print("GRAYSCALE IMAGE")
            
            # Plot and save histogram
            plt.hist(origImg.ravel(), bins=256, range=(0, 256), density=True, alpha=0.75, color='blue')
            plt.xlabel('Pixel Value')
            plt.ylabel('Normalized Frequency')
            plt.title('Histogram of Original Image')
            plt.grid(True)
            plt.savefig(f"./HS/histogram/{imgName}_histogram.png")

            if origImg.ndim == 3:
                origImg = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)
                stegoImg = np.zeros_like(origImg)  # Update stegoImg for single channel

            # Calculate the histogram and find the peak
            hist = cv2.calcHist([origImg], [0], None, [256], [0, 256]).flatten()
            peak = int(np.argmax(hist))

            # Generate payload bits up to the number of pixels at the peak level
            max_payload_size = int(hist[peak])
            payload_bits = [random.randint(0, 1) for _ in range(max_payload_size)]
            
            # Embed the payload bits
            stegoImg, map_shifted = embed_data_with_histogram_shifting(origImg, payload_bits)
            
        # Ensure that the images match in shape and type before calling cv2.PSNR
        assert origImg.shape == stegoImg.shape, "The input images must have the same dimensions."
        assert origImg.dtype == stegoImg.dtype, "The input images must have the same data type."
            
        # Calculate metrics for the grayscale image
        psnr_value = cv2.PSNR(origImg, stegoImg)
        ssim_score = ssim(origImg, stegoImg, win_size=3, data_range=stegoImg.max() - stegoImg.min())
            
        # Assuming 1 bit per pixel for payload_bits
        bpp = len(payload_bits) / (origImg.size)
            
        print(f"Payload Size (bits): {len(payload_bits)}")
        print(f"Bits Per Pixel (bpp): {bpp}")
        print(f"PSNR: {psnr_value:.4f}")
        print(f"SSIM: {ssim_score:.4f}")

        # Save stego-image
        outcome_path = "./HS/outcome/"
        if not os.path.exists(outcome_path):
            os.makedirs(outcome_path)
        cv2.imwrite(os.path.join(outcome_path, f"{imgName}_marked_histogram.{fileType}"), stegoImg)


    else:
        print("Failed to load the image. Check the file path.")
