import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
import json

# Function to calculate SSIM
def calculate_ssim(img1, img2):
    return ssim(img1, img2, win_size=3, data_range=img2.max() - img2.min(), multichannel=True)

# Function to save peak values and payload sizes
def save_metadata_for_channel(metadata, imgName, channel, metadata_dir="./HS/metadata"):
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)
    metadata_path = os.path.join(metadata_dir, f"{imgName}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

# Function to load peak values and payload sizes
def load_metadata_for_channel(imgName, metadata_dir="./HS/metadata"):
    metadata_path = os.path.join(metadata_dir, f"{imgName}_metadata.json")
    if not os.path.exists(metadata_path):
        print("Metadata file does not exist. Please check the file path.")
        return None
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

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
                if payload_index < len(payload_bits):
                    print(f"Embedding bit {payload_bits[payload_index]} at ({y}, {x})")
                    if payload_bits[payload_index] == 1:
                        stego_img[y, x] += shift
                    map_shifted[y, x] = 1
                    payload_index += 1
            elif (shift == 1 and stego_img[y, x] > peak and stego_img[y, x] < 255) or \
                 (shift == -1 and stego_img[y, x] < peak and stego_img[y, x] > 0):
                stego_img[y, x] += shift

    # Check if all bits were embedded
    if payload_index < len(payload_bits):
        raise ValueError("Not all payload bits were embedded.")

    return stego_img, map_shifted

def extract_data_with_histogram_shifting(stego_img, map_shifted, peak, shift):
    height, width = stego_img.shape[:2]
    extracted_payload = []  # To hold the extracted bits

    # Create a copy of the stego image to restore the original image
    restored_img = stego_img.copy()

    for y in range(height):
        for x in range(width):
            # Check if the pixel was used for embedding
            if map_shifted[y, x] == 1:
                # Shift pixel back to its original value
                if (shift == 1 and stego_img[y, x] == peak + 1) or \
                   (shift == -1 and stego_img[y, x] == peak - 1):
                    restored_img[y, x] -= shift
                    # The pixel was shifted to embed a '1'
                    extracted_payload.append(1)
                else:
                    # The pixel is at the peak value and was not shifted, hence it represents a '0'
                    extracted_payload.append(0)

    return extracted_payload, restored_img

def save_metadata_for_channel(metadata, imgName, metadata_dir="./HS/metadata"):
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)
    metadata_path = os.path.join(metadata_dir, f"{imgName}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

def load_metadata_for_channel(imgName, metadata_dir="./HS/metadata"):
    metadata_path = os.path.join(metadata_dir, f"{imgName}_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Metadata file does not exist. Please check the file path.")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def hs_encoding():
    imgName = input("Image name: ")
    fileType = "png"
    imagePath = f"./HS/images/{imgName}.{fileType}"
    maps_dir = "./HS/maps"

    # Check for image existence
    if not os.path.exists(imagePath):
        print("Image file does not exist. Please check the file path.")
        return None

    origImg = cv2.imread(imagePath)
    if origImg is None:
        print("Failed to load the image. Check the file path.")
        return None
    
    # Make a copy of the original image for PSNR and SSIM comparison
    stego_img = origImg.copy()

    img_hsv = cv2.cvtColor(origImg, cv2.COLOR_BGR2HSV)
    saturation = img_hsv[:,:,1].mean()
    saturation_threshold = 30
    metadata = {'peak_values': [], 'payload_sizes': [], 'shifts': []}

    if saturation > saturation_threshold:
        print("COLOR IMAGE")

        # Initialize a 2D map for the color image, assuming the last dimension is for color channels
        map_shifted_all = np.zeros(origImg.shape[:2], dtype=np.uint8)

        for i in range(3):  # Assuming BGR format
            channel_img = origImg[:, :, i]
            hist = cv2.calcHist([channel_img], [0], None, [256], [0, 256]).flatten()
            peak = int(np.argmax(hist))
            shift = -1 if peak == 255 else 1
            payload_bits = [random.randint(0, 1) for _ in range(int(hist[peak]))]
            print(f"Payload bits (before embedding): {payload_bits}")

            stego_channel, map_shifted = embed_data_with_histogram_shifting(channel_img, payload_bits)

            # Update the stego image and combine the map_shifted arrays
            stego_img[:, :, i] = stego_channel
            map_shifted_all = np.bitwise_or(map_shifted_all, map_shifted)

            # Save metadata for each channel
            metadata['peak_values'].append(peak)
            metadata['payload_sizes'].append(len(payload_bits))
            metadata['shifts'].append(shift)

        # Save the combined map for all channels after processing all channels
        np.save(f"{maps_dir}/{imgName}_map_shifted.npy", map_shifted_all)


    else:
        print("GRAYSCALE IMAGE")
        if origImg.ndim == 3:
            origImg = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([origImg], [0], None, [256], [0, 256]).flatten()
        peak = int(np.argmax(hist))
        shift = -1 if peak == 255 else 1
        payload_bits = [random.randint(0, 1) for _ in range(int(hist[peak]))]

        stego_img, map_shifted = embed_data_with_histogram_shifting(origImg, payload_bits)

        # Save metadata for grayscale image
        metadata['peak_values'].append(peak)
        metadata['payload_sizes'].append(len(payload_bits))
        metadata['shifts'].append(shift)
        
        # After embedding data into the image
        np.save(f"./HS/maps/{imgName}_map_shifted.npy", map_shifted)
            
    # Calculate metrics for the grayscale image
    psnr_value = cv2.PSNR(origImg, stego_img)
    ssim_score = ssim(origImg, stego_img, win_size=3, data_range=stego_img.max() - stego_img.min())
            
    # Assuming 1 bit per pixel for payload_bits
    bpp = len(payload_bits) / (origImg.size)
          
    print(f"Payload Size (bits): {len(payload_bits)}")
    print(f"Bits Per Pixel (bpp): {bpp}")
    print(f"PSNR: {psnr_value:.4f}")
    print(f"SSIM: {ssim_score:.4f}")

    # Save the stego image
    outcome_path = "./HS/outcome/"
    if not os.path.exists(outcome_path):
        os.makedirs(outcome_path)
    cv2.imwrite(os.path.join(outcome_path, f"{imgName}_marked_histogram.{fileType}"), stego_img)
    print(f"Stego image saved at {outcome_path}{imgName}_marked_histogram.{fileType}")
    
    save_metadata_for_channel(metadata, imgName)
    return imgName

def hs_decoding(imgName):
    fileType = "png"
    stego_img_path = f"./HS/outcome/{imgName}_marked_histogram.{fileType}"
    map_shifted_path = f"./HS/maps/{imgName}_map_shifted.npy"
    original_img_path = f"./HS/images/{imgName}.{fileType}"

    # Check if the required files exist
    if not os.path.exists(stego_img_path) or not os.path.exists(map_shifted_path) or not os.path.exists(original_img_path):
        print("One or more required files are missing. Please check the file paths.")
        return

    # Load images and metadata
    stego_img = cv2.imread(stego_img_path, cv2.IMREAD_UNCHANGED)
    original_img = cv2.imread(original_img_path, cv2.IMREAD_UNCHANGED)
    map_shifted = np.load(map_shifted_path)
    metadata = load_metadata_for_channel(imgName)
    peak_values = metadata['peak_values']
    shifts = metadata['shifts']

    # Check the dimensions of map_shifted to determine if the image is color or grayscale
    is_grayscale = map_shifted.ndim == 2 and stego_img.ndim == 2
    
    if is_grayscale:
        print("Decoding GRAYSCALE IMAGE")
        extracted_data, restored_img = extract_data_with_histogram_shifting(
            stego_img, map_shifted, peak_values[0], shifts[0]
        )
    else:
        print("Decoding COLOR IMAGE")
        restored_img = stego_img.copy()
        extracted_data = []  # Initialize an empty list to hold data from all channels
        for ch in range(3):
            channel_extracted_data, restored_channel = extract_data_with_histogram_shifting(
                stego_img[:, :, ch], map_shifted, peak_values[ch], shifts[ch]
            )
            restored_img[:, :, ch] = restored_channel
            extracted_data.extend(channel_extracted_data)

    # Extract binary data to string without assuming it represents ASCII characters
    # Ensure this is within the same scope where `extracted_payload` is defined
    binary_payload = ''.join(map(str, extracted_data))
    print(f"Extracted binary payload: {binary_payload}")


    # Calculate PSNR
    psnr_value = cv2.PSNR(original_img, restored_img)
    print(f"PSNR: {psnr_value:.4f}")

    # Before SSIM calculation
    min_dimension = min(original_img.shape[0], original_img.shape[1])
    win_size = min(7, min_dimension)
    if win_size % 2 == 0:  # Make sure window size is odd
        win_size = max(win_size - 1, 1)  # Ensure win_size is at least 1

    # Calculate SSIM
    try:
        ssim_value = ssim(
            original_img, 
            restored_img, 
            win_size=win_size, 
            multichannel=(original_img.ndim == 3),
            channel_axis=-1 if (original_img.ndim == 3) else None
        )
        print(f"SSIM: {ssim_value:.4f}")
    except ValueError as e:
        print(f"SSIM calculation failed with error: {e}")

    # Save the restored image
    restored_path = "./HS/restored"
    if not os.path.exists(restored_path):
        os.makedirs(restored_path)
    cv2.imwrite(os.path.join(restored_path, f"{imgName}_restored.{fileType}"), restored_img)
    print(f"Restored image saved successfully.")

def main():
    choice = input("Would you like to encode or decode? (E/D): ").upper()

    if choice == 'E':
        imgName = hs_encoding()  # Ensure this returns the image name upon successful encoding
        if imgName:
            post_encode_choice = input("Would you like to continue with decoding? (Y/N): ").upper()
            if post_encode_choice == 'Y':
                hs_decoding(imgName)  # Pass the image name to the decoding function
            else:
                print("Encoding complete. Exiting the program.")
        else:
            print("Encoding failed. Exiting the program.")

    elif choice == 'D':
        imgName = input("Enter the name of the stego image: ")
        hs_decoding(imgName)  # Make sure hs_decoding is adjusted to take imgName as an argument

    else:
        print("Invalid choice. Please restart the program and select either E or D.")

if __name__ == "__main__":
    main()
