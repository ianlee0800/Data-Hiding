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

def save_stego_histogram(stego_img, img_name, bins=256, save_dir="./HS/stego_histogram"):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Check if the image is color or grayscale
    if len(stego_img.shape) == 3 and stego_img.shape[2] == 3:  # Color image
        color = ('b', 'g', 'r')
        plt.figure()
        plt.title(f"{img_name} - RGB Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")

        for i, col in enumerate(color):
            hist = cv2.calcHist([stego_img], [i], None, [bins], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, bins])

        plt.savefig(f"{save_dir}/{img_name}_RGB_histogram.png")
    else:  # Grayscale image
        hist = cv2.calcHist([stego_img], [0], None, [bins], [0, 256])
        plt.figure()
        plt.title(f"{img_name} - Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(hist)
        plt.xlim([0, bins])

        plt.savefig(f"{save_dir}/{img_name}_grayscale_histogram.png")

def is_low_impact_pixel(img, y, x, peak, threshold=10):
    # Example criteria for low impact: pixel is in a darker region
    # or its value is close to the neighboring pixels
    height, width = img.shape[:2]
    pixel_value = img[y, x]

    # Check if pixel is in a darker region
    if pixel_value < threshold:
        return True

    # Check the similarity with neighboring pixels
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if 0 <= y + dy < height and 0 <= x + dx < width:
                if abs(int(img[y + dy, x + dx]) - pixel_value) > threshold:
                    return False
    return True

def embed_data_with_histogram_shifting(orig_img, payload_bits, low_impact_threshold=15, hybrid_mode=True):
    height, width = orig_img.shape[:2]
    levels = 256

    # Initialize map_shifted
    map_shifted = np.zeros_like(orig_img, dtype=np.uint8)

    # Calculate the histogram and find the peak
    hist = cv2.calcHist([orig_img], [0], None, [levels], [0, levels]).flatten()
    peak = int(np.argmax(hist))

    # Decide shift direction
    shift = -1 if peak == 255 else 1

    # Prepare the stego image
    stego_img = orig_img.copy()

    # Embed the payload
    payload_index = 0
    max_payload_size = hist[peak]

    # Edge case handling for peak at 0 or 255
    if peak == 0:
        shift = 1
    elif peak == 255:
        shift = -1

    # First, try embedding using only low-impact pixels
    for y in range(height):
        for x in range(width):
            if orig_img[y, x] == peak and is_low_impact_pixel(orig_img, y, x, peak, low_impact_threshold):
                map_shifted[y, x] = 1
                if payload_index < len(payload_bits):
                    if payload_bits[payload_index] == 1:
                        if (shift == 1 and stego_img[y, x] < 255) or (shift == -1 and stego_img[y, x] > 0):
                            stego_img[y, x] += shift
                    payload_index += 1

    # Hybrid mode embedding
    if payload_index < len(payload_bits) and hybrid_mode:
        print("Switching to hybrid embedding mode.")
        for y in range(height):
            for x in range(width):
                # Use other pixels if low-impact pixels are exhausted
                if orig_img[y, x] == peak and map_shifted[y, x] == 0:  # Not previously used for embedding
                    map_shifted[y, x] = 1  # Mark this pixel as used for embedding in hybrid mode
                    if payload_index < len(payload_bits):
                        if payload_bits[payload_index] == 1:
                            if (shift == 1 and stego_img[y, x] < 255) or (shift == -1 and stego_img[y, x] > 0):
                                stego_img[y, x] += shift
                        payload_index += 1

    # Check if all bits were embedded
    if payload_index < len(payload_bits):
        print(f"Payload index: {payload_index}, Total Payload Bits: {len(payload_bits)}")
        raise ValueError("Not all payload bits were embedded, even in hybrid mode.")

    return stego_img, map_shifted

def extract_data_with_histogram_shifting(stego_img, map_shifted, peak, shift):
    height, width = stego_img.shape[:2]
    extracted_payload = []  # To hold the extracted bits

    # Create a copy of the stego image to restore the original image
    restored_img = stego_img.copy()

    for y in range(height):
        for x in range(width):
            if map_shifted[y, x] == 1:
                was_shifted = ((shift == 1 and stego_img[y, x] == peak + 1) or
                            (shift == -1 and stego_img[y, x] == peak - 1))
                if was_shifted:
                    restored_img[y, x] -= shift
                    extracted_payload.append(1)
                else:
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
    original_payload_bits = []

    total_embedded_bits = 0  # Track the total number of bits actually embedded

    if saturation > saturation_threshold:
        print("COLOR IMAGE")
        map_shifted_all = np.zeros(origImg.shape[:2], dtype=np.uint8)

        for i in range(3):  # Assuming BGR format
            channel_img = origImg[:, :, i]
            hist = cv2.calcHist([channel_img], [0], None, [256], [0, 256]).flatten()
            peak = int(np.argmax(hist))
            shift = -1 if peak == 255 else 1
            payload_capacity = int(hist[peak])  # Capacity of this channel
            payload_bits = [random.randint(0, 1) for _ in range(payload_capacity)]
            original_payload_bits.extend(payload_bits)

            stego_channel, map_shifted = embed_data_with_histogram_shifting(channel_img, payload_bits)
            total_embedded_bits += np.sum(map_shifted)  # Count the bits actually embedded

            # Update the stego image and combine the map_shifted arrays
            stego_img[:, :, i] = stego_channel
            map_shifted_all = np.bitwise_or(map_shifted_all, map_shifted)

            # Save metadata for each channel
            metadata['peak_values'].append(peak)
            metadata['payload_sizes'].append(len(payload_bits))
            metadata['shifts'].append(shift)

        # Save the combined map for all channels after processing all channels
        np.save(f"{maps_dir}/{imgName}_map_shifted.npy", map_shifted_all)
        save_stego_histogram(stego_img, imgName)  # Call to save histogram of the stego image


    else:
        print("GRAYSCALE IMAGE")
        if origImg.ndim == 3:
            origImg = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([origImg], [0], None, [256], [0, 256]).flatten()
        peak = int(np.argmax(hist))
        shift = -1 if peak == 255 else 1
        payload_capacity = int(hist[peak])
        payload_bits = [random.randint(0, 1) for _ in range(payload_capacity)]
        original_payload_bits = payload_bits

        stego_img, map_shifted = embed_data_with_histogram_shifting(origImg, payload_bits)
        total_embedded_bits += np.sum(map_shifted)  # Count the bits actually embedded

        # Save metadata for grayscale image
        metadata['peak_values'].append(peak)
        metadata['payload_sizes'].append(len(payload_bits))
        metadata['shifts'].append(shift)
        
        # After embedding data into the image
        np.save(f"./HS/maps/{imgName}_map_shifted.npy", map_shifted)
        save_stego_histogram(stego_img, imgName)  # Call to save histogram for grayscale as well

            
    # Calculate metrics for the grayscale image
    psnr_value = cv2.PSNR(origImg, stego_img)
    ssim_score = ssim(origImg, stego_img, win_size=3, data_range=stego_img.max() - stego_img.min())
            
    # Assuming 1 bit per pixel for payload_bits
    bpp = len(payload_bits) / (origImg.size)
          
    print(f"Payload Size (bits): {len(payload_bits)}")
    print(f"Bits Per Pixel (bpp): {bpp}")
    print(f"PSNR: {psnr_value:.2f}")
    print(f"SSIM: {ssim_score:.5f}")

    # Save the stego image
    outcome_path = "./HS/outcome/"
    if not os.path.exists(outcome_path):
        os.makedirs(outcome_path)
    cv2.imwrite(os.path.join(outcome_path, f"{imgName}_marked_histogram.{fileType}"), stego_img)
    print(f"Stego image saved at {outcome_path}{imgName}_marked_histogram.{fileType}")
    
    save_metadata_for_channel(metadata, imgName)
    
    if total_embedded_bits != len(original_payload_bits):
        raise ValueError("Mismatch in total payload size and embedded payload size.")

    return imgName, original_payload_bits


def hs_decoding(imgName, original_payload_bits):
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
    
    extracted_payload_bits = []

    if is_grayscale:
        print("Decoding GRAYSCALE IMAGE")
        extracted_data, restored_img = extract_data_with_histogram_shifting(
            stego_img, map_shifted, peak_values[0], shifts[0]
        )
        extracted_payload_bits.extend(extracted_data)
    else:
        print("Decoding COLOR IMAGE")
        restored_img = stego_img.copy()
        extracted_payload_bits_per_channel = []  # List to store extracted bits per channel

        for ch in range(3):  # Assuming BGR format
            channel_extracted_data, restored_channel = extract_data_with_histogram_shifting(
                stego_img[:, :, ch], map_shifted, peak_values[ch], shifts[ch]
            )
            restored_img[:, :, ch] = restored_channel
            extracted_payload_bits_per_channel.append(channel_extracted_data)
        
        for i in range(len(extracted_payload_bits_per_channel[0])):
            for ch in range(3):
                if i < len(extracted_payload_bits_per_channel[ch]):
                    extracted_payload_bits.append(extracted_payload_bits_per_channel[ch][i])

    # Payload comparison and size validation
    if extracted_payload_bits == original_payload_bits:
        print("Error: The embedded data and the extracted data differ.")
    else:
        print("Success: The embedded data and the extracted data are identical.")
    
    # Calculate PSNR
    psnr_value = cv2.PSNR(original_img, original_img)
    print(f"PSNR: {psnr_value:.2f}")

    # Before SSIM calculation
    min_dimension = min(original_img.shape[0], original_img.shape[1])
    win_size = min(7, min_dimension)
    if win_size % 2 == 0:  # Make sure window size is odd
        win_size = max(win_size - 1, 1)  # Ensure win_size is at least 1

    # Calculate SSIM
    try:
        ssim_value = ssim(
            original_img, 
            original_img, 
            win_size=win_size, 
            multichannel=(original_img.ndim == 3),
            channel_axis=-1 if (original_img.ndim == 3) else None
        )
        print(f"SSIM: {ssim_value:.5f}")
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
        imgName, original_payload_bits = hs_encoding()  # This returns the image name and the payload bits
        post_encode_choice = input("Would you like to continue with decoding? (Y/N): ").upper()
        if post_encode_choice == 'Y':
            hs_decoding(imgName, original_payload_bits)  # Pass the image name and the payload bits for decoding
        else:
            print("Encoding complete. Exiting the program.")

    elif choice == 'D':
        imgName = input("Enter the name of the stego image: ")
        # For standalone decoding, you need a method to provide or calculate original payload bits
        # One possible approach is to read them from a file or user input
        # For example:
        # original_payload_bits = load_original_payload_bits()  # Implement this function as needed
        # hs_decoding(imgName, original_payload_bits)

        # Alternatively, if original payload bits are not needed for standalone decoding, adjust hs_decoding accordingly
        # hs_decoding(imgName)

        print("Note: Standalone decoding currently requires access to original payload bits.")

    else:
        print("Invalid choice. Please restart the program and select either E or D.")

if __name__ == "__main__":
    main()
