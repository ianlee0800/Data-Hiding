import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def number_to_bits(number, length):
    return np.array([int(bit) for bit in format(number, '0' + str(length) + 'b')])

def integer_transform(x, y):
    x, y = np.int16(x), np.int16(y)  # Convert to larger integer type
    l = (x + y) // 2
    h = x - y
    return l, h

def inverse_integer_transform(l, h):
    x = l + (h + 1) // 2
    y = l - h // 2
    x = np.clip(x, 0, 255)  # Clipping to the valid range
    y = np.clip(y, 0, 255)
    return x, y

def is_expandable(l, h):
    return abs(2 * h + 0) <= min(2 * (255 - l), 2 * l + 1) and abs(2 * h + 1) <= min(2 * (255 - l), 2 * l + 1)

def is_changeable(l, h):
    return abs(2 * (h // 2) + 0) <= min(2 * (255 - l), 2 * l + 1) and abs(2 * (h // 2) + 1) <= min(2 * (255 - l), 2 * l + 1)

def split_payload(payload, num_channels):
    """
    Splits the payload into equal parts for each channel.
    :param payload: The original payload to be split.
    :param num_channels: The number of channels to split the payload into.
    :return: A list of payloads, one for each channel.
    """
    # Calculate the size of each split payload
    split_size = len(payload) // num_channels
    
    # Split the payload using numpy array_split which allows for uneven splits if necessary
    split_payloads = np.array_split(payload, num_channels)
    
    return split_payloads

def difference_expansion_embed(x, y, bit, location_map, i, j):
    l, h = integer_transform(x, y)
    embedded = False  # Flag to indicate if a bit was embedded
    if i < location_map.shape[0] and j < location_map.shape[1]:
        if is_expandable(l, h):
            h_prime = 2 * h + bit
            location_map[i, j] = 1
            embedded = h_prime != h  # Update flag based on change
        elif is_changeable(l, h):
            h_prime = 2 * (h // 2) + bit
            embedded = h_prime != h
        else:
            h_prime = h
        x_prime, y_prime = inverse_integer_transform(l, h_prime)
        return x_prime, y_prime, embedded
    else:
        return x, y, embedded

def difference_expansion_extract(x_prime, y_prime, location_map, i, j):
    if location_map[i, j]:
        # Only proceed with extraction if the location map indicates embedding
        l_prime = (x_prime + y_prime) // 2
        h_prime = x_prime - y_prime
        bit = h_prime % 2  # Extract LSB
        h = h_prime // 2
        x = l_prime + (h + 1) // 2
        y = l_prime - h // 2
        return x, y, bit
    else:
        # If not used for embedding, return the original values
        return x_prime, y_prime, None

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

def split_payload(payload, num_channels):
    split_size = len(payload) // num_channels
    return [payload[i * split_size:(i + 1) * split_size] for i in range(num_channels)]

def process_image_channel(channel_data, payload, bit_index, max_payload_size, location_map):
    for i in range(0, channel_data.shape[0], 2):
        for j in range(0, channel_data.shape[1], 2):
            if bit_index >= len(payload):  # Check if bit_index is out of bounds
                return  # Exit the function as all bits are processed
            if bit_index < max_payload_size:
                bit = payload[bit_index]
                x, y = channel_data[i, j], channel_data[i, j+1]
                x_prime, y_prime, _ = difference_expansion_embed(x, y, bit, location_map, i//2, j//2)  # Unpack three values
                channel_data[i, j], channel_data[i, j+1] = x_prime, y_prime
            bit_index += 1

def process_image_channel_for_decoding(channel_data, extracted_payload, location_map, idx):
    for i in range(0, channel_data.shape[0], 2):
        for j in range(0, channel_data.shape[1], 2):
            if location_map[i//2, j//2]:  # Check if the pixel was used for embedding
                x_prime, y_prime = channel_data[i, j], channel_data[i, j+1]
                x, y, bit = difference_expansion_extract(x_prime, y_prime, location_map, i, j)
                channel_data[i, j], channel_data[i, j+1] = x, y
                extracted_payload.append(bit)

def perform_encoding():
    # Put your existing encoding code here
    imgName = input("Image name: ")
    fileType = "png"

    origImg = cv2.imread(f"./DE/images/{imgName}.{fileType}")

    if origImg is None:
        print("Failed to load the image. Check the file path.")
        exit(1)  # Exit the program

    # Initialize variables
    stegoImg = origImg.copy()
    location_map = np.zeros(origImg.shape[:2], dtype=np.uint8)
    original_lsbs = []
    bit_index = 0
    actual_payload_size = 0
    max_payload_size = origImg.shape[0] * origImg.shape[1] // 2
    payload_size = max_payload_size
    payload = np.random.randint(0, 2, payload_size).astype(np.uint8)
    img_hsv = cv2.cvtColor(stegoImg, cv2.COLOR_BGR2HSV)
    saturation = img_hsv[:, :, 1].mean()
    saturation_threshold = 30

    # Inside perform_encoding, under the COLOR IMAGE section
    # Inside perform_encoding, under the COLOR IMAGE section
    if saturation > saturation_threshold:  # Color Image
        print("COLOR IMAGE")
        channels = ['B', 'G', 'R']
        # Split the payload equally among channels
        payloads = split_payload(payload, len(channels))
        # Initialize variables needed for process_image_channel function
        max_payload_size = origImg.shape[0] * origImg.shape[1] // 2
        location_maps = [np.zeros(origImg.shape[:2], dtype=np.uint8) for _ in channels]
        original_lsbs = [[] for _ in channels]  # List to store LSBs for each channel
        bit_indexes = [0] * len(channels)  # Starting index for each payload

        for idx, channel in enumerate(channels):
            # Call the process_image_channel function with adjusted arguments
            process_image_channel(
                stegoImg[:, :, idx],
                payloads[idx],
                bit_indexes[idx],
                max_payload_size,
                location_maps[idx]
            )
            # Save location map for this channel
            np.save(f"./DE/location_map/{imgName}_marked_{channel}_location_map.npy", location_maps[idx])

    else:  # Grayscale Image
        print("GRAYSCALE IMAGE")
        stegoImg_gray = cv2.cvtColor(stegoImg, cv2.COLOR_BGR2GRAY)
        location_map_gray = np.zeros((stegoImg_gray.shape[0]//2, stegoImg_gray.shape[1]//2), dtype=np.uint8)

        # Call the process_image_channel function for grayscale image with adjusted arguments
        process_image_channel(stegoImg_gray, payload, bit_index, max_payload_size, location_map_gray)
        stegoImg = cv2.cvtColor(stegoImg_gray, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for consistent PSNR and SSIM

    # Calculate PSNR and SSIM using the stego image
    psnr = cv2.PSNR(origImg, stegoImg)
    ssim_score = calculate_ssim(origImg, stegoImg)  
    
    # Use the length of the payload array for calculations
    payload_size = len(payload)
    bpp = payload_size / (origImg.shape[0] * origImg.shape[1] / 4)  # Bits per pixel calculation

    print("Image Size:", origImg.shape[0], "x", origImg.shape[1])
    print("Payload Size:", payload_size)
    print("Bits Per Pixel (bpp):", bpp)
    print("PSNR:", psnr)
    print("SSIM:", ssim_score)

    # Save the stego-image in ./DE/outcome
    cv2.imwrite(f"./DE/outcome/{imgName}_marked.{fileType}", stegoImg)
    np.save(f"./DE/location_map/{imgName}_marked_location_map.npy", location_map)

    print("Encoding completed.")
    return imgName, stegoImg  # return image name and stego image

def perform_decoding(imgName):
    fileType = "png"
    # If 'imgName' already ends with '_marked', do not append another '_marked'
    if not imgName.endswith("_marked"):
        imgName += "_marked"
    stegoImgPath = f"./DE/outcome/{imgName}.{fileType}"
    stegoImg = cv2.imread(stegoImgPath)

    if stegoImg is None:
        print(f"Failed to load the stego image from {stegoImgPath}. Check the file path.")
        exit(1)

    extracted_payload = []  
    restoredImg = np.zeros_like(stegoImg)
    
    # Assuming the existence of a cv2.COLOR_BGR2HSV operation and a threshold to determine color or grayscale
    img_hsv = cv2.cvtColor(stegoImg, cv2.COLOR_BGR2HSV)
    saturation = img_hsv[:, :, 1].mean()
    saturation_threshold = 30

    if saturation > saturation_threshold:  # Color Image
        print("COLOR IMAGE")
        channels = ['B', 'G', 'R']
        for idx, channel in enumerate(channels):
            location_map_path = f"./DE/location_map/{imgName}_{channel}_location_map.npy"
            if not os.path.exists(location_map_path):
                print(f"Location map for {channel} channel not found at {location_map_path}.")
                exit(1)  # Exit the program if the location map could not be loaded
            
            location_map = np.load(location_map_path)
            # Create a copy of the channel data to modify
            channel_data = stegoImg[:, :, idx].copy()
            process_image_channel_for_decoding(channel_data, extracted_payload, location_map, idx)
            restoredImg[:, :, idx] = channel_data  # Update with the restored channel data
            
        pass

    else:  # Grayscale Image
        print("GRAYSCALE IMAGE")
        location_map_path = f"./DE/location_map/{imgName}_location_map.npy"
        if not os.path.exists(location_map_path):
            print(f"Location map not found at {location_map_path}.")
            exit(1)  # Exit the program if the location map could not be loaded

        location_map = np.load(location_map_path)
        process_image_channel_for_decoding(stegoImg[:, :, 0], extracted_payload, location_map, 0)
        restoredImg_gray = stegoImg[:, :, 0].copy()  # Create a copy of the processed grayscale channel
        restoredImg = cv2.cvtColor(restoredImg_gray, cv2.COLOR_GRAY2BGR)  # Convert back to BGR

    # Ask the user for the original image for comparison
    origImgName = input("Enter the name of the original file for comparison (from ./DE/images): ")
    origImg = cv2.imread(f"./DE/images/{origImgName}.png")
    
    if origImg is None:
        print("Failed to load the original image. Check the file path.")
        exit(1)  # Exit the program if the original image could not be loaded
    
    # The extracted_payload should now be filled with bits
    binary_payload = ''.join(str(bit) for bit in extracted_payload)
    #print(f"Extracted binary payload: {binary_payload}")
    
    # Calculate PSNR and SSIM between the original and restored images
    psnr = cv2.PSNR(origImg, restoredImg)
    ssim_score = calculate_ssim(origImg, restoredImg)

    print("PSNR:", psnr)
    print("SSIM:", ssim_score)
    
    # Save the restored image to the './DE/restored' directory
    restored_img_dir = "./DE/restored"
    restored_img_path = os.path.join(restored_img_dir, f"{imgName}_restored.{fileType}")

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(restored_img_dir):
        os.makedirs(restored_img_dir)

    # Save the restored image
    cv2.imwrite(restored_img_path, restoredImg)
    print(f"Restored image saved to {restored_img_path}")

    # Optionally, return the binary payload
    return binary_payload


def main():
    choice = input("Would you like to encode or decode? (E/D): ").upper()

    if choice == 'E':
        imgName, stegoImg = perform_encoding()
        post_encode_choice = input("Would you like to continue with decoding? (Y/N): ").upper()

        if post_encode_choice == 'Y':
            perform_decoding(imgName)
        else:
            print("Terminating the program.")
            exit(0)

    elif choice == 'D':
        imgName = input("Enter the name of the file to decode (from ./DE/outcome): ")
        perform_decoding(imgName)

    else:
        print("Invalid choice. Terminating the program.")
        exit(0)

if __name__ == '__main__':
    main()
