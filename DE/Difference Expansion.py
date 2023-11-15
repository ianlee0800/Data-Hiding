import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def number_to_bits(number, length):
    return np.array([int(bit) for bit in format(number, '0' + str(length) + 'b')])

def integer_transform(x, y):
    """
    Perform integer transform on a pair of pixels.
    :param x, y: Pixel values.
    :return: Transformed components (average and difference).
    """
    x, y = np.int16(x), np.int16(y)  # Convert to larger integer type
    l = (x + y) // 2  # Average
    h = x - y  # Difference
    return l, h

def inverse_integer_transform(l, h):
    """
    Perform inverse integer transform to reconstruct original pixels.
    :param l, h: Transformed components (average and difference).
    :return: Original pixel values.
    """
    x = l + (h + (1 if h >= 0 else -1)) // 2
    y = l - (h // 2)
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

def difference_expansion_embed(x, y, bit, location_map, changeable_map, i, j):
    l, h = integer_transform(x, y)
    embedded = False  # Flag to indicate if a bit was embedded

    if is_changeable(l, h):
        changeable_map[i // 2, j // 2] = 1  # Mark as changeable
        if is_expandable(l, h):
            h_prime = 2 * h + bit
            location_map[i // 2, j // 2] = 1  # Mark as used for embedding
            embedded = True
        else:
            h_prime = 2 * (h // 2) + bit
            embedded = True
    else:
        h_prime = h
        changeable_map[i // 2, j // 2] = 0  # Mark as not changeable

    x_prime, y_prime = inverse_integer_transform(l, h_prime)
    x_prime = np.clip(x_prime, 0, 255)  # Clipping to prevent overflow/underflow
    y_prime = np.clip(y_prime, 0, 255)
    return x_prime, y_prime, embedded


def difference_expansion_extract(x_prime, y_prime, location_map, changeable_map, i, j):
    l_prime, h_prime = integer_transform(x_prime, y_prime)

    # Determine if the current pixel is changeable or expanded based on the maps
    if changeable_map[i, j]:
        # For changeable difference values
        if location_map[i, j]:
            # If expanded, divide by 2 to get the original value
            h = h_prime // 2
        else:
            # For non-expanded but changeable values, restore the original LSB
            h = (h_prime // 2) if h_prime % 2 == 0 else (h_prime // 2 + 1)
    else:
        # For non-changeable values, keep the difference as is
        h = h_prime

    # Calculate the original values x and y using the inverse transform
    x, y = inverse_integer_transform(l_prime, h)
    x = np.clip(x, 0, 255)  # Clipping to ensure values are within valid range
    y = np.clip(y, 0, 255)

    # Extract the bit only if the pixel was used for embedding
    bit = None
    if location_map[i, j]:
        bit = h_prime % 2

    return x, y, bit

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

def process_image_channel(channel_data, payload, bit_index, max_payload_size, location_map, changeable_map):
    for i in range(0, channel_data.shape[0], 2):
        for j in range(0, channel_data.shape[1], 2):
            if bit_index >= len(payload):  # Check if bit_index is out of bounds
                return  # Exit the function as all bits are processed
            if bit_index < max_payload_size:
                bit = payload[bit_index]
                x, y = channel_data[i, j], channel_data[i, j + 1]
                x_prime, y_prime, embedded = difference_expansion_embed(x, y, bit, location_map, changeable_map, i // 2, j // 2)
                if embedded:
                    print(f"Bit {bit} embedded at ({i//2}, {j//2})")
                channel_data[i, j], channel_data[i, j + 1] = x_prime, y_prime
            bit_index += 1

def process_image_channel_for_decoding(channel_data, extracted_payload, location_map, changeable_map):
    for i in range(0, channel_data.shape[0], 2):
        for j in range(0, channel_data.shape[1], 2):
            if location_map[i // 2, j // 2]:  # Check if the pixel was used for embedding
                x_prime, y_prime = channel_data[i, j], channel_data[i, j + 1]
                x, y, bit = difference_expansion_extract(x_prime, y_prime, location_map, changeable_map, i // 2, j // 2)
                if location_map[i // 2, j // 2]:
                    print(f"Bit {bit} extracted from ({i//2}, {j//2})")
                channel_data[i, j], channel_data[i, j + 1] = x, y
                extracted_payload.append(bit)


def perform_encoding():
    imgName = input("Image name: ")
    fileType = "png"

    origImg = cv2.imread(f"./DE/images/{imgName}.{fileType}")
    if origImg is None:
        print("Failed to load the image. Check the file path.")
        exit(1)

    stegoImg = origImg.copy()
    location_map = np.zeros(origImg.shape[:2], dtype=np.uint8)
    changeable_map = np.zeros_like(location_map, dtype=np.uint8)  # Initialize changeable_map
    payload_size = origImg.shape[0] * origImg.shape[1] // 2
    payload = np.random.randint(0, 2, payload_size).astype(np.uint8)
    img_hsv = cv2.cvtColor(stegoImg, cv2.COLOR_BGR2HSV)
    saturation = img_hsv[:, :, 1].mean()
    saturation_threshold = 30

    if saturation > saturation_threshold:  # Color Image
        print("COLOR IMAGE")
        channels = ['B', 'G', 'R']
        payloads = split_payload(payload, len(channels))
        location_maps = [np.zeros(origImg.shape[:2], dtype=np.uint8) for _ in channels]
        changeable_maps = [np.zeros_like(location_map, dtype=np.uint8) for _ in channels]  # Initialize changeable maps for each channel

        for idx, channel in enumerate(channels):
            process_image_channel(
                stegoImg[:, :, idx],
                payloads[idx],
                0,  # Starting bit index
                payload_size,
                location_maps[idx],
                changeable_maps[idx]  # Pass the changeable_map
            )
            np.save(f"./DE/location_map/{imgName}_marked_{channel}_location_map.npy", location_maps[idx])
            np.save(f"./DE/changeable_map/{imgName}_marked_{channel}_changeable_map.npy", changeable_maps[idx])  # Save the changeable_map

    else:  # Grayscale Image
        print("GRAYSCALE IMAGE")
        stegoImg_gray = cv2.cvtColor(stegoImg, cv2.COLOR_BGR2GRAY)
        location_map_gray = np.zeros((stegoImg_gray.shape[0] // 2, stegoImg_gray.shape[1] // 2), dtype=np.uint8)
        changeable_map_gray = np.zeros_like(location_map_gray, dtype=np.uint8)  # Initialize changeable_map for grayscale

        process_image_channel(
            stegoImg_gray,
            payload,
            0,
            payload_size,
            location_map_gray,
            changeable_map_gray  # Pass the changeable_map
        )
        stegoImg = cv2.cvtColor(stegoImg_gray, cv2.COLOR_GRAY2BGR)
        np.save(f"./DE/location_map/{imgName}_marked_location_map.npy", location_map_gray)
        np.save(f"./DE/changeable_map/{imgName}_marked_changeable_map.npy", changeable_map_gray)  # Save the changeable_map

    psnr = cv2.PSNR(origImg, stegoImg)
    ssim_score = calculate_ssim(origImg, stegoImg)

    print("Image Size:", origImg.shape[0], "x", origImg.shape[1])
    print("Payload Size:", payload_size)
    print("Bits Per Pixel (bpp):", payload_size / (origImg.shape[0] * origImg.shape[1] / 4))
    print("PSNR:", psnr)
    print("SSIM:", ssim_score)

    cv2.imwrite(f"./DE/outcome/{imgName}_marked.{fileType}", stegoImg)

    print("Encoding completed.")
    binary_payload = ''.join(str(bit) for bit in payload)
    return imgName, stegoImg, binary_payload

def perform_decoding(imgName, original_payload):
    fileType = "png"
    if not imgName.endswith("_marked"):
        imgName += "_marked"
    stegoImgPath = f"./DE/outcome/{imgName}.{fileType}"
    stegoImg = cv2.imread(stegoImgPath)

    if stegoImg is None:
        print(f"Failed to load the stego image from {stegoImgPath}. Check the file path.")
        exit(1)

    extracted_payload = []  
    restoredImg = np.zeros_like(stegoImg)
    img_hsv = cv2.cvtColor(stegoImg, cv2.COLOR_BGR2HSV)
    saturation = img_hsv[:, :, 1].mean()
    saturation_threshold = 30

    if saturation > saturation_threshold:  # Color Image
        print("COLOR IMAGE")
        channels = ['B', 'G', 'R']
        for idx, channel in enumerate(channels):
            location_map_path = f"./DE/location_map/{imgName}_{channel}_location_map.npy"
            changeable_map_path = f"./DE/changeable_map/{imgName}_{channel}_changeable_map.npy"  # Path for changeable map

            if not os.path.exists(location_map_path) or not os.path.exists(changeable_map_path):
                print(f"Location map or changeable map for {channel} channel not found.")
                exit(1)

            location_map = np.load(location_map_path)
            changeable_map = np.load(changeable_map_path)  # Load the changeable map
            channel_data = stegoImg[:, :, idx].copy()
            process_image_channel_for_decoding(channel_data, extracted_payload, location_map, changeable_map)
            restoredImg[:, :, idx] = channel_data

    else:  # Grayscale Image
        print("GRAYSCALE IMAGE")
        location_map_path = f"./DE/location_map/{imgName}_location_map.npy"
        changeable_map_path = f"./DE/changeable_map/{imgName}_changeable_map.npy"  # Path for changeable map

        if not os.path.exists(location_map_path) or not os.path.exists(changeable_map_path):
            print("Location map or changeable map not found.")
            exit(1)

        location_map = np.load(location_map_path)
        changeable_map = np.load(changeable_map_path)  # Load the changeable map
        process_image_channel_for_decoding(stegoImg[:, :, 0], extracted_payload, location_map, changeable_map)
        restoredImg_gray = stegoImg[:, :, 0].copy()
        restoredImg = cv2.cvtColor(restoredImg_gray, cv2.COLOR_GRAY2BGR)

    # Ask the user for the original image for comparison
    origin_imgName = input("Enter the name of the original file for comparison (from ./DE/images): ")
    origin_img = cv2.imread(f"./DE/images/{origin_imgName}.png")
    
    if origin_img is None:
        print("Failed to load the original image. Check the file path.")
        exit(1)  # Exit the program if the original image could not be loaded
    
    # Convert the extracted payload to a binary string
    extracted_binary_payload = ''.join(str(bit) if bit is not None else 'None' for bit in extracted_payload)
    print(f"Extracted binary payload: {extracted_binary_payload}")
    
    # Compare with the original payload
    if extracted_binary_payload == original_payload:
        print("Success: Extracted payload matches the original payload.")
    else:
        print("Error: Extracted payload does not match the original payload.")
    
    # Calculate PSNR and SSIM between the original and restored images
    psnr = cv2.PSNR(origin_img, restoredImg)
    ssim_score = calculate_ssim(origin_img, restoredImg)

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
    return extracted_binary_payload

def main():
    choice = input("Would you like to encode or decode? (E/D): ").upper()

    if choice == 'E':
        imgName, stegoImg, original_payload = perform_encoding()
        post_encode_choice = input("Would you like to continue with decoding? (Y/N): ").upper()

        if post_encode_choice == 'Y':
            perform_decoding(imgName, original_payload)
        else:
            print("Terminating the program.")
            exit(0)

    elif choice == 'D':
        imgName = input("Enter the name of the file to decode (from ./DE/outcome): ")
        # Here you need to provide the original payload for the decoding process
        # Assuming you have a way to retrieve or input the original payload
        original_payload = input("Enter the original payload for comparison: ")
        perform_decoding(imgName, original_payload)

if __name__ == '__main__':
    main()
