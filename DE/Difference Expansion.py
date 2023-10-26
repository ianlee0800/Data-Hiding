import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import random

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
    return x, y

def is_expandable(l, h):
    return abs(2 * h + 0) <= min(2 * (255 - l), 2 * l + 1) and abs(2 * h + 1) <= min(2 * (255 - l), 2 * l + 1)

def is_changeable(l, h):
    return abs(2 * (h // 2) + 0) <= min(2 * (255 - l), 2 * l + 1) and abs(2 * (h // 2) + 1) <= min(2 * (255 - l), 2 * l + 1)

def difference_expansion_embed(x, y, bit, location_map, i, j):
    l, h = integer_transform(x, y)
    if i < location_map.shape[0] and j < location_map.shape[1]:  # 檢查索引是否在有效範圍內
        if is_expandable(l, h):
            h_prime = 2 * h + bit
            location_map[i, j] = 1  # 更新位置地圖
        elif is_changeable(l, h):
            h_prime = 2 * (h // 2) + bit
        else:
            h_prime = h
        x_prime, y_prime = inverse_integer_transform(l, h_prime)
        return x_prime, y_prime
    else:
        print(f"Warning: Index ({i}, {j}) is out of bounds.")
        return x, y  # 返回原始值

def difference_expansion_extract(x_prime, y_prime):
    # Calculate integer average 'l_prime' and new difference 'h_prime'
    l_prime = (x_prime + y_prime) // 2
    h_prime = x_prime - y_prime
    # Extract the least significant bit (LSB) from 'h_prime'
    bit = h_prime % 2
    # Calculate the original difference value 'h'
    h = h_prime // 2
    # Calculate the original values x and y
    x = l_prime + (h + 1) // 2
    y = l_prime - h // 2
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

def process_image_channel(channel_data, payload, bit_index, max_payload_size, location_map, original_lsbs, actual_payload_size):
    # Debug: Counters to keep track of the number of times conditions are met or not met
    count_expandable = 0
    count_not_expandable = 0
    for i in range(0, channel_data.shape[0], 2):
        for j in range(0, channel_data.shape[1], 2):
            if bit_index >= len(payload):  # Check if bit_index is out of bounds
                break
            if bit_index < max_payload_size:
                count_expandable += 1  # Debug: Update counter
                bit = payload[bit_index]
                x, y = channel_data[i, j], channel_data[i, j+1]
                x_prime, y_prime = difference_expansion_embed(x, y, bit, location_map, i//2, j//2)
                channel_data[i, j], channel_data[i, j+1] = x_prime, y_prime
                bit_index += 1
                actual_payload_size += 1
            else:
                l, h = integer_transform(channel_data[i, j], channel_data[i, j+1])
                if not is_expandable(l, h):
                    count_not_expandable += 1  # Debug: Update counter
                    original_lsbs.append(h % 2)

    print("Debug: count_expandable =", count_expandable)  # Debug: Print counter
    print("Debug: count_not_expandable =", count_not_expandable)  # Debug: Print counter
    return bit_index, actual_payload_size

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
min_payload_size = 0
payload_size = random.randint(min_payload_size, max_payload_size)
payload = np.random.randint(0, 2, payload_size).astype(np.uint8)
img_hsv = cv2.cvtColor(stegoImg, cv2.COLOR_BGR2HSV)
saturation = img_hsv[:, :, 1].mean()
saturation_threshold = 30

if saturation > saturation_threshold:  # Color Image
    print("COLOR IMAGE")
    channels = ['B', 'G', 'R']
    for idx, color in enumerate(channels):
        bit_index, actual_payload_size = process_image_channel(origImg[:, :, idx], payload, bit_index, max_payload_size, location_map, original_lsbs, actual_payload_size)


else:  # Grayscale Image
    print("GRAYSCALE IMAGE")
    stegoImg_gray = cv2.cvtColor(stegoImg, cv2.COLOR_BGR2GRAY)
    location_map_gray = np.zeros((stegoImg_gray.shape[0]//2, stegoImg_gray.shape[1]//2), dtype=np.uint8)
    original_lsbs_gray = []
    bit_index, actual_payload_size = process_image_channel(stegoImg_gray, payload, bit_index, max_payload_size, location_map_gray, original_lsbs_gray, actual_payload_size)
    stegoImg = cv2.cvtColor(stegoImg_gray, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for consistent PSNR and SSIM

# Calculate PSNR and SSIM using the stego image
psnr = cv2.PSNR(origImg, stegoImg)
ssim_score = calculate_ssim(origImg, stegoImg)
payload_size = actual_payload_size
bpp = payload_size / (origImg.shape[0] * origImg.shape[1] / 4)   
   
print("Image Size:", origImg.shape[0], "x" , origImg.shape[1])
print("Payload Size:", payload_size)
print("Bits Per Pixel (bpp):", bpp)
print("PSNR:", psnr)
print("SSIM:", ssim_score)

# Save the stego-image in ./DE/outcome
cv2.imwrite(f"./DE/outcome/{imgName}_marked.{fileType}", stegoImg)
