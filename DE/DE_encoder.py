import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import random

# Constants
channel_names = ['R', 'G', 'B']
FILE_TYPE = "png"
DE_IMAGES_PATH = "./DE/images/"
DE_MARKED_PATH = "./DE/marked/"
LOCATION_MAP_PATH = "./DE/location_map/"
LSB_PATH = "./DE/lsb/"
EMBED_DATA_PATH = "./DE/embed_data"

def save_embedded_data_as_text(img_name, data_to_hide, embedded_payload_size):
    with open(f"{EMBED_DATA_PATH}/{img_name}_embed_data.txt", "w") as file:
        file.write(''.join(map(str, data_to_hide[:embedded_payload_size])))

def save_embedded_data_as_text_color(img_name, data_to_hide, channel_payload_sizes):
    with open(f"{EMBED_DATA_PATH}/{img_name}_embed_data.txt", "w") as file:
        for c in range(3):
            payload_size = int(channel_payload_sizes[c])  # 确保为整数
            channel_data = data_to_hide[:payload_size]
            file.write(''.join(map(str, channel_data)))
            if c < 2:  # 在通道数据之间添加分隔符（如果需要）
                file.write("\n")

def calculate_psnr(original, marked):
    if len(original.shape) == 3 and len(marked.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    mse = np.mean((original - marked) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255 / np.sqrt(mse))

def calculate_ssim(original, marked):
    # 确定窗口大小
    win_size = min(3, original.shape[0], original.shape[1])  # 保证窗口大小不超过图像尺寸

    # 确保两个图像的维度一致
    if len(original.shape) == 3 and len(marked.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    elif len(original.shape) == 2 and len(marked.shape) == 3:
        marked = cv2.cvtColor(marked, cv2.COLOR_BGR2GRAY)

    # 计算SSIM
    return ssim(original, marked, win_size=win_size)

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def is_grayscale(image):
    if len(image.shape) < 3:
        return True
    elif image.shape[2] == 1:
        return True
    else:
        # Calculate saturation and determine if it's grayscale
        saturation = np.max(image, axis=2) - np.min(image, axis=2)
        return np.all(saturation < 10)  # Threshold for grayscale

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
    x = l + (h + 1) // 2
    y = l - (h // 2)
    x = np.clip(x, 0, 255)  # Clipping to the valid range
    y = np.clip(y, 0, 255)
    return x, y

def embed_data_in_difference(h, bit):
    if bit == 1:
        h = 2 * h + 1
    else:
        h = 2 * h
    return h

def generate_random_payload(payload_size, ratio_of_ones):
    num_ones = int(payload_size * ratio_of_ones)
    num_zeros = payload_size - num_ones

    payload = [1] * num_ones + [0] * num_zeros
    random.shuffle(payload)
    return payload

def is_expandable(h, l):
    if h == 0 or h == 1:
        return False
    else:
        return abs(2 * h + 1) <= min(2 * (255 - l), 2 * l + 1) and \
           abs(2 * h) <= min(2 * (255 - l), 2 * l + 1)

def is_changeable(h, l):
    # 確保 h 和 l 是單個數字
    return abs(2 * (h // 2) + 1) <= min(2 * (255 - l), 2 * l + 1) and \
           abs(2 * (h // 2)) <= min(2 * (255 - l), 2 * l + 1)

def hide_data_in_grayscale_channel(channel, data, location_map, original_lsb):
    if len(channel.shape) == 3:
        channel = cv2.cvtColor(channel, cv2.COLOR_BGR2GRAY)

    height, width = channel.shape
    i = 0
    for y in range(height):
        for x in range(0, width, 2):
            if i >= len(data):
                break

            pixel1 = channel[y, x]
            pixel2 = channel[y, x + 1] if x + 1 < width else pixel1  # 处理最后一列像素

            l, h = integer_transform(pixel1, pixel2)
            original_lsb[y, x] = pixel1 % 2

            if is_expandable(h, l):
                location_map[y, x] = 1  # 标记为用于嵌入
                h = embed_data_in_difference(h, data[i])
                i += 1
            else:
                location_map[y, x] = 0  # 标记为未使用

            x_new, y_new = inverse_integer_transform(l, h)
            channel[y, x] = x_new
            if x + 1 < width:
                channel[y, x + 1] = y_new

    updated_payload_size = np.sum(location_map)
    return channel, updated_payload_size

def hide_data_in_color_channel(channel, data, location_map, original_lsb):
    height, width = channel.shape
    i = 0
    for y in range(height):
        for x in range(0, width, 2):
            if i >= len(data):
                break

            pixel1 = channel[y, x]
            pixel2 = channel[y, x + 1] if x + 1 < width else pixel1  # 处理最后一列像素

            l, h = integer_transform(pixel1, pixel2)
            original_lsb[y, x] = pixel1 % 2

            if is_expandable(h, l):
                location_map[y, x] = 1  # 标记为用于嵌入
                h = embed_data_in_difference(h, data[i])
                i += 1
            else:
                location_map[y, x] = 0  # 标记为未使用

            x_new, y_new = inverse_integer_transform(l, h)
            channel[y, x] = x_new
            if x + 1 < width:
                channel[y, x + 1] = y_new

    updated_payload_size = np.sum(location_map)
    return channel, updated_payload_size

def calculate_max_payload_size(image):
    height, width = image.shape[:2]
    return height * width // 2  # Assuming one bit per pixel pair

def main():
    img_name = input("Image name: ")
    orig_img = cv2.imread(DE_IMAGES_PATH + f"{img_name}.{FILE_TYPE}")

    if orig_img is None:
        print("Error reading original image file")
        return
    
    height, width = orig_img.shape[:2]
    grayscale = is_grayscale(orig_img)
    max_payload_size = calculate_max_payload_size(orig_img)

    # 生成随机有效载荷数据
    ratio_of_ones = 0.5  # 这里可以根据需要调整 1 的比例，舉例來說 ratio_of_ones = 0.5 表示 50% 的 1 和 50% 的 0
    data_to_hide = generate_random_payload(max_payload_size, ratio_of_ones)

    location_map = np.zeros((height, width), dtype=int)
    original_lsb = np.zeros((height, width), dtype=int)

    if grayscale:
        print(f"{img_name} is a Grayscale image")
        marked_img, updated_payload_size = hide_data_in_grayscale_channel(orig_img.copy(), data_to_hide, location_map, original_lsb)
        save_embedded_data_as_text(img_name, data_to_hide, updated_payload_size)        
        np.save(LOCATION_MAP_PATH + f"{img_name}_location_map.npy", location_map)
        np.save(LSB_PATH + f"{img_name}_original_lsb.npy", original_lsb)

    else:
        print(f"{img_name} is a Color image")
        marked_img = orig_img.copy()
        location_maps = [np.zeros((height, width)) for _ in range(3)]
        original_lsbs = [np.zeros((height, width)) for _ in range(3)]

        channel_payload_sizes = []
        for c in range(3):
            channel = marked_img[:, :, c]
            marked_channel, channel_payload_size = hide_data_in_color_channel(channel, data_to_hide, location_maps[c], original_lsbs[c])
            marked_img[:, :, c] = marked_channel
            channel_payload_sizes.append(channel_payload_size)

        save_embedded_data_as_text_color(img_name, data_to_hide, channel_payload_sizes)
        
        updated_payload_size = sum(channel_payload_sizes)
        for c in range(3):
            np.save(LOCATION_MAP_PATH + f"{img_name}_channel_{channel_names[c]}_location_map.npy", location_maps[c])
            np.save(LSB_PATH + f"{img_name}_channel_{channel_names[c]}_original_lsb.npy", original_lsbs[c])

    cv2.imwrite(DE_MARKED_PATH + f"{img_name}_markedImg.{FILE_TYPE}", marked_img)
    
    # Calculate and display metrics
    print(f"Embedded Payload size: {updated_payload_size}")
    print(f"Payload content: {''.join(map(str, data_to_hide[:50]))}...")

    bpp = updated_payload_size / (orig_img.shape[0] * orig_img.shape[1])
    print(f"Bits per pixel (bpp): {bpp:.4f}")
    
    psnr_value = calculate_psnr(orig_img, marked_img)
    print(f"PSNR: {psnr_value:.2f} dB")

    ssim_value = calculate_ssim(orig_img, marked_img)
    print(f"SSIM: {ssim_value:.5f}")

    # Displaying images based on user choice
    display_choice = input("Display images? (y/n): ")
    if display_choice.lower() == 'y':
        cv2.imshow("Original", orig_img)
        cv2.imshow("Marked", marked_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

