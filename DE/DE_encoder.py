import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

# Constants
FILE_TYPE = "png"
DE_IMAGES_PATH = "./DE/images/"
DE_MARKED_PATH = "./DE/marked/"
LOCATION_MAP_PATH = "./DE/location_map/"
LSB_PATH = "./DE/lsb/"

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

def is_expandable(h, l):
    # 確保 h 和 l 是單個數字
    return abs(2 * h + 1) <= min(2 * (255 - l), 2 * l + 1) and \
           abs(2 * h) <= min(2 * (255 - l), 2 * l + 1)

def is_changeable(h, l):
    # 確保 h 和 l 是單個數字
    return abs(2 * (h // 2) + 1) <= min(2 * (255 - l), 2 * l + 1) and \
           abs(2 * (h // 2)) <= min(2 * (255 - l), 2 * l + 1)

def hide_data_in_grayscale_channel(channel, data, location_map, lsb, debug_file_path):
    with open(debug_file_path, 'w') as debug_file:
        if len(channel.shape) == 3:
            channel = cv2.cvtColor(channel, cv2.COLOR_BGR2GRAY)

        height, width = channel.shape
        i = 0
        for y in range(0, height, 2):
            for x in range(0, width - 1, 2):
                if i >= len(data):
                    break

                pixel1 = channel[y, x]
                pixel2 = channel[y, x + 1]
                l, h = integer_transform(pixel1, pixel2)

                # 检查是否可扩展
                is_expandable = abs(2 * h + data[i]) <= min(2 * (255 - l), 2 * l + 1)
                
                if is_expandable:
                    location_map.append(1)  # 标记为用于嵌入
                    h = 2 * h + data[i]
                    i += 1
                else:
                    location_map.append(0)  # 标记为未使用

                x_new, y_new = inverse_integer_transform(l, h)
                channel[y, x], channel[y, x + 1] = x_new, y_new

        # 更新Payload size
        updated_payload_size = sum(location_map)
        return channel, updated_payload_size

def hide_data_in_color_channel(channel, data, location_map, lsb, debug_file_path):
    with open(debug_file_path, 'w') as debug_file:
        # 单个颜色通道的图像，所以只有两个维度
        height, width = channel.shape
        i = 0
        for y in range(0, height, 2):
            for x in range(0, width - 1, 2):
                if i >= len(data):
                    break

                pixel1 = channel[y, x]
                pixel2 = channel[y, x + 1]
                l, h = integer_transform(pixel1, pixel2)
                debug_file.write(f"Pixel1: {pixel1}, Pixel2: {pixel2}, l: {l}, h: {h}\n")

                original_lsb = pixel1 % 2
                lsb.append(original_lsb)

                if is_expandable(h, l):
                    location_map.append(1)  # 标记为用于嵌入
                    h = 2 * h + data[i]
                    i += 1
                else:
                    location_map.append(0)  # 标记为未使用

                channel[y, x], channel[y, x + 1] = inverse_integer_transform(l, h)
        
        updated_payload_size = sum(location_map)
        return channel, updated_payload_size

def calculate_max_payload_size(image):
    height, width = image.shape[:2]
    return height * width // 2  # Assuming one bit per pixel pair

def main():
    img_name = input("Image name: ")
    orig_img = cv2.imread(DE_IMAGES_PATH + f"{img_name}.{FILE_TYPE}")

    if orig_img is None:
        print("Error reading original image file")
        exit()

    grayscale = is_grayscale(orig_img)
    max_payload_size = calculate_max_payload_size(orig_img)
    data_to_hide = [1] * max_payload_size
    location_map = []
    original_lsb = []

    debug_file_path = "./DE/debug_info.txt"  # 调试信息文件路径
    channel_names = ['R', 'G', 'B']  # 用于命名文件的通道名

    if grayscale:
        print(f"{img_name} is a Grayscale image")
        marked_img, updated_payload_size = hide_data_in_grayscale_channel(orig_img.copy(), data_to_hide, location_map, original_lsb, debug_file_path)
    else:
        print(f"{img_name} is a Color image")
        marked_img = orig_img.copy()
        location_maps = [[], [], []]
        original_lsbs = [[], [], []]
        total_payload_size = 0  # 用于累加彩色图像的payload size

        for c in range(3):
            channel = marked_img[:, :, c]
            channel_location_map = []
            channel_lsb = []
            marked_channel, channel_payload_size = hide_data_in_color_channel(channel, data_to_hide, channel_location_map, channel_lsb, debug_file_path)
            marked_img[:, :, c] = marked_channel
            location_maps[c] = channel_location_map
            original_lsbs[c] = channel_lsb
            total_payload_size += channel_payload_size

        updated_payload_size = total_payload_size  # 更新总payload size

    # Save location map and original LSBs
    create_directory_if_not_exists(LOCATION_MAP_PATH)
    create_directory_if_not_exists(LSB_PATH)
    if grayscale:
        np.save(LOCATION_MAP_PATH + f"{img_name}_location_map.npy", location_map)
        np.save(LSB_PATH + f"{img_name}_original_lsb.npy", original_lsb)
    else:
        for c in range(3):
            np.save(LOCATION_MAP_PATH + f"{img_name}_channel_{channel_names[c]}_location_map.npy", location_maps[c])
            np.save(LSB_PATH + f"{img_name}_channel_{channel_names[c]}_original_lsb.npy", original_lsbs[c])

    cv2.imwrite(DE_MARKED_PATH + f"{img_name}_markedImg.{FILE_TYPE}", marked_img)
    
    # Calculate and display metrics
    print(f"Payload size: {updated_payload_size}")
    print(f"Payload content: {''.join(map(str, data_to_hide[:50]))}...")  # Print first 50 bits for brevity

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


