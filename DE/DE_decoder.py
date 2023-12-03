import cv2
import numpy as np
from DE_encoder import *  # 從 DE_encoder 模塊導入函數

# 配置路徑
ORIGINAL_IMAGES_PATH = "./DE/images/"
MARKED_IMAGES_PATH = "./DE/marked/"
DECODED_IMAGES_PATH = "./DE/decoded/"
LOCATION_MAP_PATH = "./DE/location_map/"
LSB_PATH = "./DE/lsb/"
EXTRACT_DATA_PATH = "./DE/extract_data/"

def read_image(file_path):
    """讀取影像"""
    return cv2.imread(file_path)

def save_extracted_data_as_text(img_name, extracted_data, is_gray=True):
    suffix = 'grayscale' if is_gray else 'color'
    with open(f"{EXTRACT_DATA_PATH}/{img_name}_extracted_{suffix}.txt", "w") as file:
        if is_gray:
            file.write(extracted_data)
        else:
            for channel in ['R', 'G', 'B']:
                file.write(''.join(map(str, extracted_data[channel])) + "\n")

def compare_extracted_and_embedded_data(img_name, extracted_data, is_gray=True):
    embed_data_path = f"{EMBED_DATA_PATH}/Splash_embed_data.txt" if not is_gray else f"{EMBED_DATA_PATH}/{img_name}_embed_data.txt"
    try:
        with open(embed_data_path, "r") as file:
            embedded_data = file.read()

        if is_gray:
            comparison = extracted_data == embedded_data
        else:
            embedded_data_channels = embedded_data.split('\n')
            comparison = True
            for i, channel in enumerate(['R', 'G', 'B']):
                extracted_channel_data = ''.join(map(str, extracted_data[channel]))
                if extracted_channel_data != embedded_data_channels[i]:
                    comparison = False
                    break

        if comparison:
            print("Success: Extracted data matches embedded data.")
        else:
            print("Error: Extracted data does not match embedded data.")
    except FileNotFoundError:
        print("Embedded data file not found.")

def process_pixel_pair(pixel1, pixel2, location_map_value, lsb_value):
    l, h = integer_transform(pixel1, pixel2)
    
    extracted_bit = None
    if location_map_value:
        extracted_bit = h & 1
        h = (h - extracted_bit) // 2
    else:
        h = (h + lsb_value) // 2
    
    pixel1, pixel2 = inverse_integer_transform(l, h)
    return pixel1, pixel2, extracted_bit

def extract_info_from_grayscale(img, location_map, lsb):
    """从灰度图像中提取信息"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = img.shape
    extracted_bits = []
    for y in range(height):
        for x in range(0, width, 2):
            pixel1 = img[y, x]
            pixel2 = img[y, x + 1] if x + 1 < width else pixel1

            pixel1, pixel2, extracted_bit = process_pixel_pair(pixel1, pixel2, location_map[y, x], lsb[y, x])

            if location_map[y, x]:
                extracted_bits.append(extracted_bit)

            img[y, x], img[y, x + 1 if x + 1 < width else x] = pixel1, pixel2

    extracted_payload_size = len(extracted_bits)  # 計算提取出來的有效載荷大小
    extracted_payload_content = ''.join(map(str, extracted_bits))
    
    # 將提取出來的資料寫入文件
    with open("./DE/debug_info.txt", "w") as file:
        file.write(extracted_payload_content)

    return img, extracted_payload_size, extracted_payload_content

def extract_info_from_color(img, location_maps, lsbs):
    height, width = img.shape[:2]
    extracted_bits = {'R': [], 'G': [], 'B': []}
    missing_pairs = {'R': [], 'G': [], 'B': []}  # 用于记录未能正确提取的像素对位置

    for c in range(3):  # 对于每个颜色通道
        channel = img[:, :, c]
        for y in range(height):
            for x in range(0, width, 2):
                pixel1 = channel[y, x]
                pixel2 = channel[y, x + 1] if x + 1 < width else pixel1

                # 正确地传递单个 location_map 值和 lsb 值
                pixel1, pixel2, extracted_bit = process_pixel_pair(pixel1, pixel2, location_maps[c][y, x], lsbs[c][y, x])

                if location_maps[c][y, x]:
                    if extracted_bit is not None:
                        extracted_bits['RGB'[c]].append(extracted_bit)
                    else:
                        missing_pairs['RGB'[c]].append((y, x))  # 记录未能提取的像素对位置

                channel[y, x], channel[y, x + 1 if x + 1 < width else x] = pixel1, pixel2

        # 调试输出
        print(f"Channel {'RGB'[c]} - Total extracted pairs: {len(extracted_bits['RGB'[c]])}")
        print(f"Channel {'RGB'[c]} - Missing pairs count: {len(missing_pairs['RGB'[c]])}")
        if missing_pairs['RGB'[c]]:
            print(f"Channel {'RGB'[c]} - Missing pairs positions:", missing_pairs['RGB'[c]][:10])  # 只打印前10个缺失位置
    
    # 計算每個通道的提取出來的有效載荷大小並合計
    total_extracted_payload_size = sum(len(extracted_bits[channel]) for channel in extracted_bits)

    return img, total_extracted_payload_size, extracted_bits

def main():
    img_name = input("image name: ")
    file_type = "png"
    marked_img = read_image(f"{MARKED_IMAGES_PATH}{img_name}_markedImg.{file_type}")

    is_gray = is_grayscale(marked_img)

    if is_gray:
        location_map = np.load(f"{LOCATION_MAP_PATH}{img_name}_location_map.npy")
        lsb = np.load(f"{LSB_PATH}{img_name}_original_lsb.npy")
        
        # 修改后的检查
        if lsb.shape != (marked_img.shape[0], marked_img.shape[1]):
            print("Error: The size of the LSB array does not match the expected size.")
            return
                    
        out_img, extracted_payload_size, extracted_payload_content = extract_info_from_grayscale(marked_img, location_map, lsb)
        save_extracted_data_as_text(img_name, extracted_payload_content, is_gray=True)
        compare_extracted_and_embedded_data(img_name, extracted_payload_content, is_gray=True)
        
        print(f"Extracted payload size: {extracted_payload_size} bits")
        print(f"Extracted payload content (first 100 bits): {extracted_payload_content[:100]}...")
    
    else:
        location_maps = [np.load(f"{LOCATION_MAP_PATH}{img_name}_channel_{color}_location_map.npy") for color in ['R', 'G', 'B']]
        lsbs = [np.load(f"{LSB_PATH}{img_name}_channel_{color}_original_lsb.npy") for color in ['R', 'G', 'B']]
        out_img, total_extracted_payload_size, extracted_bits = extract_info_from_color(marked_img, location_maps, lsbs)
        
        save_extracted_data_as_text(img_name, extracted_bits, is_gray=False)
        compare_extracted_and_embedded_data(img_name, extracted_bits, is_gray=False)
        
        print(f"Total extracted payload size: {total_extracted_payload_size} bits")

    # 计算和显示指标
    orig_img = read_image(f"{ORIGINAL_IMAGES_PATH}{img_name}.{file_type}")
    psnr, ssim = calculate_psnr(orig_img, orig_img), calculate_ssim(orig_img, orig_img)
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.3f}")
    
    # Displaying images based on user choice
    display_choice = input("Display images? (y/n): ")
    if display_choice.lower() == 'y':
        cv2.imshow("Original", orig_img)
        cv2.imshow("Decoded Image", out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()