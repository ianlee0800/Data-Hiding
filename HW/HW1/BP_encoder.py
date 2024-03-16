import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Constants
FILE_TYPE = "tiff"
LEVELS = 256
HISTOGRAM_PATH = "./HW/HW1/bitplane_histogram/"
HS_IMAGES_PATH = "./HW/HW1/bitplanes/"
HS_MARKED_PATH = "./HW/HW1/bitplane_shifted/"
PEAK_PATH = "./HW/HW1/peak/"
HS_HIDE_DATA_PATH = "./HW/HW1/hide_data/"

def draw_histogram(img_name, img, save_path):
    height, width, _ = img.shape
    q = 256
    count = [0] * q
    for y in range(height):
        for x in range(width):
            bgr = int(img[y, x, 0])
            count[bgr] += 1
    plt.figure()
    plt.xlabel("Pixel Intensity")  # 添加X軸標籤
    plt.ylabel("Pixel Count")  # 添加Y軸標籤
    plt.bar(range(1, 257), count)
    plt.savefig(save_path + f"{img_name}_histogram.png")
    plt.close()

def calculate_psnr(img1, img2, max_pixel=255):
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")

    height, width, channels = img1.shape
    size = height * width

    # Constants for SSIM calculation
    C1 = (255 * 0.01) ** 2
    C2 = (255 * 0.03) ** 2
    C3 = C2 / 2

    ssim_total = 0

    for channel in range(channels):
        img1_c = img1[:, :, channel]
        img2_c = img2[:, :, channel]

        mean1 = np.mean(img1_c)
        mean2 = np.mean(img2_c)
        var1 = np.var(img1_c)
        var2 = np.var(img2_c)
        covar = np.cov(img1_c.flatten(), img2_c.flatten())[0][1]

        luminance = (2 * mean1 * mean2 + C1) / (mean1 ** 2 + mean2 ** 2 + C1)
        contrast = (2 * np.sqrt(var1) * np.sqrt(var2) + C2) / (var1 + var2 + C2)
        structure = (covar + C3) / (np.sqrt(var1) * np.sqrt(var2) + C3)

        ssim_channel = luminance * contrast * structure
        ssim_total += ssim_channel

    return ssim_total / channels

def count_levels(img, levels=256):
    level_counts = np.zeros(levels, dtype=int)
    for channel in range(img.shape[2]):
        channel_data = img[:, :, channel].flatten()
        level_counts += np.bincount(channel_data, minlength=levels)
    return level_counts

def find_peak_luminance(img):
    level_counts = count_levels(img)
    peak_luminance = np.argmax(level_counts)
    return peak_luminance

def generate_bitstream(capacity, embed_rate=0.1):
    embed_capacity = int(capacity * embed_rate)
    bitstream = [0, 1] * (embed_capacity // 2)
    if embed_capacity % 2 != 0:
        bitstream.append(0)
    return np.array(bitstream)

def find_peak_pixels(img, peak_luminance, tolerance=1):
    peak_pixels = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if abs(img[y, x, 0] - peak_luminance) <= tolerance:
                peak_pixels.append((y, x))
    return peak_pixels

def hide_data(marked_img, bitstream, shift, peak_pixels, max_intensity=255, min_intensity=0):
    embedded_bits = []
    embedded_positions = []
    peak_luminance_values = []
    embed_order = 0

    for i, (y, x) in enumerate(peak_pixels):
        if i >= len(bitstream):
            break

        pixel = marked_img[y, x, 0]

        if shift == 1 and pixel < max_intensity - shift:
            marked_img[y, x, 0] = np.clip(marked_img[y, x, 0] + shift * bitstream[i], min_intensity, max_intensity)
            embedded_bits.append(bitstream[i])
            embedded_positions.append((y, x, embed_order))
            peak_luminance_values.append(pixel)
            embed_order += 1
        elif shift == -1 and pixel > min_intensity - shift:
            marked_img[y, x, 0] = np.clip(marked_img[y, x, 0] - shift * bitstream[i], min_intensity, max_intensity)
            embedded_bits.append(bitstream[i])
            embedded_positions.append((y, x, embed_order))
            peak_luminance_values.append(pixel)
            embed_order += 1

    return marked_img, embedded_bits, embedded_positions, peak_luminance_values, len(embedded_bits)

def process_image(img_name, orig_img, embed_rate=0.05):
    height, width, channels = orig_img.shape
    size = height * width
    marked_img = orig_img.copy()

    print(f"{img_name} is a Grayscale image")

    peak_luminance = find_peak_luminance(orig_img)
    np.save(PEAK_PATH + f"{img_name}_peak.npy", peak_luminance)
    print(f"Peak luminance value: {peak_luminance}")

    bitstream = generate_bitstream(size, embed_rate=embed_rate)

    # 初始化峰值序列列表
    peak_sequence = [peak_luminance]
    # 初始化嵌入位置列表
    embedded_positions = []

    # 找到峰值像素的坐標
    peak_pixels = find_peak_pixels(orig_img, peak_luminance, tolerance=1)
    print(f"Number of peak pixels: {len(peak_pixels)}")

    # Step 2: Right-shift histogram and hide bitstream
    marked_img, embedded_bits_right, pos_right, peak_luminance_values_right, actual_embedded_bits_right = hide_data(marked_img, bitstream, shift=1, peak_pixels=peak_pixels)
    if actual_embedded_bits_right > 0:
        print(f"Right-shift embedding completed. Embedded bits: {actual_embedded_bits_right}")
    embedded_positions.extend(pos_right)  # 將右移嵌入位置添加到列表中
    # 保存右移後的峰值
    peak_sequence.append(find_peak_luminance(marked_img))

    # Step 3: Repeat Step 2 until maximum luminance reaches 255 or no more data can be embedded
    while np.max(marked_img) < 255 and len(embedded_positions) < len(bitstream):
        peak_pixels = find_peak_pixels(marked_img, peak_luminance, tolerance=1)
        print(f"Number of peak pixels in Step 3: {len(peak_pixels)}")
        if len(peak_pixels) == 0:
            print("No more peak pixels found in Step 3. Exiting loop.")
            break
        marked_img, _, pos, _, actual_embedded_bits = hide_data(marked_img, bitstream[len(embedded_positions):], shift=1, peak_pixels=peak_pixels)
        embedded_positions.extend(pos)  # 將右移嵌入位置添加到列表中
        # 保存右移後的峰值
        peak_sequence.append(find_peak_luminance(marked_img))

    # Step 4: Left-shift histogram and hide bitstream
    peak_pixels = find_peak_pixels(marked_img, peak_luminance, tolerance=1)
    print(f"Number of peak pixels in Step 4: {len(peak_pixels)}")
    marked_img, embedded_bits_left, pos_left, peak_luminance_values_left, actual_embedded_bits_left = hide_data(marked_img, bitstream[len(embedded_positions):], shift=-1, peak_pixels=peak_pixels)
    if actual_embedded_bits_left > 0:
        print(f"Left-shift embedding completed. Embedded bits: {actual_embedded_bits_left}")
    embedded_positions.extend(pos_left)  # 將左移嵌入位置添加到列表中
    # 保存左移後的峰值
    peak_sequence.append(find_peak_luminance(marked_img))

    # Step 5: Repeat Step 4 until minimum luminance reaches 0 or no more data can be embedded
    while np.min(marked_img) > 0 and len(embedded_positions) < len(bitstream):
        peak_pixels = find_peak_pixels(marked_img, peak_luminance, tolerance=1)
        print(f"Number of peak pixels in Step 5: {len(peak_pixels)}")
        if len(peak_pixels) == 0:
            print("No more peak pixels found in Step 5. Exiting loop.")
            break
        marked_img, _, pos, _, actual_embedded_bits = hide_data(marked_img, bitstream[len(embedded_positions):], shift=-1, peak_pixels=peak_pixels)
        embedded_positions.extend(pos)  # 將左移嵌入位置添加到列表中
        # 保存左移後的峰值
        peak_sequence.append(find_peak_luminance(marked_img))

    actual_payload = actual_embedded_bits_right + actual_embedded_bits_left
    embedded_bits = np.concatenate((embedded_bits_right, embedded_bits_left))[:actual_payload]

    # 將嵌入的數據、峰值序列、嵌入位置和嵌入的bits轉換為NumPy數組並保存到文件
    hide_array = np.array(list(map(int, embedded_bits)))
    np.save(HS_HIDE_DATA_PATH + f"{img_name}_HS_hide_data.npy", hide_array)
    np.save(HS_HIDE_DATA_PATH + f"{img_name}_peak_sequence.npy", np.array(peak_sequence))
    np.save(HS_HIDE_DATA_PATH + f"{img_name}_embedded_positions.npy", np.array(embedded_positions))
    np.save(HS_HIDE_DATA_PATH + f"{img_name}_embedded_bits.npy", np.array(embedded_bits))

    draw_histogram(f"{img_name}", orig_img, HISTOGRAM_PATH)
    draw_histogram(f"{img_name}_marked", marked_img, HISTOGRAM_PATH)

    psnr = calculate_psnr(orig_img, marked_img)
    ssim = calculate_ssim(orig_img, marked_img)
    bpp = actual_payload / size

    print(f"{img_name} marked size = {height} x {width}")
    print(f"Payload = {actual_payload}")
    print(f"Bits per pixel (bpp) = {bpp:.4f}")
    print(f"PSNR = {psnr:.2f}")
    print(f"SSIM = {ssim:.6f}")

    cv2.imwrite(HS_MARKED_PATH + f"{img_name}_shifted.{FILE_TYPE}", marked_img)

    question_for_displaying_images = input("Display images? (y/n): ")
    if question_for_displaying_images == "y":
        cv2.imshow("Original", orig_img)
        cv2.imshow("Marked", marked_img)
        cv2.waitKey(0)

def main():
    img_name = input("Image name: ")
    orig_img = cv2.imread(HS_IMAGES_PATH + f"{img_name}.{FILE_TYPE}", cv2.IMREAD_GRAYSCALE)

    if orig_img is None:
        print("Error reading original image file")
        exit()

    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    print("original image read successfully")
    process_image(img_name, orig_img)

if __name__ == "__main__":
    main()