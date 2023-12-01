import cv2
import numpy as np
import matplotlib.pyplot as plt

# Constants
FILE_TYPE = "png"
LEVELS = 256
HISTOGRAM_PATH = "./HS/histogram/"
HS_IMAGES_PATH = "./HS/images/"
HS_MARKED_PATH = "./HS/marked/"
PEAK_PATH = "./HS/peak/"
HS_HIDE_DATA_PATH = "./HS/hide_data/"

def draw_histogram_gray(img_name, img, save_path):
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

def draw_histogram_color(img_name, img, save_path):
    height, width, _ = img.shape
    q = 256
    count = np.zeros((q, 3), dtype=int)
    for y in range(height):
        for x in range(width):
            for c in range(3):
                bgr = int(img[y, x, c])
                count[bgr, c] += 1

    plt.figure()
    colors = ['b', 'g', 'r']  # BGR順序的顏色
    channel_names = ['Blue Channel', 'Green Channel', 'Red Channel']
    for c in range(3):
        plt.plot(range(1, 257), count[:, c], color=colors[c], label=channel_names[c])
    plt.xlabel("Pixel Intensity")  # 添加X軸標籤
    plt.ylabel("Pixel Count")  # 添加Y軸標籤
    plt.legend()
    plt.savefig(save_path + f"{img_name}_histogram.png")
    plt.close()

def calculate_psnr(img1, img2, max_pixel=255):
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    psnr = 10 * np.log10(max_pixel / np.sqrt(mse))
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

def count_levels_color(img):
    count = [np.bincount(img[:, :, channel].flatten(), minlength=256) for channel in range(3)]
    return count

def hide_data_in_grayscale_image(marked_img, peak, shift, hide_array):
    height, width = marked_img.shape[:2]
    i = 0

    # 遍歷圖像的每個像素
    for y in range(height):
        for x in range(width):
            if i >= len(hide_array):  # 檢查是否所有的隱藏數據都已處理
                break

            pixel = marked_img[y, x, 0]
            if (shift == 1 and pixel >= peak and pixel != 255) or \
               (shift == -1 and pixel <= peak and pixel != 0):
                if pixel == peak and hide_array[i] == 1:
                    marked_img[y, x, 0] += shift
                    i += 1
                elif pixel != peak:
                    marked_img[y, x, 0] += shift

    return marked_img

def hide_data_in_color_channel(channel_img, peak, shift, hide_array):
    height, width = channel_img.shape
    i = 0

    for y in range(height):
        for x in range(width):
            if i >= len(hide_array):  # 检查是否所有的隐藏数据都已处理
                break

            pixel = channel_img[y, x]
            if (shift == 1 and pixel >= peak and pixel != 255) or \
               (shift == -1 and pixel <= peak and pixel != 0):
                if pixel == peak and hide_array[i] == 1:
                    channel_img[y, x] += shift
                    i += 1
                elif pixel != peak:
                    channel_img[y, x] += shift

    return channel_img

def main():
    img_name = input("Image name: ")
    orig_img = cv2.imread(HS_IMAGES_PATH + f"{img_name}.{FILE_TYPE}")

    if orig_img is None:
        print("Error reading original image file")
        exit()

    print("original image read successfully")
    process_image(img_name, orig_img)

def process_image(img_name, orig_img):
    height, width, channels = orig_img.shape
    size = height * width
    marked_img = orig_img.copy()

    if channels == 1 or (np.allclose(orig_img[:,:,0], orig_img[:,:,1]) and np.allclose(orig_img[:,:,1], orig_img[:,:,2])):
        print(f"{img_name} is a Grayscale image")
        count = count_levels(marked_img)
        peak = np.argmax(count)
        shift = -1 if peak == 255 else 1

        eligible_pixels = np.sum(marked_img[:, :, 0] == peak)
        hide_array = [np.random.choice([0, 1], p=[0.5, 0.5]) for _ in range(eligible_pixels)]
        np.save(HS_HIDE_DATA_PATH + f"{img_name}_HS_hide_data.npy", hide_array)

        marked_img = hide_data_in_grayscale_image(marked_img, peak, shift, hide_array)
        draw_histogram_gray(f"{img_name}", orig_img, HISTOGRAM_PATH)
        draw_histogram_gray(f"{img_name}_marked", marked_img, HISTOGRAM_PATH)

        # 保存灰度图像的峰值信息
        np.save(PEAK_PATH + f"{img_name}_peak.npy", peak)
        print(f"Peak level = {peak}")

    else:
        print(f"{img_name} is a Color image")
        peak_levels = []
        for c in range(3):
            channel_img = marked_img[:, :, c]
            channel_count = np.bincount(channel_img.flatten(), minlength=256)
            peak = np.argmax(channel_count)
            peak_levels.append(peak)
            shift = -1 if peak == 255 else 1

            eligible_pixels = np.sum(channel_img == peak)
            hide_array = [np.random.choice([0, 1], p=[0.5, 0.5]) for _ in range(eligible_pixels)]
            marked_img[:, :, c] = hide_data_in_color_channel(channel_img, peak, shift, hide_array)

        draw_histogram_color(f"{img_name}", orig_img, HISTOGRAM_PATH)
        draw_histogram_color(f"{img_name}_marked", marked_img, HISTOGRAM_PATH)

        # 保存彩色图像的峰值信息
        np.save(PEAK_PATH + f"{img_name}_peak.npy", peak_levels)
        print(f"Peak levels = [Red: {peak_levels[2]}, Green: {peak_levels[1]}, Blue: {peak_levels[0]}]")

    # 计算和打印度量标准
    psnr = calculate_psnr(orig_img, marked_img)
    ssim = calculate_ssim(orig_img, marked_img)
    bpp = len(hide_array) / size

    print(f"{img_name} marked size = {height} x {width}")
    print(f"Payload = {len(hide_array)}")
    print(f"Bits per pixel (bpp) = {bpp:.4f}")
    print(f"PSNR = {psnr:.2f}") 
    print(f"SSIM = {ssim:.6f}") 

    # 保存处理过的图像
    cv2.imwrite(HS_MARKED_PATH + f"{img_name}_markedImg.{FILE_TYPE}", marked_img)

    # 可选的显示和保存嵌入数据
    question_for_showing_embedding_data = input("Show embedding data? (y/n): ")
    if question_for_showing_embedding_data == "y":
        print(f"Message = {hide_array}")
        print(f"Message length = {len(hide_array)}")

    question_for_displaying_images = input("Display images? (y/n): ")
    if question_for_displaying_images == "y":
        cv2.imshow("Original", orig_img)
        cv2.imshow("Marked", marked_img)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()