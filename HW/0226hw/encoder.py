import cv2
import numpy as np
import matplotlib.pyplot as plt

# Constants
FILE_TYPE = "png"
LEVELS = 256
HISTOGRAM_PATH = "./HW/0226hw/histogram/"
HS_IMAGES_PATH = "./HW/0226hw/images/"
HS_MARKED_PATH = "./HS/0226hw/marked/"
PEAK_PATH = "./HW/0226hw/peak/"
HS_HIDE_DATA_PATH = "./HW/0226hw/hide_data/"

def draw_histogram_gray_s_curve(img_name, img, save_path):
    if img.ndim == 2:
        height, width = img.shape
    else:
        height, width, _ = img.shape
        img = img[:, :, 0]

    img = img.astype(np.int16)  # 转换图像数据类型以避免溢出
    diff_count = [0] * 512

    for y in range(height):
        if y % 2 == 0:  # 偶数行：左边像素减去右边像素
            for x in range(1, width):
                diff = img[y, x - 1] - img[y, x]
                diff_index = diff + 255
                diff_count[diff_index] += 1
        else:  # 奇数行：右边像素减去左边像素
            for x in range(width - 1):
                diff = img[y, x + 1] - img[y, x]
                diff_index = diff + 255
                diff_count[diff_index] += 1

    plt.figure()
    plt.xlabel("Pixel Intensity Difference")
    plt.ylabel("Difference Count")
    plt.bar(range(-255, 257), diff_count)
    plt.savefig(save_path + f"{img_name}_diff_histogram.png")
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

import numpy as np

def calculate_ssim(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")
    
    # 检查图像是否为灰度图像
    if img1.ndim == 2:
        # 灰度图像只有两个维度
        height, width = img1.shape
        channels = 1
    else:
        # 彩色图像有三个维度
        height, width, channels = img1.shape

    size = height * width

    # Constants for SSIM calculation
    C1 = (255 * 0.01) ** 2
    C2 = (255 * 0.03) ** 2
    C3 = C2 / 2

    ssim_total = 0

    for channel in range(channels):
        if channels > 1:
            img1_c = img1[:, :, channel]
            img2_c = img2[:, :, channel]
        else:
            img1_c = img1
            img2_c = img2

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

def count_intensity_diffs_s_curve(img, levels=512):
    # 检查图像是否为二维（灰度图像），如果不是，假设它是三维的，并获取第一个通道
    if img.ndim == 2:
        height, width = img.shape
    else:
        height, width, _ = img.shape
        img = img[:, :, 0]  # 如果是彩色图像，只取一个通道
    
    img = img.astype(np.int16)  # 避免溢出
    diff_counts = np.zeros(levels, dtype=int)  # 初始化差异计数器，考虑-255到255的范围

    for y in range(height):
        if y % 2 == 0:  # 偶数行：左边像素减去右边像素
            for x in range(1, width):
                diff = img[y, x - 1] - img[y, x] + 255  # +255是为了使索引从0开始
                diff_counts[diff] += 1
        else:  # 奇数行：右边像素减去左边像素
            for x in range(width - 1):
                diff = img[y, x + 1] - img[y, x] + 255  # 同上
                diff_counts[diff] += 1

    return diff_counts

def count_levels_color(img):
    count = [np.bincount(img[:, :, channel].flatten(), minlength=256) for channel in range(3)]
    return count

def hide_data_in_grayscale_image_with_diff(marked_img, secret_data):
    # 假设我们选择差值为0的位置进行隐藏
    peak_diff = 0  # 这是基于您提到的直方图分析结果

    height, width = marked_img.shape[:2]
    marked_img = marked_img.astype(np.int16)  # 为避免溢出，使用更大范围的整型
    data_index = 0  # 初始化数据索引

    for y in range(height):
        for x in range(width - 1):
            if data_index >= len(secret_data):
                break

            # 计算当前像素对的差值
            current_diff = marked_img[y, x + 1] - marked_img[y, x]

            if current_diff == peak_diff:
                bit = int(secret_data[data_index])
                # 为了简化，我们只在差值为0的地方隐藏数据
                # 并且只在需要隐藏1的时候调整像素值
                if bit == 1:
                    # 通过增加相邻像素的值来隐藏数据
                    marked_img[y, x + 1] += 1
                data_index += 1

    # 限制值范围并转换回uint8
    marked_img = np.clip(marked_img, 0, 255).astype(np.uint8)

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
    orig_img = cv2.imread(HS_IMAGES_PATH + f"{img_name}.{FILE_TYPE}", cv2.IMREAD_GRAYSCALE)

    if orig_img is None:
        print("Error reading original image file")
        exit()

    print("original image read successfully")
    process_image(img_name, orig_img)

def process_image(img_name, orig_img):
    height, width = orig_img.shape[:2]
    size = height * width
    marked_img = orig_img.copy()
    eligible_pixels = int(height * width / 2)


    # 检查图像是否为灰度图像
    if len(orig_img.shape) == 2 or (np.allclose(orig_img[:,:,0], orig_img[:,:,1]) and np.allclose(orig_img[:,:,1], orig_img[:,:,2])):
        print(f"{img_name} is a Grayscale image")
        diff_counts = count_intensity_diffs_s_curve(orig_img)
        # 选择嵌入点：选择频率最低的非零差值
        non_zero_diff_counts = np.where(diff_counts > 0, diff_counts, np.inf)
        peak_diff_index = np.argmin(non_zero_diff_counts)
        peak_diff = peak_diff_index - 255  # 将索引转换回差值
        
        hide_array = ''.join(np.random.choice(['0', '1'], p=[0.5, 0.5], size=eligible_pixels))

        
        # 嵌入数据
        marked_img = hide_data_in_grayscale_image_with_diff(marked_img, hide_array)  # 注意这里的改动
        
        # 绘制原始和标记图像的直方图
        draw_histogram_gray_s_curve(f"{img_name}", orig_img, HISTOGRAM_PATH)
        draw_histogram_gray_s_curve(f"{img_name}_marked", marked_img, HISTOGRAM_PATH)

        # 保存灰度图像的峰值信息和嵌入数据
        np.save(HS_HIDE_DATA_PATH + f"{img_name}_HS_hide_data.npy", np.array([bit for bit in hide_array]))
        np.save(PEAK_PATH + f"{img_name}_peak_diff.npy", np.array([peak_diff]))
        print(f"Peak difference level for hiding: {peak_diff}")

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