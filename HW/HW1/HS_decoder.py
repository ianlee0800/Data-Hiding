import cv2
import numpy as np
import os

# Constants
FILE_TYPE = "tiff"
LEVELS = 256
HISTOGRAM_PATH = "./HW/HW1/histogram/"
HS_IMAGES_PATH = "./HW/HW1/images/"
HS_MARKED_PATH = "./HW/HW1/marked/"
PEAK_PATH = "./HW/HW1/peak/"
HS_HIDE_DATA_PATH = "./HW/HW1/hide_data/"

def revert_data(marked_img, peak_sequence, embedded_positions):
    height, width = marked_img.shape[:2]
    reverted_img = marked_img.copy()

    # 按相反順序執行左移和右移操作
    for i in range(len(peak_sequence) - 1, 0, -1):
        cur_peak = peak_sequence[i]
        prev_peak = peak_sequence[i - 1]

        for y, x in embedded_positions:
            pixel = marked_img[y, x, 0]

            if pixel == cur_peak:
                if cur_peak > prev_peak:  # 右移操作
                    reverted_img[y, x, 0] -= 1
                else:  # 左移操作
                    reverted_img[y, x, 0] += 1

    return reverted_img

def extract_data(marked_img, peak_luminance):
    height, width = marked_img.shape[:2]
    extracted_bits = ''
    right_shift_done = False
    left_shift_done = False

    # 遍歷圖像每個像素
    for y in range(height):
        for x in range(width):
            pixel = marked_img[y, x, 0]

            # 右移階段
            if not right_shift_done:
                if pixel == peak_luminance + 1:
                    extracted_bits += '1'
                elif pixel == peak_luminance:
                    extracted_bits += '0'
                else:
                    # 遇到超過最大值255的像素,右移階段結束
                    if pixel > 255:
                        right_shift_done = True
                        break

            # 左移階段
            elif not left_shift_done:
                if pixel == peak_luminance - 1:
                    extracted_bits += '1'
                elif pixel == peak_luminance:
                    extracted_bits += '0'
                else:
                    # 遇到低於最小值0的像素,左移階段結束
                    if pixel < 0:
                        left_shift_done = True
                        break

            # 兩個階段都結束,提取完畢
            if right_shift_done and left_shift_done:
                break

    return extracted_bits

def recover_image(marked_img, peak_luminance):
    height, width = marked_img.shape[:2]
    recovered_img = marked_img.copy()

    for y in range(height):
        for x in range(width):
            pixel = marked_img[y, x, 0]
            if pixel != peak_luminance:
                recovered_img[y, x, 0] = peak_luminance

    return recovered_img

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

def decode_and_recover(img_name):
    marked_img_path = os.path.join("./HW/HW1/histogram_shifted/", f"{img_name}_shifted.{FILE_TYPE}")
    marked_img = cv2.imread(marked_img_path, cv2.IMREAD_GRAYSCALE)
    marked_img = cv2.cvtColor(marked_img, cv2.COLOR_GRAY2BGR)
    orig_img = cv2.imread(HS_IMAGES_PATH + f"{img_name}.{FILE_TYPE}", cv2.IMREAD_GRAYSCALE)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)

    peak_luminance = np.load(PEAK_PATH + f"{img_name}_peak.npy")
    hide_data = np.load(HS_HIDE_DATA_PATH + f"{img_name}_HS_hide_data.npy")
    peak_sequence = np.load(HS_HIDE_DATA_PATH + f"{img_name}_peak_sequence.npy")
    embedded_positions = np.load(HS_HIDE_DATA_PATH + f"{img_name}_embedded_positions.npy")

    extracted_bits = extract_data(marked_img, peak_luminance)
    # reverted_img = recover_image(marked_img, peak_luminance)
    reverted_img = revert_data(marked_img, peak_sequence, embedded_positions)

    print(f"Extracted bits: {extracted_bits}")
    print(f"Original embedded bits: {''.join(map(str, hide_data))}")

    if extracted_bits == ''.join(map(str, hide_data)):
        print("Extracted data is correct.")
    else:
        print("Extracted data is incorrect.")

    psnr = calculate_psnr(orig_img, reverted_img)
    ssim = calculate_ssim(orig_img, reverted_img)

    print(f"PSNR between original and recovered image: {psnr:.2f}")
    print(f"SSIM between original and recovered image: {ssim:.6f}")

    cv2.imwrite(HS_MARKED_PATH + f"{img_name}_recoveredImg.{FILE_TYPE}", reverted_img)

    question_for_displaying_images = input("Display images? (y/n): ")
    if question_for_displaying_images == "y":
        cv2.imshow("Original", orig_img)
        cv2.imshow("Marked", marked_img)
        cv2.imshow("Recovered", reverted_img)
        cv2.waitKey(0)

def main():
    img_name = input("Image name: ")
    decode_and_recover(img_name)

if __name__ == "__main__":
    main()