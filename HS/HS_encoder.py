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
    if len(img.shape) == 2:  # Grayscale image has no channel dimension
        height, width = img.shape
        channels = 1
    else:
        height, width, channels = img.shape

    level_counts = np.zeros(levels, dtype=int)

    for channel in range(channels):
        for level in range(levels):
            level_counts[level] += np.sum(img[:, :, channel] == level) if channels > 1 else np.sum(img == level)

    return level_counts

# Main Logic
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
    marked_img = None  # Renaming marked_img to marked_img

    # Determine if the image is Grayscale or Color
    if np.allclose(orig_img[:,:,0], orig_img[:,:,1]) and np.allclose(orig_img[:,:,1], orig_img[:,:,2]):
        print(f"{img_name} is a Grayscale image")

        draw_histogram_gray(f"{img_name}", orig_img, HISTOGRAM_PATH)
        marked_img = orig_img.copy()
        count = count_levels(marked_img)  # Calculate levels

    # Find peak level
    peak = np.argmax(count)
    np.save(PEAK_PATH + f"{img_name}_peak.npy", peak)

    # Determine shift direction
    shift = -1 if peak == 255 else 1
    map = np.zeros((height, width))

    # Hide data
    # 現在只為等於peak的像素生成隨機數據
    eligible_pixels = np.sum(marked_img[:, :, 0] == peak)
    hide_array = [np.random.choice([0, 1], p=[0.5, 0.5]) for _ in range(eligible_pixels)]
    np.save(HS_HIDE_DATA_PATH + f"{img_name}_HS_hide_data.npy", hide_array)
    i = 0

    # Data hiding logic for grayscale image
    for y in range(height):
        for x in range(width):
            if (shift == 1 and marked_img[y, x, 0] >= peak and marked_img[y, x, 0] != 255) or \
            (shift == -1 and marked_img[y, x, 0] <= peak and marked_img[y, x] != 0):
                if marked_img[y, x, 0] == peak and i < len(hide_array) and hide_array[i] == 1:
                    marked_img[y, x, 0] += shift
                    i += 1
                elif marked_img[y, x, 0] != peak:
                    marked_img[y, x, 0] += shift

        # Draw histogram for processed grayscale image
        draw_histogram_gray(f"{img_name}_marked", marked_img, HISTOGRAM_PATH)

    else:
        print(f"{img_name} is a Color image")

        draw_histogram_color(f"{img_name}", orig_img, HISTOGRAM_PATH)
        marked_img = orig_img.copy()
        count = count_levels(marked_img)  # Calculate levels

        # Find peak level
        peak = np.argmax(count)
        np.save(PEAK_PATH + f"{img_name}_peak.npy", peak)

        # Determine shift direction
        shift = -1 if peak == 255 else 1
        map = np.zeros((height, width, channels))

        # Hide data
        payload = count[peak]
        hide_array = [np.random.choice([0, 1], p=[0.5, 0.5]) for _ in range(payload)]
        i = 0
        np.save(HS_HIDE_DATA_PATH + f"{img_name}_HS_hide_data.npy", hide_array)

        # Data hiding logic for color image
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    if (shift == 1 and marked_img[y, x, c] >= peak and marked_img[y, x, c] != 255) or \
                       (shift == -1 and marked_img[y, x, c] <= peak and marked_img[y, x, c] != 0):
                        if marked_img[y, x, c] == peak and hide_array[i] == 1:
                            marked_img[y, x, c] += shift
                            i += 1
                        else:
                            marked_img[y, x, c] += shift

        # Draw histogram for processed color image
        draw_histogram_color(f"{img_name}_marked", marked_img, HISTOGRAM_PATH)

    # Calculate and print metrics
    psnr = cv2.PSNR(orig_img, marked_img)
    ssim = calculate_ssim(orig_img, marked_img)
    bpp = payload / size
    
    print(f"{img_name} marked size = {height} x {width}")
    print(f"Peak level = {peak}")
    print(f"Payload = {payload}")
    print(f"Bits per pixel (bpp) = {bpp:.4f}")
    print(f"PSNR = {psnr:.2f}") 
    print(f"SSIM = {ssim:.6f}") 

    # Save processed image and info
    cv2.imwrite(HS_MARKED_PATH + f"{img_name}_markedImg.{FILE_TYPE}", marked_img)
    with open(HS_MARKED_PATH + f"{img_name}_info.txt", "w") as f:
        f.write("Grayscale" if np.allclose(orig_img[:,:,0], orig_img[:,:,1]) and np.allclose(orig_img[:,:,1], orig_img[:,:,2]) else "Color")

    # Save and display messages (optional)
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