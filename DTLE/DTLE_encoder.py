import cv2
import numpy as np
import matplotlib.pyplot as plt

# Constants
FILE_TYPE = "png"
DTLE_IMAGES_PATH = "./DTLE/images/"
DTLE_MARKED_PATH = "./DTLE/marked/"
DTLE_LOCATION_MAP_PATH = "./DTLE/data_for_embedding/location_map/"
DTLE_LSB_PATH = "./DTLE/data_for_embedding/lsb_plane/"
DTLE_HSB_PATH = "./DTLE/data_for_embedding/hsb_plane/"
DTLE_PREDICTED_PATH = "./DTLE/data_for_embedding/predicted_img/"
DTLE_ADJUSTED_PATH = "./DTLE/data_for_embedding/diff_img_adjusted/"
DTLE_EMIN_PATH = "./DTLE/data_for_embedding/e_min/"
HISTOGRAM_PATH = "./DTLE/histogram/"

def med_predictor(img):
    height, width = img.shape
    predicted_img = np.zeros((height, width), dtype=np.int32)
    diff_img = np.zeros((height, width), dtype=np.int32)

    for y in range(1, height):
        for x in range(1, width):
            a = int(img[y, x - 1])
            b = int(img[y - 1, x])
            c = int(img[y - 1, x - 1])

            if c <= min(a, b):
                predicted_val = max(a, b)
            elif c >= max(a, b):
                predicted_val = min(a, b)
            else:
                predicted_val = a + b - c

            predicted_img[y, x] = np.clip(predicted_val, 0, 255)
            diff_img[y, x] = img[y, x] - predicted_img[y, x]

    e_min = np.min(diff_img)
    diff_img_adjusted = diff_img + abs(e_min)

    return predicted_img.astype(np.uint8), diff_img_adjusted.astype(np.uint8), e_min

def create_location_map(I_H):
    height, width = I_H.shape
    LM = np.zeros((height, width), dtype=np.uint8)

    # 根據公式10修改像素值並創建位置圖
    for y in range(height):
        for x in range(width):
            if I_H[y, x] == 63:
                LM[y, x] = 1
                I_H[y, x] -= 2
            elif I_H[y, x] == 62:
                LM[y, x] = 1
                I_H[y, x] -= 1
            elif I_H[y, x] == 0:
                LM[y, x] = 1
                I_H[y, x] += 2
            elif I_H[y, x] == 1:
                LM[y, x] = 1
                I_H[y, x] += 1

    return LM, I_H

def decompose_bit_planes(img, hsb_bits):
    # 檢查 hsb_bits 的有效性
    if not (1 <= hsb_bits <= 7):
        raise ValueError("hsb_bits must be between 1 and 7")

    height, width = img.shape
    hsb_plane = np.zeros((height, width), dtype=np.uint8)
    lsb_plane = np.zeros((height, width), dtype=np.uint8)

    # HSB 和 LSB 的遮罩
    hsb_mask = 0xFF << (8 - hsb_bits)  # e.g., 若 hsb_bits = 3, 則 hsb_mask = 0b11100000
    lsb_mask = 0xFF >> hsb_bits       # e.g., 若 hsb_bits = 3, 則 lsb_mask = 0b00011111

    # 使用位操作提取 HSB 和 LSB
    hsb_plane = (img & hsb_mask) >> (8 - hsb_bits)
    lsb_plane = img & lsb_mask

    return hsb_plane, lsb_plane

def create_location_map(hsb_img):
    height, width = hsb_img.shape
    location_map = np.zeros((height, width), dtype=np.uint8)

    # 論文中的特定條件
    location_map[(hsb_img == 63)] = 1  # 最大值
    location_map[(hsb_img == 0)] = 1   # 最小值
    location_map[(hsb_img == 62)] = 1  # 最大值減 1
    location_map[(hsb_img == 1)] = 1   # 最小值加 1

    # 根據位置地圖進行像素調整
    hsb_img_adjusted = np.copy(hsb_img)
    hsb_img_adjusted[hsb_img == 63] -= 2
    hsb_img_adjusted[hsb_img == 62] -= 1
    hsb_img_adjusted[hsb_img == 0] += 2
    hsb_img_adjusted[hsb_img == 1] += 1

    return location_map, hsb_img_adjusted

def extract_neighbors(img, x, y):
    height, width = img.shape[:2]
    
    # 定义邻近像素的相对位置
    dx = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dy = np.array([-1, 0, 1, -1, 1, -1, 0, 1])

    # 计算邻近像素的坐标
    nx = x + dx
    ny = y + dy

    # 确保邻近像素在图像范围内
    valid = (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height)
    nx = nx[valid]
    ny = ny[valid]

    # 提取邻近像素的值
    neighbors = img[ny, nx]

    return neighbors

def calculate_predicted_values(img, x, y):
    # 使用 extract_neighbors 函数获取邻居像素
    neighbors = extract_neighbors(img, x, y)

    # 检查是否有8个邻居像素值
    if len(neighbors) != 8:
        raise ValueError("必须提供8个邻居像素值")

    # 根据论文中的描述计算 p1 和 p2
    sorted_values = sorted(neighbors)
    p1 = sum(sorted_values[:6]) // 6  # 前6个邻居像素的平均值
    p2 = sum(sorted_values[2:8]) // 6  # 第3到第8个邻居像素的平均值
    return p1, p2

def create_hide_array(image, location_map):
    height, width = image.shape
    # 可用于嵌入的像素数量等于未标记为溢出的像素数量
    payload_size = ((height * width) - np.sum(location_map)) // 2

    # 直接生成指定数量的隐藏数据对
    hide_array = [(np.random.choice([0, 1]), np.random.choice([0, 1])) for _ in range(payload_size)]
    return hide_array

def embed_data(pixel, bit, error, predicted_value):
    # 根据预测误差和秘密数据位更新像素值
    if error == 1:
        new_pixel = pixel + bit  # 如果误差为1，加上秘密数据位
    elif error > 1:
        new_pixel = pixel + 1  # 如果误差大于1，加1
    elif error == 0:
        new_pixel = pixel - bit  # 如果误差为0，减去秘密数据位
    elif error < 0:
        new_pixel = pixel - 1  # 如果误差小于0，减1
    else:
        new_pixel = pixel  # 其他情况保持像素不变

    # 确保像素值在0到255的范围内
    new_pixel = np.clip(new_pixel, 0, 255)
    return new_pixel

def embed_data_chessboard(I_H, secret_data):
    height, width = I_H.shape
    data_index = 0
    max_data_index = len(secret_data)

    # 棋盘式嵌入
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if data_index >= max_data_index:
                break

            # 检查是否是蓝色像素位置
            if (x + y) % 2 == 0:  # 根据棋盘式掃描的模式，只处理蓝色像素位置
                p1, p2 = calculate_predicted_values(I_H, x, y)
                e1 = I_H[y, x] - p1
                e2 = I_H[y, x] - p2

                b1, b2 = secret_data[data_index]  # 提取要嵌入的两位数据

                # 第一层嵌入
                I_H[y, x] = embed_data(I_H[y, x], b1, e1, p1)
                # 更新e2的值以用于第二层嵌入
                e2 = I_H[y, x] - p2
                # 第二层嵌入
                I_H[y, x] = embed_data(I_H[y, x], b2, e2, p2)

                data_index += 1

    return I_H

def draw_histogram_gray(img_name, img, save_path):
    if len(img.shape) == 2:  # 如果是灰度图像
        height, width = img.shape
    else:  # 如果是彩色图像
        height, width, _ = img.shape

    q = 256
    count = [0] * q
    for y in range(height):
        for x in range(width):
            intensity = int(img[y, x]) if len(img.shape) == 2 else int(img[y, x, 0])
            count[intensity] += 1

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

def calculate_ssim_single_channel(img1_c, img2_c):
    # Constants for SSIM calculation
    C1 = (255 * 0.01) ** 2
    C2 = (255 * 0.03) ** 2
    C3 = C2 / 2

    mean1 = np.mean(img1_c)
    mean2 = np.mean(img2_c)
    var1 = np.var(img1_c)
    var2 = np.var(img2_c)
    covar = np.cov(img1_c.flatten(), img2_c.flatten())[0][1]

    luminance = (2 * mean1 * mean2 + C1) / (mean1 ** 2 + mean2 ** 2 + C1)
    contrast = (2 * np.sqrt(var1) * np.sqrt(var2) + C2) / (var1 + var2 + C2)
    structure = (covar + C3) / (np.sqrt(var1) * np.sqrt(var2) + C3)

    return luminance * contrast * structure

def calculate_ssim(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")

    if len(img1.shape) == 2:  # 如果是灰度图像
        return calculate_ssim_single_channel(img1, img2)
    else:  # 如果是彩色图像
        height, width, channels = img1.shape
        ssim_total = 0

        for channel in range(channels):
            img1_c = img1[:, :, channel]
            img2_c = img2[:, :, channel]
            ssim_total += calculate_ssim_single_channel(img1_c, img2_c)

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

def hide_data_in_grayscale_image_chessboard(marked_img, hide_array):
    print("First few elements of hide_array:", hide_array[:10])  # 打印前10个元素
    
    if not hide_array:
        print("No data to hide. hide_array is empty.")
        return marked_img  # 或者处理这种情况
    
    # 确保图像是二维的
    if len(marked_img.shape) != 2:
        raise ValueError("图像必须是灰度图像（二维数组）")

    # 计算图像的平均阈值
    threshold = np.mean(marked_img)

    # 将hide_array转换为与marked_img同样大小的二维数组
    height, width = marked_img.shape
    hide_data_2d = np.zeros((height, width), dtype=np.int32)

    data_index = 0
    for y in range(height):
        for x in range(width):
            if marked_img[y, x] > threshold and data_index < len(hide_array):
                hide_data_2d[y, x] = hide_array[data_index]
                data_index += 1

    # 使用棋盘式扫描方法嵌入数据
    marked_img = embed_data_chessboard(marked_img, hide_data_2d)

    # 确保处理了所有的隐藏数据
    if data_index < len(hide_array):
        raise ValueError("未能嵌入所有数据，隐藏数据数组可能太长")

    return marked_img

def main():
    img_name = input("Image name: ")
    orig_img = cv2.imread(DTLE_IMAGES_PATH + f"{img_name}.{FILE_TYPE}")

    if orig_img is None:
        print("Error reading original image file")
        exit()

    print("original image read successfully")
    process_image(img_name, orig_img)

def process_image(img_name, orig_img):
    # 檢查圖像是否為灰度圖像
    if len(orig_img.shape) > 2 and orig_img.shape[2] == 3:
        # 如果是彩色圖像，轉換為灰度圖像
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    height, width = orig_img.shape
    size = height * width

    # 调用MED预处理器
    predicted_img, diff_img_adjusted, e_min = med_predictor(orig_img)

    # 调用位平面分解函数
    hsb_bits = 2  # 或者根据您的需求选择其他位数
    hsb_plane, lsb_plane = decompose_bit_planes(diff_img_adjusted, hsb_bits)

    # 创建位置图
    location_map, hsb_plane = create_location_map(hsb_plane)

    # 创建隐藏数据数组
    hide_array = create_hide_array(hsb_plane, location_map)

    # 使用棋盘式嵌入方法
    marked_img = hide_data_in_grayscale_image_chessboard(hsb_plane, hide_array)

    draw_histogram_gray(f"{img_name}", orig_img, HISTOGRAM_PATH)
    draw_histogram_gray(f"{img_name}_marked", marked_img, HISTOGRAM_PATH)

    # 计算和打印度量标准
    psnr = calculate_psnr(orig_img, marked_img)
    ssim = calculate_ssim(orig_img, marked_img)
    bpp = len(hide_array) / size

    print(f"{img_name} marked size = {height} x {width}")
    print(f"Payload = {len(hide_array)}")
    print(f"Bits per pixel (bpp) = {bpp:.4f}")
    print(f"PSNR = {psnr:.2f}") 
    print(f"SSIM = {ssim:.6f}") 

    # 保存位置图和其他数据
    np.save(DTLE_LOCATION_MAP_PATH + f"{img_name}_location_map.npy", location_map)
    np.save(DTLE_HSB_PATH + f"{img_name}_hsb_plane.npy", hsb_plane)
    np.save(DTLE_LSB_PATH + f"{img_name}_lsb_plane.npy", lsb_plane)
    np.save(DTLE_PREDICTED_PATH + f"{img_name}_predicted_img.npy", predicted_img)
    np.save(DTLE_ADJUSTED_PATH + f"{img_name}_diff_img_adjusted.npy", diff_img_adjusted)
    np.save(DTLE_EMIN_PATH + f"{img_name}_e_min.npy", e_min)
    cv2.imwrite(DTLE_MARKED_PATH + f"{img_name}_markedImg.{FILE_TYPE}", marked_img)

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