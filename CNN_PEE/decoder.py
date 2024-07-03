import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def load_embedding_info(path):
    return np.load(path, allow_pickle=True)

def pee_extracting(stego_image, embedding_info):
    extracted_data = []
    restored_image = stego_image.copy()

    for info in embedding_info:
        i, j = info['position']
        stego_value = int(stego_image[i, j])
        predicted_value = info['predicted_value']
        error = info['error']
        original_value = info['original_value']

        if error >= 0:
            extracted_bit = (stego_value - predicted_value - 2 * error)
        else:
            extracted_bit = -(stego_value - predicted_value - 2 * error)

        extracted_data.append(extracted_bit)
        restored_image[i, j] = original_value

        print(f"位置 ({i}, {j}): stego值 = {stego_value}, 预测值 = {predicted_value}, 错误 = {error}")
        print(f"提取位 = {extracted_bit}, 还原值 = {original_value}")

    return np.array(extracted_data), restored_image

def calculate_psnr(original, restored):
    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

if __name__ == "__main__":
    try:
        # 读取 stego 图像
        stego_image_path = "./CNN_PEE/stego/Tank_stego.png"
        stego_image = load_image(stego_image_path)

        # 读取嵌入信息
        embedding_info = load_embedding_info("./CNN_PEE/stego/embedding_info.npy")

        # 读取原始秘密数据
        original_secret_data = np.load("./CNN_PEE/stego/original_secret_data.npy")

        # 读取原始图像（用于比较）
        original_image_path = "./CNN_PEE/origin/Tank.png"
        original_image = load_image(original_image_path)

        # 验证加载的 stego 图像
        saved_stego_values = np.load("./CNN_PEE/stego/stego_values.npy")
        stego_match = np.array_equal(stego_image, saved_stego_values)
        print(f"加载的 stego 图像与保存的 stego 值完全匹配: {stego_match}")

        if not stego_match:
            mismatch_count = np.sum(stego_image != saved_stego_values)
            print(f"不匹配的像素数: {mismatch_count}")
            print(f"不匹配率: {mismatch_count / (stego_image.shape[0] * stego_image.shape[1]):.4f}")

        # 打印嵌入信息的前10项
        print(f"嵌入信息的前10项：")
        for i, info in enumerate(embedding_info[:10]):
            print(f"位置 {i}: {info}")

        # 提取数据和还原图像
        extracted_data, restored_image = pee_extracting(stego_image, embedding_info)
        
        # 计算 PSNR 和 SSIM
        psnr = calculate_psnr(original_image, restored_image)
        ssim_value = ssim(original_image, restored_image, data_range=original_image.max() - original_image.min())

        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM: {ssim_value:.6f}")

        # 验证提取的数据
        print(f"嵌入数据长度: {len(original_secret_data)}, 提取的数据长度: {len(extracted_data)}")

        # 比较提取的数据与原始秘密数据
        data_match = np.array_equal(extracted_data, original_secret_data[:len(extracted_data)])
        print(f"提取的数据与原始数据完全匹配: {data_match}")

        if not data_match:
            mismatched_count = np.sum(extracted_data != original_secret_data[:len(extracted_data)])
            print(f"不匹配的位元数: {mismatched_count}")
            print(f"不匹配率: {mismatched_count / len(extracted_data):.4f}")
            
            # 显示不匹配的位置
            mismatch_indices = np.where(extracted_data != original_secret_data[:len(extracted_data)])[0]
            print("不匹配的位置及其值：")
            for idx in mismatch_indices:
                print(f"位置 {idx}: 提取值 {extracted_data[idx]}, 原始值 {original_secret_data[idx]}")

        # 验证还原的图像
        image_match = np.array_equal(restored_image, original_image)
        print(f"还原的图像与原始图像完全匹配: {image_match}")

        if not image_match:
            diff = np.abs(restored_image.astype(int) - original_image.astype(int))
            print(f"不匹配的像素数: {np.sum(diff != 0)}")
            print(f"最大像素差异: {np.max(diff)}")
            print(f"平均像素差异: {np.mean(diff):.4f}")
            
            # 显示一些不匹配像素的详细信息
            mismatch_indices = np.where(diff != 0)
            for i in range(min(10, len(mismatch_indices[0]))):  # 显示前10个不匹配的像素
                x, y = mismatch_indices[0][i], mismatch_indices[1][i]
                print(f"位置 ({x}, {y}): 原始值 {original_image[x, y]}, 还原值 {restored_image[x, y]}")

        # 保存还原的图像
        cv2.imwrite("./CNN_PEE/restored/Tank_restored.png", restored_image)

        print(f"嵌入位置数量: {len(embedding_info)}")
        print(f"嵌入错误范围: {min([info['error'] for info in embedding_info])} 到 {max([info['error'] for info in embedding_info])}")
        print(f"原始秘密数据: {original_secret_data}")
        print(f"提取的秘密数据: {extracted_data}")

        print("解码完成。还原的图像已保存。")

    except Exception as e:
        print(f"发生未预期的错误：{e}")
        import traceback
        traceback.print_exc()