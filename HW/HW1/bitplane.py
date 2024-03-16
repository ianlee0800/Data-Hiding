import cv2
import os

# 讀取圖像的函數
def read_image(folder_path, filename):
    img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
    return img

# 提取位元平面的函數
def extract_bitplanes(img):
    bitplanes = []
    for i in range(8):
        bitplane = np.zeros_like(img)
        bitplane[(img & (1 << i)) > 0] = 255
        bitplanes.append(bitplane)
    return bitplanes

# 主程式
if __name__ == "__main__":
    # 詢問使用者要讀取的圖片名稱
    filename = input("請輸入要讀取的圖片名稱(不含副檔名): ")
    filename = f"{filename}.tiff"

    # 讀取圖像
    folder_path = "./HW/HW1/images"
    img = read_image(folder_path, filename)

    if img is None:
        print(f"無法讀取圖像 {filename}")
        exit()

    # 建立輸出資料夾
    bitplanes_folder = "./HW/HW1/bitplanes"
    os.makedirs(bitplanes_folder, exist_ok=True)

    # 提取位元平面
    bitplanes = extract_bitplanes(img)

    # 儲存每個位元平面為圖像
    for i in range(8):
        output_path = os.path.join(bitplanes_folder, f"{os.path.splitext(filename)[0]}_bitplane{i}.tiff")
        cv2.imwrite(output_path, bitplanes[i])