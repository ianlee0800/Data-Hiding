import cv2
import numpy as np
import os
from itertools import product
from HS_encoder import FILE_TYPE, HISTOGRAM_PATH, HS_IMAGES_PATH, HS_MARKED_PATH, PEAK_PATH, HS_HIDE_DATA_PATH
from HS_encoder import calculate_psnr, calculate_ssim

def revert_data(marked_img, peak_value, peak_sequence, embedded_positions):
    height, width = marked_img.shape[:2]
    reverted_img = marked_img.copy()

    # 按相反順序執行左移和右移操作
    for i in range(len(peak_sequence) - 1, 0, -1):
        cur_peak = peak_sequence[i]
        prev_peak = peak_sequence[i - 1]

        for y, x, embed_order in embedded_positions:
            pixel = marked_img[y, x, 0]

            if pixel == cur_peak:
                if cur_peak > prev_peak:  # 右移操作
                    reverted_img[y, x, 0] = peak_value
                else:  # 左移操作
                    reverted_img[y, x, 0] = peak_value

    return reverted_img

def extract_data(marked_img, peak_value, peak_sequence, embedded_positions):
    extracted_bits = []
    height, width = marked_img.shape[:2]
    sequence_length = len(peak_sequence)
    embedded_bits_index = 0

    for y, x, embed_order in embedded_positions:
        pixel = marked_img[y, x, 0]
        
        if embed_order < sequence_length:
            prev_peak = peak_sequence[embed_order]
        else:
            prev_peak = peak_value
        
        if embed_order < sequence_length - 1:
            cur_peak = peak_sequence[embed_order + 1]
        else:
            cur_peak = peak_value

        if pixel == cur_peak:
            if cur_peak > prev_peak:  # 右移操作
                extracted_bits.append(1)
            else:  # 左移操作
                extracted_bits.append(0)
        else:
            if cur_peak > prev_peak:  # 右移操作
                extracted_bits.append(0)
            else:  # 左移操作
                extracted_bits.append(1)

        # 調試語句：輸出提取的比特流和嵌入的位置
        print(f"Pixel ({y}, {x}): Embed Order={embed_order}, Pixel Value={pixel}, Extracted Bit={extracted_bits[-1]}")

    return extracted_bits

def decode_and_recover(marked_img, orig_img, peak_value, peak_sequence, embedded_positions, hide_data, img_name):
    embedded_bits = np.load(os.path.join(HS_HIDE_DATA_PATH, f"{img_name}_embedded_bits.npy"))
    extracted_bits = extract_data(marked_img, peak_value, peak_sequence, embedded_positions)
    reverted_img = revert_data(marked_img, peak_value, peak_sequence, embedded_positions)

    print(f"Extracted bits: {extracted_bits}")
    print(f"Embedded bits: {embedded_bits}")
    print(f"Original embedded bits: {hide_data}")

    if np.array_equal(extracted_bits, embedded_bits):
        print("Extracted data matches embedded data.")
    else:
        print("Extracted data does not match embedded data.")

    if np.array_equal(embedded_bits, hide_data):
        print("Embedded data matches original data.")
    else:
        print("Embedded data does not match original data.")

    print(f"Length of extracted bits: {len(extracted_bits)}")
    print(f"Length of embedded bits: {len(embedded_bits)}")
    print(f"Length of original embedded bits: {len(hide_data)}")

    if len(extracted_bits) == len(embedded_bits) == len(hide_data):
        print("Length of extracted bits, embedded bits, and original embedded bits is the same.")
    else:
        print("Length of extracted bits, embedded bits, and original embedded bits is different.")
        print("WARNING: The length of the extracted bitstream does not match the original embedded bitstream.")

    psnr = calculate_psnr(orig_img, reverted_img)
    ssim = calculate_ssim(orig_img, reverted_img)

    print(f"PSNR between original and recovered image: {psnr:.2f}")
    print(f"SSIM between original and recovered image: {ssim:.6f}")

    cv2.imwrite(os.path.join(HS_MARKED_PATH, f"{img_name}_recovered.{FILE_TYPE}"), reverted_img)

    question_for_displaying_images = input("Display images? (y/n): ")
    if question_for_displaying_images == "y":
        cv2.imshow("Original", orig_img)
        cv2.imshow("Marked", marked_img)
        cv2.imshow("Recovered", reverted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    img_name = input("Image name: ")

    marked_img_path = os.path.join(HS_MARKED_PATH, f"{img_name}_shifted.{FILE_TYPE}")
    marked_img = cv2.imread(marked_img_path, cv2.IMREAD_GRAYSCALE)
    
    if marked_img is None:
        print(f"Error: Could not read the marked image file '{marked_img_path}'")
        return
    
    marked_img = cv2.cvtColor(marked_img, cv2.COLOR_GRAY2BGR)

    orig_img_path = os.path.join(HS_IMAGES_PATH, f"{img_name}.{FILE_TYPE}")
    orig_img = cv2.imread(orig_img_path, cv2.IMREAD_GRAYSCALE)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)

    peak_value = np.load(os.path.join(PEAK_PATH, f"{img_name}_peak.npy"))
    peak_sequence = np.load(os.path.join(HS_HIDE_DATA_PATH, f"{img_name}_peak_sequence.npy"))
    embedded_positions = np.load(os.path.join(HS_HIDE_DATA_PATH, f"{img_name}_embedded_positions.npy"))
    hide_data = np.load(os.path.join(HS_HIDE_DATA_PATH, f"{img_name}_HS_hide_data.npy"))

    decode_and_recover(marked_img, orig_img, peak_value, peak_sequence, embedded_positions, hide_data, img_name)

if __name__ == "__main__":
    main()