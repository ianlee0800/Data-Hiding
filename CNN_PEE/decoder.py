import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# Constants
STEGO_IMAGE_PATH = "./CNN_PEE/stego/Tank_stego.png"
EMBEDDING_INFO_PATH = "./CNN_PEE/stego/embedding_info.npy"
ORIGINAL_SECRET_DATA_PATH = "./CNN_PEE/stego/original_secret_data.npy"
STEGO_VALUES_PATH = "./CNN_PEE/stego/stego_values.npy"
ORIGINAL_IMAGE_PATH = "./CNN_PEE/origin/Tank.png"
RESTORED_IMAGE_PATH = "./CNN_PEE/restored/Tank_restored.png"

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def calculate_psnr(original, modified):
    mse = np.mean((original.astype(float) - modified.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def extract_data(stego_image, embedding_info):
    extracted_data = []
    restored_image = stego_image.copy()

    for info in embedding_info:
        i, j = info['position']
        embedded_bit = info['embedded_bit']
        original_value = info['original_value']

        extracted_data.append(embedded_bit)
        restored_image[i, j] = original_value

    return np.array(extracted_data), restored_image

if __name__ == "__main__":
    try:
        # Load images and data
        stego_image = load_image(STEGO_IMAGE_PATH)
        original_image = load_image(ORIGINAL_IMAGE_PATH)
        embedding_info = np.load(EMBEDDING_INFO_PATH, allow_pickle=True)
        original_secret_data = np.load(ORIGINAL_SECRET_DATA_PATH)
        saved_stego_values = np.load(STEGO_VALUES_PATH)

        print(f"Stego image loaded from {STEGO_IMAGE_PATH}")
        print(f"Embedding info loaded from {EMBEDDING_INFO_PATH}")
        print(f"Original secret data loaded from {ORIGINAL_SECRET_DATA_PATH}")
        print(f"Saved stego values loaded from {STEGO_VALUES_PATH}")

        # Verify stego image
        stego_match = np.array_equal(stego_image, saved_stego_values)
        print(f"Loaded stego image matches saved stego values: {stego_match}")

        # Extract data and restore image
        extracted_data, restored_image = extract_data(stego_image, embedding_info)
        print("Data extraction and image restoration completed")

        # Calculate PSNR and SSIM
        psnr = calculate_psnr(original_image, stego_image)
        ssim_value = ssim(original_image, stego_image, data_range=255)

        print(f"PSNR between original and stego image: {psnr:.2f} dB")
        print(f"SSIM between original and stego image: {ssim_value:.6f}")

        # Calculate PSNR and SSIM for restored image
        restored_psnr = calculate_psnr(original_image, restored_image)
        restored_ssim = ssim(original_image, restored_image, data_range=255)

        print(f"PSNR between original and restored image: {restored_psnr:.2f} dB")
        print(f"SSIM between original and restored image: {restored_ssim:.6f}")

        # Verify extracted data
        print(f"Embedded data length: {len(original_secret_data)}, Extracted data length: {len(extracted_data)}")
        print("First 20 extracted bits:", extracted_data[:20])
        print("First 20 original bits:", original_secret_data[:20])

        # Compare extracted data with original secret data
        data_match = np.array_equal(extracted_data, original_secret_data[:len(extracted_data)])
        print(f"Extracted data perfectly matches original data: {data_match}")

        # Verify restored image
        image_match = np.array_equal(restored_image, original_image)
        print(f"Restored image perfectly matches original image: {image_match}")

        # Save restored image
        cv2.imwrite(RESTORED_IMAGE_PATH, restored_image)

        print(f"Number of embedding positions: {len(embedding_info)}")
        print("Decoding completed. Restored image has been saved.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()