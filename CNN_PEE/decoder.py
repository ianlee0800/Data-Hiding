import numpy as np
import cv2
import os
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(origin_image, restored_image):
    mse = np.mean((origin_image - restored_image) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(origin_image, restored_image):
    ssim_value = ssim(origin_image, restored_image, data_range=origin_image.max() - origin_image.min())
    return ssim_value

def preprocess_image_decoding(image):
    # Divide the image into four parts
    h, w = image.shape
    part1 = image[0:h:2, 0:w:2]
    part2 = image[0:h:2, 1:w:2]
    part3 = image[1:h:2, 0:w:2]
    part4 = image[1:h:2, 1:w:2]

    # Choose one of the parts randomly
    selected_part_index = np.random.choice([0, 1, 2, 3])
    selected_part = [part1, part2, part3, part4][selected_part_index]

    # Create a square weighted mean filter kernel
    filter_kernel = np.array([
        [1, 2, 3, 2, 1],
        [2, 4, 6, 4, 2],
        [3, 6, 9, 6, 3],
        [2, 4, 6, 4, 2],
        [1, 2, 3, 2, 1]
    ])
    filter_kernel = filter_kernel / np.sum(filter_kernel)

    # Apply the weighted mean filter to the selected part
    preprocessed_part = convolve2d(selected_part, filter_kernel, mode='same')

    # Combine the preprocessed part with the other three parts
    preprocessed_image = image.copy()
    if selected_part_index == 0:
        preprocessed_image[0:h:2, 0:w:2] = preprocessed_part
    elif selected_part_index == 1:
        preprocessed_image[0:h:2, 1:w:2] = preprocessed_part
    elif selected_part_index == 2:
        preprocessed_image[1:h:2, 0:w:2] = preprocessed_part
    else:
        preprocessed_image[1:h:2, 1:w:2] = preprocessed_part

    return preprocessed_image

def predict_image_decoding(image):
    # Use a simple averaging of neighboring pixels for prediction
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) / 8
    predicted_image = convolve2d(image, kernel, mode='same')
    return predicted_image

def pee_extracting(stego_image, secret_data_length):
    preprocessed_image = preprocess_image_decoding(stego_image)
    predicted_image = predict_image_decoding(preprocessed_image)

    extracted_data = []
    data_index = 0
    for i in range(stego_image.shape[0]):
        for j in range(stego_image.shape[1]):
            if data_index < secret_data_length:
                prediction_error = stego_image[i, j].astype(int) - predicted_image[i, j].astype(int)
                if prediction_error >= 0:
                    extracted_bit = prediction_error % 2
                    extracted_data.append(extracted_bit)
                    data_index += 1
                else:
                    extracted_bit = (-prediction_error) % 2
                    extracted_data.append(extracted_bit)
                    data_index += 1
            else:
                break

    restored_image = stego_image.copy()
    for i in range(stego_image.shape[0]):
        for j in range(stego_image.shape[1]):
            prediction_error = stego_image[i, j].astype(int) - predicted_image[i, j].astype(int)
            if prediction_error >= 0:
                restored_image[i, j] = predicted_image[i, j] + prediction_error // 2
            else:
                restored_image[i, j] = predicted_image[i, j] + (prediction_error + 1) // 2

    return np.array(extracted_data), restored_image

# Get user input for the stego image name
stego_image_name = input("Enter the stego image name (without extension): ")
stego_image_path = f"./CNN_PEE/stego/{stego_image_name}_stego.png"

# Check if the stego image exists
if not os.path.isfile(stego_image_path):
    print(f"Error: Stego image '{stego_image_name}.png' not found in the './CNN_PEE/stego' directory.")
    exit(1)

# Read the secret data from the file
secret_data_file = f"./CNN_PEE/stego/{stego_image_name.rsplit('_', 1)[0]}_secret_data.txt"
with open(secret_data_file, "r") as file:
    secret_data = list(map(int, file.read().split(",")))

stego_image = cv2.imread(stego_image_path, cv2.IMREAD_GRAYSCALE)
extracted_data, restored_image = pee_extracting(stego_image, len(secret_data))

# Calculate PSNR and SSIM between the restored image and the original image
origin_image_path = f"./CNN_PEE/origin/{stego_image_name.rsplit('_', 1)[0]}.png"
origin_image = cv2.imread(origin_image_path, cv2.IMREAD_GRAYSCALE)
psnr_value = calculate_psnr(origin_image, restored_image)
ssim_value = calculate_ssim(origin_image, restored_image)

print(f"PSNR between the restored image and the original image: {psnr_value:.2f} dB")
print(f"SSIM between the restored image and the original image: {ssim_value:.4f}")

# Compare the extracted data with the original secret data
if np.array_equal(secret_data, extracted_data):
    print("The extracted data matches the original secret data.")
else:
    print("The extracted data does not match the original secret data.")

# Save the extracted data to a file
extracted_data_file = f"./CNN_PEE/extracted/{stego_image_name.rsplit('_', 1)[0]}_extracted_data.txt"
with open(extracted_data_file, "w") as file:
    file.write(",".join(map(str, extracted_data)))

print(f"Extracted data saved as '{stego_image_name.rsplit('_', 1)[0]}_extracted_data.txt' in the './CNN_PEE/extracted' directory.")

# Save the restored image
restored_image_path = f"./CNN_PEE/restored/{stego_image_name.rsplit('_', 1)[0]}_restored.png"
cv2.imwrite(restored_image_path, restored_image)

print(f"Restored image saved as '{stego_image_name.rsplit('_', 1)[0]}_restored.png' in the './CNN_PEE/restored' directory.")