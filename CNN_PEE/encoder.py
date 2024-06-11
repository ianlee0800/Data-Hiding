import numpy as np
import cv2
import os
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim

def preprocess_image(image):
    # Divide the image into four parts
    h, w = image.shape
    part1 = image[0:h:2, 0:w:2]
    part2 = image[0:h:2, 1:w:2]
    part3 = image[1:h:2, 0:w:2]
    part4 = image[1:h:2, 1:w:2]

    # Create preprocessed images for each part
    preprocessed_parts = []
    for part in [part1, part2, part3, part4]:
        # Create a square weighted mean filter kernel
        filter_kernel = np.array([
            [1, 2, 3, 2, 1],
            [2, 4, 6, 4, 2],
            [3, 6, 9, 6, 3],
            [2, 4, 6, 4, 2],
            [1, 2, 3, 2, 1]
        ])
        filter_kernel = filter_kernel / np.sum(filter_kernel)

        # Apply the weighted mean filter to the part
        filtered_part = convolve2d(part, filter_kernel, mode='same')

        # Combine the filtered part with the other three parts
        preprocessed_part = np.zeros_like(image)
        preprocessed_part[0:h:2, 0:w:2] = filtered_part
        preprocessed_part[0:h:2, 1:w:2] = part2
        preprocessed_part[1:h:2, 0:w:2] = part3
        preprocessed_part[1:h:2, 1:w:2] = part4
        preprocessed_parts.append(preprocessed_part)

    # Choose one of the preprocessed parts randomly
    preprocessed_image = preprocessed_parts[np.random.randint(0, 4)]
    return preprocessed_image

def predict_image(image):
    # Use a simple non-deep learning prediction method
    # Replace this function with the CNN-based predictor once trained
    predicted_image = image.copy()
    return predicted_image

def pee_embedding(cover_image, secret_data):
    # Preprocess the cover image
    preprocessed_image = preprocess_image(cover_image)

    # Predict the preprocessed image
    predicted_image = predict_image(preprocessed_image)

    # Calculate the prediction errors
    prediction_errors = cover_image - predicted_image

    # Create a binary mask to store the positions of modified pixels
    modified_positions = np.zeros_like(cover_image, dtype=bool)

    # Perform PEE embedding
    stego_image = cover_image.copy()
    data_index = 0
    for i in range(cover_image.shape[0]):
        for j in range(cover_image.shape[1]):
            if data_index < len(secret_data):
                prediction_error = prediction_errors[i, j]
                if prediction_error >= 0:
                    stego_image[i, j] = predicted_image[i, j] + 2 * prediction_error + secret_data[data_index]
                    modified_positions[i, j] = True
                    data_index += 1
                else:
                    stego_image[i, j] = predicted_image[i, j] + 2 * prediction_error - secret_data[data_index]
                    modified_positions[i, j] = True
                    data_index += 1
            else:
                break

    return stego_image, modified_positions


def calculate_psnr(origin_image, stego_image):
    mse = np.mean((origin_image - stego_image) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(origin_image, stego_image):
    ssim_value = ssim(origin_image, stego_image, data_range=origin_image.max() - origin_image.min())
    return ssim_value

# Get user input for the cover image name
cover_image_name = input("Enter the cover image name (without extension): ")
cover_image_path = f"./CNN_PEE/origin/{cover_image_name}.png"

# Check if the cover image exists
if not os.path.isfile(cover_image_path):
    print(f"Error: Cover image '{cover_image_name}.png' not found in the './CNN_PEE/origin' directory.")
    exit(1)

# Example secret data
secret_data = [0, 1, 1, 0, 1, 0, 0, 1]

cover_image = cv2.imread(cover_image_path, cv2.IMREAD_GRAYSCALE)
stego_image, modified_positions = pee_embedding(cover_image, secret_data)

# Save the modified positions to a file
np.save(f"./CNN_PEE/stego/{cover_image_name}_modified_positions.npy", modified_positions)

# Calculate PSNR and SSIM
psnr_value = calculate_psnr(cover_image, stego_image)
ssim_value = calculate_ssim(cover_image, stego_image)

print(f"PSNR: {psnr_value:.2f} dB")
print(f"SSIM: {ssim_value:.4f}")

# Save the secret data to a file
with open(f"./CNN_PEE/stego/{cover_image_name}_secret_data.txt", "w") as file:
    file.write(",".join(map(str, secret_data)))

# Save the stego image with the modified name
stego_image_path = f"./CNN_PEE/stego/{cover_image_name}_stego.png"
cv2.imwrite(stego_image_path, stego_image)

print(f"Stego image saved as '{cover_image_name}_stego.png' in the './CNN_PEE/stego' directory.")