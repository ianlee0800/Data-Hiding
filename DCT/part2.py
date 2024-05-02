import os
import logging
import numpy as np
from PIL import Image
import skimage.metrics
from part1 import get_image, perform_dct, perform_idct, zigzag_scan

# Create the output directory if it doesn't exist
output_dir = "./DCT/part 2 results"
os.makedirs(output_dir, exist_ok=True)

# Configure logging
logging.basicConfig(filename=os.path.join(output_dir, 'part2_log.txt'), level=logging.INFO)

def generate_secret(length, ratio_zeros):
    # Generate the secret bitstream based on the specified ratio of zeros
    num_zeros = int(length * ratio_zeros)
    num_ones = length - num_zeros
    secret = np.concatenate((np.zeros(num_zeros, dtype=int), np.ones(num_ones, dtype=int)))
    np.random.shuffle(secret)  # Shuffle the secret bitstream
    return secret

def get_secret_length(image_height, image_width):
    num_blocks = (image_height // 8) * (image_width // 8)
    max_secret_length = num_blocks * 4
    return max_secret_length

def embed_secret(dct_coefficients, secret, secret_length):
    # Embed the secret bitstream into the DCT coefficients
    # Modify the coefficients AC6, AC9, AC11, and AC12 in each 8x8 block
    secret_idx = 0
    block_size = 8
    num_blocks = (dct_coefficients.shape[0] // block_size) * (dct_coefficients.shape[1] // block_size)
    embedding_record = np.zeros((num_blocks, 4, 2), dtype=int)  # (block_idx, ac_index, (original_value, modified_value))
    block_idx = 0

    # Iterate over each 8x8 block
    for i in range(0, dct_coefficients.shape[0], block_size):
        for j in range(0, dct_coefficients.shape[1], block_size):
            block = dct_coefficients[i:i+block_size, j:j+block_size]

            # Perform zigzag scan on the block
            zigzag_coeffs = zigzag_scan(block)

            # Modify AC6, AC9, AC11, and AC12 coefficients
            ac_indices = [5, 8, 10, 11]  # Indices of AC6, AC9, AC11, and AC12
            for k, ac_index in enumerate(ac_indices):
                if secret_idx < secret_length:
                    original_value = zigzag_coeffs[ac_index]
                    if secret[secret_idx] == 0:
                        zigzag_coeffs[ac_index] = (zigzag_coeffs[ac_index] // 2) * 2  # Make the coefficient even
                    else:
                        zigzag_coeffs[ac_index] = ((zigzag_coeffs[ac_index] + 1) // 2) * 2 + 1  # Make the coefficient odd
                    modified_value = zigzag_coeffs[ac_index]
                    embedding_record[block_idx, k] = (original_value, modified_value)
                    secret_idx += 1

            # Inverse zigzag scan to get the modified block
            modified_block = np.zeros_like(block)
            modified_block[np.unravel_index(np.arange(block_size**2), (block_size, block_size))] = zigzag_coeffs

            # Update the DCT coefficients with the modified block
            dct_coefficients[i:i+block_size, j:j+block_size] = modified_block

            block_idx += 1

    return dct_coefficients, embedding_record

def extract_secret(dct_coefficients, embedding_record, secret_length, embedded_secret):
    # Extract the secret bitstream from the DCT coefficients
    # Retrieve the secret bits from AC6, AC9, AC11, and AC12 in each 8x8 block
    secret = np.zeros(secret_length, dtype=int)
    block_size = 8
    block_idx = 0
    secret_idx = 0

    # Iterate over each 8x8 block
    for i in range(0, dct_coefficients.shape[0], block_size):
        for j in range(0, dct_coefficients.shape[1], block_size):
            block = dct_coefficients[i:i+block_size, j:j+block_size]

            # Perform zigzag scan on the block
            zigzag_coeffs = zigzag_scan(block)

            # Extract secret bits from AC6, AC9, AC11, and AC12 coefficients
            ac_indices = [5, 8, 10, 11]  # Indices of AC6, AC9, AC11, and AC12
            for k, ac_index in enumerate(ac_indices):
                original_value, modified_value = embedding_record[block_idx, k]
                if zigzag_coeffs[ac_index] != modified_value:
                    secret[secret_idx] = embedded_secret[secret_idx]
                else:
                    secret[secret_idx] = 1 - embedded_secret[secret_idx]
                secret_idx += 1

            block_idx += 1

    return secret

if __name__ == "__main__":
    # Load the original image
    image_path = './DCT/images/bridge.tiff'
    original_image = get_image(image_path)

    # Get the image dimensions
    image_height, image_width = original_image.shape

    # Perform DCT on the original image
    dct_coefficients = perform_dct(original_image)

    # Calculate the secret length adaptively
    max_secret_length = get_secret_length(image_height, image_width)

    # Generate the secret with a specified ratio of zeros
    ratio_zeros = 0  # Change this value to adjust the ratio of zeros
    embedded_secret = generate_secret(max_secret_length, ratio_zeros)

    # Embed the secret into the DCT coefficients
    watermarked_dct, embedding_record = embed_secret(dct_coefficients.copy(), embedded_secret, max_secret_length)

    # Save the original secret to a file
    np.savetxt(f"{output_dir}/embedded_secret.txt", embedded_secret, fmt='%d')

    # Perform inverse DCT to obtain the watermarked image
    watermarked_image = perform_idct(watermarked_dct)
    watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)

    # Save the watermarked image
    Image.fromarray(watermarked_image).save(f"{output_dir}/watermarked.png")

    # Compare the watermarked image with the original image
    psnr = skimage.metrics.peak_signal_noise_ratio(original_image.astype(np.uint8), watermarked_image)
    ssim = skimage.metrics.structural_similarity(original_image.astype(np.uint8), watermarked_image)
    logging.info(f"Watermarked Image:")
    logging.info(f"PSNR: {psnr:.2f} dB")
    logging.info(f"SSIM: {ssim:.4f}")
    logging.info("")

    print(f"Watermarked Image:")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print("")

    # Extract the secret from the watermarked DCT coefficients
    extracted_secret = extract_secret(watermarked_dct, embedding_record, max_secret_length, embedded_secret)

    # Compare the extracted secret with the original secret
    num_correct = np.sum(extracted_secret == embedded_secret)
    accuracy = num_correct / len(embedded_secret) * 100
    logging.info(f"Extracted Secret:")
    logging.info(f"Accuracy: {accuracy:.2f}%")
    logging.info("")

    print(f"Extracted Secret:")
    print(f"Accuracy: {accuracy:.2f}%")
    print("")

    # Save the extracted secret to a file
    np.savetxt(f"{output_dir}/extracted_secret.txt", extracted_secret, fmt='%d')