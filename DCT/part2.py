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
    
    # Clip the modified DCT coefficients to a reasonable range
    dct_coefficients = np.clip(dct_coefficients, -1000, 1000)

    return dct_coefficients, embedding_record

def extract_secret(dct_coefficients, embedding_record, secret_length, embedded_secret):
    # Extract the secret bitstream from the DCT coefficients
    # Retrieve the secret bits from AC6, AC9, AC11, and AC12 in each 8x8 block
    secret = np.zeros(secret_length, dtype=int)
    block_size = 8
    block_idx = 0
    secret_idx = 0
    
    # Clip the DCT coefficients to the same range used during embedding
    dct_coefficients = np.clip(dct_coefficients, -1000, 1000)

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

    # Generate secrets
    secrets = {
        'random': generate_secret(max_secret_length, 0.5),
        'all_zeros': generate_secret(max_secret_length, 1.0),
        'all_ones': generate_secret(max_secret_length, 0.0)
    }
    secret_types = list(secrets.keys())  # Convert dict_keys to a list

    # Embed secrets into the DCT coefficients
    watermarked_dcts = {}
    embedding_records = {}
    for secret_type, secret in secrets.items():
        watermarked_dct, embedding_record = embed_secret(dct_coefficients.copy(), secret, max_secret_length)
        watermarked_dcts[secret_type] = watermarked_dct
        embedding_records[secret_type] = embedding_record

        # Save the embedded secret to a file
        embedded_secret_filename = f"{output_dir}/bridge-w{secret_types.index(secret_type) + 1}_embedded_secret.txt"
        np.savetxt(embedded_secret_filename, secret, fmt='%d')

        # Save the embedding record to a file
        embedding_record_filename = f"{output_dir}/bridge-w{secret_types.index(secret_type) + 1}_embedding_record.npy"
        np.save(embedding_record_filename, embedding_record)

    # Perform inverse DCT to obtain the watermarked images
    for secret_type, watermarked_dct in watermarked_dcts.items():
        watermarked_image = perform_idct(watermarked_dct)
        watermarked_image += 128  # Level shift
        watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)

        # Save the watermarked image
        watermarked_image_filename = f"{output_dir}/bridge-w{secret_types.index(secret_type) + 1}.tiff"
        Image.fromarray(watermarked_image).save(watermarked_image_filename)

    # Extract and compare secrets
    for secret_type, watermarked_dct in watermarked_dcts.items():
        embedding_record = embedding_records[secret_type]
        embedded_secret_filename = f"{output_dir}/bridge-w{secret_types.index(secret_type) + 1}_embedded_secret.txt"
        embedded_secret = np.loadtxt(embedded_secret_filename, dtype=int)
        extracted_secret = extract_secret(watermarked_dct, embedding_record, max_secret_length, embedded_secret)

        # Compare the extracted secret with the original secret
        num_correct = np.sum(extracted_secret == embedded_secret)
        accuracy = num_correct / len(embedded_secret) * 100
        logging.info(f"Extracted Secret ({secret_type}):")
        logging.info(f"Accuracy: {accuracy:.2f}%")
        logging.info("")

        print(f"Extracted Secret ({secret_type}):")
        print(f"Accuracy: {accuracy:.2f}%")
        print("")

        # Save the extracted secret to a file
        extracted_secret_filename = f"{output_dir}/bridge-w{secret_types.index(secret_type) + 1}_extracted_secret.txt"
        np.savetxt(extracted_secret_filename, extracted_secret, fmt='%d')