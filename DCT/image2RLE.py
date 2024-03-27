import cv2
import numpy as np
import math
import os

# Import zigzag functions
from zigzag import zigzag_scan, inverse_zigzag_scan

QUANTIZATION_MAT = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

BLOCK_SIZE = 8

def get_padded_image(image: np.ndarray) -> np.ndarray:
    """
    Pad the input image to a size that is a multiple of the block size.

    Args:
        image (numpy.ndarray): The input grayscale image.

    Returns:
        numpy.ndarray: The padded image.
    """
    height, width = image.shape

    # Calculate the number of blocks needed in height and width
    num_blocks_height = math.ceil(height / BLOCK_SIZE)
    num_blocks_width = math.ceil(width / BLOCK_SIZE)

    # Calculate the padded height and width
    padded_height = num_blocks_height * BLOCK_SIZE
    padded_width = num_blocks_width * BLOCK_SIZE

    # Create a padded image with zeros
    padded_image = np.zeros((padded_height, padded_width), dtype=np.uint8)

    # Copy the values of the input image into the padded image
    padded_image[:height, :width] = image

    return padded_image


def get_block_indices(height: int, width: int) -> tuple:
    """
    Generate the row and column indices for blocks in the image.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.

    Returns:
        tuple: A tuple containing the row and column indices for blocks.
    """
    num_blocks_height = math.ceil(height / BLOCK_SIZE)
    num_blocks_width = math.ceil(width / BLOCK_SIZE)

    row_indices = []
    col_indices = []

    for i in range(num_blocks_height):
        row_start = i * BLOCK_SIZE
        row_end = row_start + BLOCK_SIZE
        row_indices.append((row_start, row_end))

    for j in range(num_blocks_width):
        col_start = j * BLOCK_SIZE
        col_end = col_start + BLOCK_SIZE
        col_indices.append((col_start, col_end))

    return row_indices, col_indices


def get_run_length_encoding(image: np.ndarray) -> str:
    """
    Perform run-length encoding on the input image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        str: The run-length encoded bitstream.
    """
    i = 0
    skip = 0
    bitstream = ""
    image = image.astype(int)

    while i < image.shape[0]:
        if image[i] != 0:
            bitstream += f"{image[i]} {skip} "
            skip = 0
        else:
            skip += 1
        i += 1

    return bitstream


def encode_image(image: np.ndarray) -> tuple[str, np.ndarray]:
    """
    Encode the input image using DCT and run-length encoding.

    Args:
        image (numpy.ndarray): The input grayscale image.

    Returns:
        str: The encoded bitstream.
    """
    # Pad the image to a size that is a multiple of the block size
    padded_image = get_padded_image(image)

    # Get the row and column indices for blocks
    row_indices, col_indices = get_block_indices(padded_image.shape[0], padded_image.shape[1])

    # Start encoding
    for row_start, row_end in row_indices:
        for col_start, col_end in col_indices:
            block = padded_image[row_start:row_end, col_start:col_end]

            # Apply 2D discrete cosine transform
            dct_block = cv2.dct(block.astype(np.float32))

            # Keep only the top 8 coefficients by absolute value
            abs_dct_block = np.abs(dct_block)
            threshold = np.sort(abs_dct_block, axis=None)[-8]
            quantized_block = np.where(abs_dct_block >= threshold, dct_block, 0)

            # Reorder DCT coefficients in zigzag order
            reordered_block = zigzag_scan(quantized_block)

            # Reshape the reordered block back to (block_size, block_size)
            reshaped_block = np.reshape(reordered_block, (BLOCK_SIZE, BLOCK_SIZE))

            # Copy the reshaped block into the padded image
            padded_image[row_start:row_end, col_start:col_end] = reshaped_block

    arranged_data = padded_image.flatten()

    # Perform run-length encoding
    bitstream = get_run_length_encoding(arranged_data)

    # Add image size information to the bitstream
    bitstream = f"{padded_image.shape[0]} {padded_image.shape[1]} {bitstream};"

    return bitstream, padded_image

import os

def main():
    # Read the input image in grayscale
    image_path = r".\\DCT\\harry.jpg"
    if not os.path.isfile(image_path):
        print(f"Error: '{image_path}' file not found.")
        return

    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        print(f"Error: Unable to read '{image_path}'.")
        return

    # Encode the image
    encoded_bitstream, padded_image = encode_image(input_image)

    # Write the encoded bitstream to a file
    with open("image.txt", "w") as file:
        file.write(encoded_bitstream)

    # Get the image name from the path
    image_name, _ = os.path.splitext(os.path.basename(image_path))

    # Save the encoded image with a specific name
    output_image_path = f"{image_name}_DCT.bmp"
    cv2.imwrite(output_image_path, np.uint8(padded_image))

    cv2.imshow('Encoded Image', np.uint8(padded_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()