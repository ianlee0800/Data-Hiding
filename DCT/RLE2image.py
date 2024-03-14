import cv2
import numpy as np
import math

from zigzag import inverse_zigzag_scan

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


def read_encoded_bitstream(file_path: str) -> tuple[int, int, np.ndarray]:
    """
    Read the encoded bitstream from a file and return the image dimensions and encoded data.

    Args:
        file_path (str): The path to the file containing the encoded bitstream.

    Returns:
        tuple[int, int, np.ndarray]: A tuple containing the image height, width, and the encoded data array.
    """
    with open(file_path, 'r') as file:
        bitstream = file.read()

    # Split the bitstream into tokens separated by spaces
    details = bitstream.split()

    # Get the image height and width from the first two tokens
    height = int(''.join(filter(str.isdigit, details[0])))
    width = int(''.join(filter(str.isdigit, details[1])))

    # Create an array to store the encoded data
    encoded_data = np.zeros(height * width, dtype=int)

    k = 0
    i = 2
    while k < encoded_data.shape[0]:
        # Check if the bitstream has ended
        if details[i] == ';':
            break

        # Get the signed integer value from the token
        value = int(''.join(filter(str.isdigit, details[i])))
        if "-" in details[i]:
            value = -value
        encoded_data[k] = value

        # Get the skip value from the next token
        if i + 3 < len(details):
            skip = int(''.join(filter(str.isdigit, details[i + 3])))
            if skip == 0:
                k += 1
            else:
                k += skip + 1
            i += 2

    return height, width, encoded_data


def decode_image(encoded_data: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Decode the encoded data and reconstruct the image.

    Args:
        encoded_data (numpy.ndarray): The encoded data array.
        height (int): The height of the image.
        width (int): The width of the image.

    Returns:
        numpy.ndarray: The reconstructed image.
    """
    # Reshape the encoded data into a 2D array
    encoded_array = np.reshape(encoded_data, (height, width))

    # Initialize the padded image
    padded_image = np.zeros((height, width), dtype=np.float32)

    # Loop over the blocks and reconstruct the image
    for i in range(0, height, BLOCK_SIZE):
        for j in range(0, width, BLOCK_SIZE):
            block = encoded_array[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE]
            reordered_block = inverse_zigzag_scan(block.flatten(), BLOCK_SIZE, BLOCK_SIZE)
            dequantized_block = np.multiply(reordered_block, QUANTIZATION_MAT)
            padded_image[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE] = cv2.idct(dequantized_block)

    # Clamp the pixel values to the valid range
    padded_image[padded_image > 255] = 255
    padded_image[padded_image < 0] = 0

    return np.uint8(padded_image)


def main():
    # Read the encoded bitstream from the file
    height, width, encoded_data = read_encoded_bitstream('image.txt')

    # Decode the image
    decoded_image = decode_image(encoded_data, height, width)

    # Save the decoded image
    cv2.imwrite("decoded_image.bmp", decoded_image)


if __name__ == "__main__":
    main()