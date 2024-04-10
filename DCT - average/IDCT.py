import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# import zigzag functions
from zigzag import *

# defining block size
block_size = 8

# defining block size
block_size = 8

# Quantization Matrix 
QUANTIZATION_MAT = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                             [12, 12, 14, 19, 26, 58, 60, 55],
                             [14, 13, 16, 24, 40, 57, 69, 56],
                             [14, 17, 22, 29, 51, 87, 80, 62],
                             [18, 22, 37, 56, 68, 109, 103, 77],
                             [24, 35, 55, 64, 81, 104, 113, 92],
                             [49, 64, 78, 87, 103, 121, 120, 101],
                             [72, 92, 95, 98, 112, 100, 103, 99]])

def adaptive_quantization(block, quality_factor):
    # Compute the adaptive quantization matrix based on the block's characteristics
    block_variance = np.var(block)
    if block_variance < 100:
        scale_factor = 1.0
    elif block_variance < 500:
        scale_factor = 0.8
    else:
        scale_factor = 0.6
    
    quantization_matrix = np.floor((QUANTIZATION_MAT * scale_factor * (quality_factor / 100))).astype(int)
    return quantization_matrix

def decode_run_length_encoding(encoded_data, expected_size):
    decoded_data = []
    data_pairs = encoded_data.split()
    
    for i in range(0, len(data_pairs), 2):
        value = int(data_pairs[i])
        count = int(data_pairs[i+1])
        
        decoded_data.extend([value] * count)
    
    # Truncate or pad the decoded data to match the expected size
    if len(decoded_data) > expected_size:
        decoded_data = decoded_data[:expected_size]
    elif len(decoded_data) < expected_size:
        decoded_data.extend([0] * (expected_size - len(decoded_data)))
    
    return np.array(decoded_data, dtype=np.int32)

# read the compressed data from the file
with open("compressed_data.txt", "r") as file:
    dimensions = file.readline().split()
    h, w = int(dimensions[0]), int(dimensions[1])
    rle_encoded = file.read()

# decode the run-length encoded data
expected_size = h * w
decoded_data = decode_run_length_encoding(rle_encoded, expected_size)

# reshape the decoded data to the original image dimensions
array = np.reshape(decoded_data, (h, w))

# initialize the padded image
padded_img = np.zeros((h, w), dtype=np.float64)

# define the quality factor (0-100)
quality_factor = 80

# iterate over the blocks and apply inverse DCT
for i in range(0, h, block_size):
    for j in range(0, w, block_size):
        block = array[i:i+block_size, j:j+block_size]
        
        # apply inverse zigzag scan to the block
        block = inverse_zigzag(block.flatten(), block_size, block_size)
        
        # perform adaptive de-quantization
        quantization_matrix = adaptive_quantization(block, quality_factor)
        de_quantized = block * quantization_matrix
        
        # apply inverse DCT to the de-quantized block
        padded_img[i:i+block_size, j:j+block_size] = cv2.idct(de_quantized)

# clip the pixel values to the range [0, 255]
padded_img = np.clip(padded_img, 0, 255)

# convert the image to uint8 data type
padded_img = padded_img.astype(np.uint8)

# save the decompressed image
cv2.imwrite("decompressed_image.bmp", padded_img)

# load the uncompressed and decompressed images
uncompressed_img = cv2.imread("uncompressed.bmp", cv2.IMREAD_GRAYSCALE)
decompressed_img = cv2.imread("decompressed_image.bmp", cv2.IMREAD_GRAYSCALE)

# calculate PSNR
mse = np.mean((uncompressed_img - decompressed_img) ** 2)
if mse == 0:
    psnr = float('inf')
else:
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

# calculate SSIM
ssim_value = ssim(uncompressed_img, decompressed_img)

print("PSNR: {:.2f} dB".format(psnr))
print("SSIM: {:.4f}".format(ssim_value))