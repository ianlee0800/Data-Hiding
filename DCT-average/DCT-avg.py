import os
from PIL import Image
import numpy as np
from scipy import fftpack

def get_image(image_path='./DCT-average/images/boat.png'):
    image = Image.open(image_path)
    img_grey = image.convert('L')
    img = np.array(img_grey, dtype=np.float64)
    return img, image

def get_8x8_dct(block):
    return fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')

def get_8x8_idct(coefficients):
    return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')

def get_reconstructed_image(raw):
    img = raw.clip(0, 255)
    img = img.astype('uint8')
    img = Image.fromarray(img)
    return img

if __name__ == "__main__":
    pixels, original_image = get_image()
    block_size = 8
    image_height, image_width = pixels.shape

    # Initialize an array to store the DCT coefficients
    dct_coeffs = np.zeros((8, 8))

    # Initialize a counter to keep track of the number of blocks
    block_count = 0

    for i in range(0, pixels.shape[0], block_size):
        for j in range(0, pixels.shape[1], block_size):
            block = pixels[i:i+block_size, j:j+block_size]
            dct_block = get_8x8_dct(block)

            # Accumulate the DCT coefficients
            dct_coeffs += dct_block

            # Increment the block counter
            block_count += 1

    # Calculate the average DCT coefficients
    dct_coeffs /= block_count

    # Create a new image using the average DCT coefficients
    reconstructed_image = np.zeros((image_height, image_width), dtype=np.float64)
    for i in range(0, image_height, block_size):
        for j in range(0, image_width, block_size):
            reconstructed_block = get_8x8_idct(dct_coeffs)
            reconstructed_image[i:i+block_size, j:j+block_size] = reconstructed_block

    reconstructed_image = get_reconstructed_image(reconstructed_image)

    print("Average DCT coefficients:")
    for row in dct_coeffs:
        print([f"{val:.2f}" for val in row])

    # Save the original and reconstructed images
    save_path = './DCT-average/reconstructed_images/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    original_image.save(os.path.join(save_path, 'boat_original.png'))
