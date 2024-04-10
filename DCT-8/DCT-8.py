import os
from PIL import Image
import numpy as np
from scipy import fftpack
import skimage.metrics

def get_image(image_path='./DCT-8/images/lena.png'):
    image = Image.open(image_path)
    img_grey = image.convert('L')
    img = np.array(img_grey, dtype=np.float64)
    return img

def get_8x8_dct(block):
    return fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')

def get_8x8_idct(coefficients):
    return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')

def zigzag_scan(coefficients):
    zz = np.concatenate([np.diagonal(coefficients[::-1,:], i)[::(2*(i%2)-1)] for i in range(1-coefficients.shape[0], coefficients.shape[0])])
    return zz

def get_reconstructed_image(raw):
    img = raw.clip(0, 255)
    img = img.astype('uint8')
    img = Image.fromarray(img)
    return img

if __name__ == "__main__":
    original_image = get_image()
    pixels = original_image.copy()
    block_size = 8
    reconstructed_images = []

    for i in range(0, pixels.shape[0], block_size):
        for j in range(0, pixels.shape[1], block_size):
            block = pixels[i:i+block_size, j:j+block_size]
            dct_block = get_8x8_dct(block)

            # Perform zigzag scan and keep top-k coefficients
            zigzag_coeffs = zigzag_scan(dct_block)
            k = 8  # Number of coefficients to keep
            zigzag_coeffs[k:] = 0

            # Reconstruct the block from zigzag coefficients
            dct_block = np.zeros_like(dct_block)
            sorted_indices = np.unravel_index(np.argsort(np.abs(dct_block), axis=None), dct_block.shape)
            dct_block[sorted_indices[0][:k], sorted_indices[1][:k]] = zigzag_coeffs[:k]

            reconstructed_block = get_8x8_idct(dct_block)
            pixels[i:i+block_size, j:j+block_size] = reconstructed_block

    reconstructed_image = get_reconstructed_image(pixels)
    reconstructed_images.append(reconstructed_image)

    save_path = './DCT-8/reconstructed_images/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    reconstructed_images[0].save(os.path.join(save_path, 'lena_reconstructed.jpg'))

    # Calculate PSNR and SSIM
    original_image = np.array(original_image, dtype=np.uint8)
    reconstructed_image = np.array(reconstructed_images[0])
    psnr = skimage.metrics.peak_signal_noise_ratio(original_image, reconstructed_image)
    ssim = skimage.metrics.structural_similarity(original_image, reconstructed_image, multichannel=True)

    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")