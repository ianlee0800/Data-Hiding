import os
import logging
from PIL import Image
import numpy as np
import skimage.metrics
import matplotlib.pyplot as plt

# Create the output directory if it doesn't exist
output_dir = "./DCT/part 1 results"
os.makedirs(output_dir, exist_ok=True)

# Configure logging
logging.basicConfig(filename=os.path.join(output_dir, 'part1_log.txt'), level=logging.INFO)

def dct_1d(x):
    N = len(x)
    y = np.zeros(N)
    for k in range(N):
        for n in range(N):
            y[k] += x[n] * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
        if k == 0:
            y[k] *= np.sqrt(1 / N)
        else:
            y[k] *= np.sqrt(2 / N)
    return y

def idct_1d(y):
    N = len(y)
    x = np.zeros(N)
    for n in range(N):
        for k in range(N):
            if k == 0:
                x[n] += y[k] * np.sqrt(1 / N) * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
            else:
                x[n] += y[k] * np.sqrt(2 / N) * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    return x

def dct_2d(image):
    M, N = image.shape
    x, y = np.meshgrid(range(M), range(N))
    
    print("Calculating DCT coefficients...")
    
    cx = np.sqrt(1 / M) * np.cos(np.pi * (2 * x + 1) * np.arange(M)[:, np.newaxis] / (2 * M))
    cy = np.sqrt(1 / N) * np.cos(np.pi * (2 * y + 1) * np.arange(N) / (2 * N))
    
    cx[1:, :] *= np.sqrt(2)
    cy[:, 1:] *= np.sqrt(2)
    
    dct_coefficients = cx.T.dot(image).dot(cy)
    
    print("DCT coefficients calculated.")
    
    return dct_coefficients

def idct_2d(coefficients):
    M, N = coefficients.shape
    x, y = np.meshgrid(range(M), range(N))
    
    print("Calculating reconstructed image...")
    
    cx = np.sqrt(1 / M) * np.cos(np.pi * (2 * np.arange(M)[:, np.newaxis] + 1) * x / (2 * M))
    cy = np.sqrt(1 / N) * np.cos(np.pi * (2 * np.arange(N) + 1) * y / (2 * N))
    
    cx[:, 1:] *= np.sqrt(2)
    cy[1:, :] *= np.sqrt(2)
    
    reconstructed_image = cx.dot(coefficients).dot(cy.T)
    
    print("Reconstructed image calculated.")
    
    return reconstructed_image

def get_dct_basis():
    basis = np.zeros((8, 8, 8, 8))
    for i in range(8):
        for j in range(8):
            basis_vec_i = np.zeros(8)
            basis_vec_i[i] = 1
            basis_vec_j = np.zeros(8)
            basis_vec_j[j] = 1
            basis[i, j] = np.outer(dct_1d(basis_vec_i), dct_1d(basis_vec_j))
    return basis

def visualize_dct_basis(basis):
    for i in range(8):
        for j in range(8):
            plt.subplot(8, 8, i * 8 + j + 1)
            plt.imshow(basis[i, j], cmap='gray')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('dct_basis.png')
    plt.close()

def get_image(image_path, target_size=None):
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    img_grey = image.convert('L')
    if target_size is not None:
        img_grey = img_grey.resize(target_size)
    img = np.array(img_grey, dtype=np.float64)
    return img

def perform_dct(image):
    print("Performing DCT...")
    image = image.astype(np.float32)
    image /= 255.0
    dct_coefficients = dct_2d(image)
    print("DCT completed.")
    return dct_coefficients

def perform_idct(coefficients):
    print("Performing IDCT...")
    reconstructed_image = idct_2d(coefficients)
    reconstructed_image *= 255.0
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    print("IDCT completed.")
    return reconstructed_image

def level_shift(image):
    shifted_image = image - 128
    return shifted_image

def quantize(coefficients, quantization_matrix):
    quantized_coefficients = coefficients / quantization_matrix
    return quantized_coefficients

def zigzag_scan(coefficients):
    zz = np.concatenate([np.diagonal(coefficients[::-1, :], i)[::(2*(i%2)-1)] for i in range(1-coefficients.shape[0], coefficients.shape[0])])
    return zz

def zonal_coding(coefficients, num_coeffs):
    M, N = coefficients.shape
    reconstructed_coefficients = np.zeros_like(coefficients)
    block_size = 8
    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            block = coefficients[i:i+block_size, j:j+block_size]
            zigzag_coeffs = zigzag_scan(block)
            zigzag_coeffs[num_coeffs:] = 0
            reconstructed_block = np.zeros_like(block)
            reconstructed_block[np.unravel_index(np.arange(num_coeffs), (block_size, block_size))] = zigzag_coeffs[:num_coeffs]
            reconstructed_coefficients[i:i+block_size, j:j+block_size] = reconstructed_block
    return reconstructed_coefficients

def threshold_coding(coefficients, num_coeffs):
    M, N = coefficients.shape
    reconstructed_coefficients = np.zeros_like(coefficients)
    block_size = 8
    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            block = coefficients[i:i+block_size, j:j+block_size]
            zigzag_coeffs = zigzag_scan(block)
            sorted_indices = np.argsort(np.abs(zigzag_coeffs))[::-1]
            top_k_indices = sorted_indices[:num_coeffs]
            reconstructed_block = np.zeros_like(block)
            reconstructed_block[np.unravel_index(top_k_indices, (block_size, block_size))] = zigzag_coeffs[top_k_indices]
            reconstructed_coefficients[i:i+block_size, j:j+block_size] = reconstructed_block
    return reconstructed_coefficients

def calculate_energy_distribution(coefficients, block_size=8):
    M, N = coefficients.shape
    num_blocks = (M // block_size) * (N // block_size)
    energy_per_block = np.zeros(num_blocks)
    total_energy = 0
    block_index = 0
    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            block = coefficients[i:i+block_size, j:j+block_size]
            energy = np.sum(block ** 2)
            energy_per_block[block_index] = energy
            total_energy += energy
            block_index += 1
    energy_distribution = energy_per_block / total_energy
    return energy_distribution

def plot_energy_distribution(energy_distribution):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, len(energy_distribution) + 1), energy_distribution)
    plt.xlabel('Coefficient')
    plt.ylabel('Energy Percentage')
    plt.title('Energy Distribution (Linear Scale)')
    
    plt.subplot(1, 2, 2)
    plt.semilogy(np.arange(1, len(energy_distribution) + 1), energy_distribution)
    plt.xlabel('Coefficient')
    plt.ylabel('Energy Percentage')
    plt.title('Energy Distribution (Log Scale)')
    
    plt.tight_layout()

if __name__ == "__main__":
    # Calculate and visualize DCT basis
    logging.info("Calculating DCT basis...")
    dct_basis = get_dct_basis()
    logging.info("Visualizing DCT basis...")
    visualize_dct_basis(dct_basis)
    plt.savefig(os.path.join(output_dir, 'dct_basis.png'))
    logging.info("DCT basis visualization completed.")
    
    # Load and process test images
    test_images = ['./DCT/images/bridge.tiff', './DCT/images/male.tiff']
    for image_path in test_images:
        logging.info(f"Processing image: {image_path}")
        original_image = get_image(image_path)
        shifted_image = level_shift(original_image)
        
        # Perform DCT and save coefficients
        logging.info("Performing DCT...")
        dct_coefficients = perform_dct(shifted_image)
        logging.info("Saving DCT coefficients...")
        np.save(os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '_dct.npy'), dct_coefficients)
        
        # Perform inverse DCT and compare with original image
        logging.info("Performing IDCT...")
        reconstructed_image = perform_idct(dct_coefficients)
        reconstructed_image = reconstructed_image + 128
        reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
        psnr = skimage.metrics.peak_signal_noise_ratio(original_image.astype(np.uint8), reconstructed_image)
        ssim = skimage.metrics.structural_similarity(original_image.astype(np.uint8), reconstructed_image)
        logging.info(f"Image: {image_path}")
        logging.info(f"PSNR: {psnr:.2f} dB")
        logging.info(f"SSIM: {ssim:.4f}")
        logging.info("")
        
        # Perform ZONAL coding and calculate PSNR and SSIM
        num_coeffs_zonal = [1, 2, 3, 4, 5, 6]
        for num_coeffs in num_coeffs_zonal:
            logging.info(f"Performing ZONAL coding with {num_coeffs} coefficient(s)...")
            reconstructed_coefficients = zonal_coding(dct_coefficients, num_coeffs)
            reconstructed_image = perform_idct(reconstructed_coefficients)
            reconstructed_image = reconstructed_image + 128
            reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
            psnr = skimage.metrics.peak_signal_noise_ratio(original_image.astype(np.uint8), reconstructed_image)
            ssim = skimage.metrics.structural_similarity(original_image.astype(np.uint8), reconstructed_image)
            logging.info(f"ZONAL Coding - {num_coeffs} coefficient(s):")
            logging.info(f"PSNR: {psnr:.2f} dB")
            logging.info(f"SSIM: {ssim:.4f}")
            logging.info("")
        
        # Perform THRESHOLD coding and calculate PSNR and SSIM
        num_coeffs_threshold = [1, 2, 3, 4, 5, 6]
        for num_coeffs in num_coeffs_threshold:
            logging.info(f"Performing THRESHOLD coding with {num_coeffs} coefficient(s)...")
            reconstructed_coefficients = threshold_coding(dct_coefficients, num_coeffs)
            reconstructed_image = perform_idct(reconstructed_coefficients)
            reconstructed_image = reconstructed_image + 128
            reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
            psnr = skimage.metrics.peak_signal_noise_ratio(original_image.astype(np.uint8), reconstructed_image)
            ssim = skimage.metrics.structural_similarity(original_image.astype(np.uint8), reconstructed_image)
            logging.info(f"THRESHOLD Coding - {num_coeffs} coefficient(s):")
            logging.info(f"PSNR: {psnr:.2f} dB")
            logging.info(f"SSIM: {ssim:.4f}")
            logging.info("")
        
        # Calculate and plot energy distribution
        logging.info("Calculating energy distribution...")
        energy_distribution = calculate_energy_distribution(dct_coefficients)
        logging.info("Plotting energy distribution...")
        plot_energy_distribution(energy_distribution)
        plt.savefig(os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '_energy_distribution.png'))
        plt.close()
        logging.info("Energy distribution plot completed.")
        logging.info("")