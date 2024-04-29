import os
from PIL import Image
import numpy as np
import skimage.metrics
import matplotlib.pyplot as plt

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
    dct_coefficients = np.zeros_like(image)
    for i in range(image.shape[0]):
        dct_coefficients[i, :] = dct_1d(image[i, :])
    for j in range(image.shape[1]):
        dct_coefficients[:, j] = dct_1d(dct_coefficients[:, j])
    return dct_coefficients

def idct_2d(coefficients):
    reconstructed_image = np.zeros_like(coefficients)
    for j in range(coefficients.shape[1]):
        reconstructed_image[:, j] = idct_1d(coefficients[:, j])
    for i in range(coefficients.shape[0]):
        reconstructed_image[i, :] = idct_1d(reconstructed_image[i, :])
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

def get_image(image_path):
    image = Image.open(image_path)
    img_grey = image.convert('L')
    img = np.array(img_grey, dtype=np.float64)
    return img

def perform_dct(image):
    dct_coefficients = dct_2d(image)
    return dct_coefficients

def perform_idct(coefficients):
    reconstructed_image = idct_2d(coefficients)
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
    zigzag_coeffs = zigzag_scan(coefficients)
    zigzag_coeffs[num_coeffs:] = 0
    reconstructed_coefficients = np.zeros_like(coefficients)
    reconstructed_coefficients[np.unravel_index(np.arange(num_coeffs), (8, 8))] = zigzag_coeffs[:num_coeffs]
    return reconstructed_coefficients

def threshold_coding(coefficients, num_coeffs):
    zigzag_coeffs = zigzag_scan(coefficients)
    sorted_indices = np.argsort(np.abs(zigzag_coeffs))[::-1]
    top_k_indices = sorted_indices[:num_coeffs]
    reconstructed_coefficients = np.zeros_like(coefficients)
    reconstructed_coefficients[np.unravel_index(top_k_indices, (8, 8))] = zigzag_coeffs[top_k_indices]
    return reconstructed_coefficients

def calculate_energy_distribution(coefficients):
    energy = np.sum(coefficients ** 2, axis=(1, 2))
    total_energy = np.sum(energy)
    energy_distribution = energy / total_energy
    return energy_distribution

def plot_energy_distribution(energy_distribution):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(energy_distribution)
    plt.xlabel('Coefficient')
    plt.ylabel('Energy Percentage')
    plt.title('Energy Distribution (Linear Scale)')
    
    plt.subplot(1, 2, 2)
    plt.semilogy(energy_distribution)
    plt.xlabel('Coefficient')
    plt.ylabel('Energy Percentage')
    plt.title('Energy Distribution (Log Scale)')
    
    plt.tight_layout()
    plt.savefig('energy_distribution.png')
    plt.close()

if __name__ == "__main__":
    # Calculate and visualize DCT basis
    dct_basis = get_dct_basis()
    visualize_dct_basis(dct_basis)
    
    # Load and process test images
    test_images = ['./DCT/images/bridge.tiff', './DCT/images/male.tiff']
    for image_path in test_images:
        original_image = get_image(image_path)
        shifted_image = level_shift(original_image)
        
        # Perform DCT and save coefficients
        dct_coefficients = perform_dct(shifted_image)
        np.save(os.path.splitext(image_path)[0] + '_dct.npy', dct_coefficients)
        
        # Perform inverse DCT and compare with original image
        reconstructed_image = perform_idct(dct_coefficients)
        reconstructed_image = reconstructed_image + 128
        reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
        psnr = skimage.metrics.peak_signal_noise_ratio(original_image.astype(np.uint8), reconstructed_image)
        ssim = skimage.metrics.structural_similarity(original_image.astype(np.uint8), reconstructed_image)
        print(f"Image: {image_path}")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM: {ssim:.4f}")
        print()
        
        # Perform ZONAL coding and calculate PSNR and SSIM
        num_coeffs_zonal = [1, 2, 3, 4, 5, 6]
        for num_coeffs in num_coeffs_zonal:
            reconstructed_coefficients = zonal_coding(dct_coefficients, num_coeffs)
            reconstructed_image = perform_idct(reconstructed_coefficients)
            reconstructed_image = reconstructed_image + 128
            reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
            psnr = skimage.metrics.peak_signal_noise_ratio(original_image.astype(np.uint8), reconstructed_image)
            ssim = skimage.metrics.structural_similarity(original_image.astype(np.uint8), reconstructed_image)
            print(f"ZONAL Coding - {num_coeffs} coefficient(s):")
            print(f"PSNR: {psnr:.2f} dB")
            print(f"SSIM: {ssim:.4f}")
            print()
        
        # Perform THRESHOLD coding and calculate PSNR and SSIM
        num_coeffs_threshold = [1, 2, 3, 4, 5, 6]
        for num_coeffs in num_coeffs_threshold:
            reconstructed_coefficients = threshold_coding(dct_coefficients, num_coeffs)
            reconstructed_image = perform_idct(reconstructed_coefficients)
            reconstructed_image = reconstructed_image + 128
            reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
            psnr = skimage.metrics.peak_signal_noise_ratio(original_image.astype(np.uint8), reconstructed_image)
            ssim = skimage.metrics.structural_similarity(original_image.astype(np.uint8), reconstructed_image)
            print(f"THRESHOLD Coding - {num_coeffs} coefficient(s):")
            print(f"PSNR: {psnr:.2f} dB")
            print(f"SSIM: {ssim:.4f}")
            print()
        
        # Calculate and plot energy distribution
        energy_distribution = calculate_energy_distribution(dct_coefficients)
        plot_energy_distribution(energy_distribution)