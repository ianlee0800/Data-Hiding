import torch
import torch.nn as nn
import cv2
import numpy as np
from scipy.signal import convolve2d
from CNN import ImprovedAdaptiveCNNPredictor, preprocess_image
import scipy.ndimage as ndimage
from skimage.metrics import structural_similarity as ssim

# Constants
MODEL_PATH = './CNN_PEE/model/model_epoch_50.pth'
COVER_IMAGE_PATH = "./CNN_PEE/origin/Tank.png"
STEGO_IMAGE_PATH = "./CNN_PEE/stego/Tank_stego.png"
EMBEDDING_INFO_PATH = "./CNN_PEE/stego/embedding_info.npy"
STEGO_VALUES_PATH = "./CNN_PEE/stego/stego_values.npy"
ORIGINAL_SECRET_DATA_PATH = "./CNN_PEE/stego/original_secret_data.npy"

def load_cnn_model(model_path, device):
    model = ImprovedAdaptiveCNNPredictor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def cnn_predict(model, image, device):
    with torch.no_grad():
        preprocessed_image = preprocess_image(image)
        image_tensor = torch.from_numpy(preprocessed_image).unsqueeze(0).unsqueeze(0).float().to(device)
        prediction = model(image_tensor)
    return prediction.squeeze().cpu().numpy()

def get_edge_map(image, threshold=30):
    edges = ndimage.sobel(image)
    return edges > threshold

def pee_embedding(cover_image, secret_data, model, device):
    print("Original secret data:", secret_data)
    secret_data = secret_data.astype(int)
    print("Secret data after type conversion:", secret_data)
    
    predicted_image = cnn_predict(model, cover_image, device)

    embedding_info = []
    stego_image = cover_image.copy()
    data_index = 0

    # Calculate prediction error histogram
    errors = cover_image - predicted_image
    hist, _ = np.histogram(errors, bins=np.arange(-256, 257))

    # Find optimal embedding range
    center = np.argmax(hist) - 256
    left = max(center - 1, -256)
    right = min(center + 1, 255)

    for i in range(cover_image.shape[0]):
        for j in range(cover_image.shape[1]):
            if data_index >= len(secret_data):
                break
            
            original_value = int(cover_image[i, j])
            predicted_value = int(round(predicted_image[i, j]))
            error = original_value - predicted_value

            # Embed only within the optimal range
            if left <= error <= right:
                bit_to_embed = secret_data[data_index]
                if error >= center:
                    stego_value = predicted_value + 2 * (error - center) + bit_to_embed + center
                else:
                    stego_value = predicted_value + 2 * (error - center) - bit_to_embed + center

                stego_image[i, j] = np.clip(stego_value, 0, 255)
                
                embedding_info.append({
                    'position': (i, j),
                    'original_value': original_value,
                    'predicted_value': predicted_value,
                    'error': error,
                    'embedded_bit': bit_to_embed,
                    'stego_value': stego_value
                })

                data_index += 1

    # Verify embedded data
    embedded_bits = [info['embedded_bit'] for info in embedding_info]
    if not np.array_equal(embedded_bits, secret_data[:len(embedded_bits)]):
        print("Warning: Embedded bits do not match original secret data")
        for i, (embedded, original) in enumerate(zip(embedded_bits, secret_data[:len(embedded_bits)])):
            if embedded != original:
                print(f"Position {i}: Embedded {embedded}, Original {original}")
                print(f"Detailed info: {embedding_info[i]}")

    # Save and verify embedding information
    np.save(EMBEDDING_INFO_PATH, embedding_info)
    loaded_info = np.load(EMBEDDING_INFO_PATH, allow_pickle=True)
    if not all(a == b for a, b in zip(embedding_info, loaded_info)):
        print("Warning: Saved and loaded embedding information do not match")

    # Save stego image pixel values
    np.save(STEGO_VALUES_PATH, stego_image)

    print(f"Number of embedding positions: {len(embedding_info)}")
    print(f"Embedding error range: {min([info['error'] for info in embedding_info])} to {max([info['error'] for info in embedding_info])}")
    print(f"Original secret data: {secret_data}")
    
    np.save(ORIGINAL_SECRET_DATA_PATH, secret_data)

    return stego_image, embedding_info, data_index

# Main program
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CNN model
    cnn_model = load_cnn_model(MODEL_PATH, device)

    # Read cover image
    cover_image = cv2.imread(COVER_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if cover_image is None:
        raise ValueError(f"Failed to load image from {COVER_IMAGE_PATH}")

    # Generate secret data (example)
    payload_size = 10000
    secret_data = np.random.randint(0, 2, payload_size)

    # Perform PEE embedding
    stego_image, embedding_info, embedded_bits = pee_embedding(cover_image, secret_data, cnn_model, device)
    
    print(f"Number of actually embedded bits: {embedded_bits}")
    print(f"Embedding rate: {embedded_bits / (cover_image.shape[0] * cover_image.shape[1]):.4f} bpp")
    
    # Save stego image
    cv2.imwrite(STEGO_IMAGE_PATH, stego_image)
    np.save(STEGO_VALUES_PATH, stego_image)

    # Save embedding information
    np.save(EMBEDDING_INFO_PATH, embedding_info)

    print("Embedding completed. Stego image has been saved.")

    # Calculate PSNR
    mse = np.mean((cover_image - stego_image) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    print(f"PSNR: {psnr:.2f} dB")

    # Calculate SSIM
    ssim_value = ssim(cover_image, stego_image, data_range=stego_image.max() - stego_image.min())
    print(f"SSIM: {ssim_value:.4f}")