import numpy as np
import pywt
from skimage.transform import radon, iradon
from scipy.fftpack import dct, idct
from skimage.util import view_as_blocks
import cv2

# 參數設置
P = 32
H = 8
C = 4
alpha = 0.1
wavelet = 'haar'

# 步驟1: HDR圖像預處理
def preprocess(hdr_image):
    # 1.1 提取亮度分量Y並轉換到LogLuv域
    Y = 0.2126 * hdr_image[:,:,0] + 0.7152 * hdr_image[:,:,1] + 0.0722 * hdr_image[:,:,2]
    L = np.log10(Y)
    
    # 1.2 線性縮放到[0,1]範圍內
    N = (L - L.min()) / (L.max() - L.min())
    
    # 1.3 小波分解,選取高頻子帶
    LL, (LH, HL, HH) = pywt.dwt2(N, 'haar')
    gamma = HH
    
    # 1.4 分割成B個P×P的塊
    P = 32
    blocks = view_as_blocks(gamma, (P, P))
    return blocks

# 步驟2: Radon-DCT變換
def rdct(block):
    # 2.1 Radon變換
    theta = np.linspace(0., 180., max(block.shape), endpoint=False)
    sinogram = radon(block, theta=theta, circle=True)
    
    # 2.2 對每個投影進行DCT變換
    dct_coeffs = dct(sinogram, axis=0, norm='ortho')
    return dct_coeffs

# 步驟3: 系數選擇與QIM嵌入
def embed(block_dct, watermark, alpha):
    # 3.1 根據RDCT係數能量排序
    energy = np.sum(np.abs(block_dct), axis=0)
    sorted_indices = np.argsort(energy)[::-1]
    
    # 3.2 選擇H個方向嵌入水印
    H = 8
    selected_indices = sorted_indices[:H]
    
    # 3.3 在每個選定方向上選C個係數並縮放
    C = 4
    for idx in selected_indices:
        coeffs = block_dct[:,idx][:C]
        coeffs_norm = (coeffs - coeffs.min()) / (coeffs.max() - coeffs.min())
        
        # 3.4 使用QIM嵌入水印
        delta = alpha * coeffs.max()
        quantized = (coeffs_norm / delta).astype(int)
        watermarked = (quantized + watermark) * delta
        block_dct[:,idx][:C] = watermarked
        
    return block_dct

# 步驟4: 水印圖像重構
def combine_blocks(blocks):
    n_rows, n_cols = blocks.shape[0], blocks.shape[1]
    block_size = blocks.shape[2]
    combined_image = np.zeros((n_rows * block_size, n_cols * block_size))
    for i in range(n_rows):
        for j in range(n_cols):
            combined_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = blocks[i, j]
    return combined_image

def reconstruct(watermarked_blocks):
    # 4.1 對嵌入水印的RDCT係數進行逆DCT和Radon變換
    watermarked_dct_blocks = watermarked_blocks.reshape(-1, P, P)
    reconstructed_blocks = []
    for block in watermarked_dct_blocks:
        dct_coeffs = block
        sinogram = idct(dct_coeffs, axis=0, norm='ortho')
        block_recon = iradon(sinogram, theta=theta, circle=True)
        reconstructed_blocks.append(block_recon)
    reconstructed_gamma = combine_blocks(reconstructed_blocks)
    reconstructed_image = reconstructed_gamma  # 這裡需要根據實際的逆小波變換過程進行修改
    
    # 4.2 - 4.4 小波逆變換,指數運算,拼接色度分量(省略)
    return reconstructed_image

# 步驟5: 水印提取
def extract(watermarked_image, alpha):
    # 5.1 對含水印圖像進行預處理和RDCT變換
    watermarked_blocks = preprocess(watermarked_image)
    watermarked_dct_blocks = [rdct(block) for block in watermarked_blocks]
    
    extracted_watermark = []
    for block_dct in watermarked_dct_blocks:
        # 5.2 提取嵌入水印的RDCT係數並歸一化
        energy = np.sum(np.abs(block_dct), axis=0)
        sorted_indices = np.argsort(energy)[::-1]
        selected_indices = sorted_indices[:H]
        
        block_watermark = []
        for idx in selected_indices:
            coeffs = block_dct[:,idx][:C]
            coeffs_norm = (coeffs - coeffs.min()) / (coeffs.max() - coeffs.min())
            
            # 5.3 使用最小距離解碼提取水印信息
            delta = alpha * coeffs.max()
            quantized = (coeffs_norm / delta).astype(int)
            extracted_bits = (quantized % 2).astype(int)
            block_watermark.append(extracted_bits)
        
        # 5.4 多數表決
        block_watermark = np.array(block_watermark)
        extracted_watermark.append(np.median(block_watermark, axis=0))
        
    extracted_watermark = np.array(extracted_watermark).flatten()
    return extracted_watermark

# 主函數
def main():
    # 讀取HDR圖像
    hdr_image = cv2.imread('./HDR/HDR images/nave.hdr', flags=cv2.IMREAD_ANYDEPTH)
    
    # 生成水印信息
    watermark = np.random.randint(0, 2, size=(B*H*C,))
    
    # 嵌入水印
    blocks = preprocess(hdr_image)
    dct_blocks = [rdct(block) for block in blocks]
    watermarked_dct_blocks = [embed(block, watermark[i*H*C:(i+1)*H*C], alpha=0.1) for i, block in enumerate(dct_blocks)]
    watermarked_image = reconstruct(watermarked_dct_blocks)
    
    # 提取水印
    extracted_watermark = extract(watermarked_image, alpha=0.1)
    
    # 評估水印提取準確率
    accuracy = np.sum(extracted_watermark == watermark) / len(watermark)
    print(f'Watermark extraction accuracy: {accuracy:.2f}')
    
if __name__ == '__main__':
    main()