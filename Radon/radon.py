import numpy as np
import pywt
from skimage.transform import radon, iradon
from scipy.fftpack import dct, idct
from skimage.util import view_as_blocks
from skimage import color
import OpenEXR
import Imath

# 參數設置
P = 32
H = 8
C = 4
alpha = 0.1
wavelet = 'haar'

def read_exr_image(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    
    channels = header['channels'].keys()
    dataWindow = header['dataWindow']
    size = (dataWindow.max.x - dataWindow.min.x + 1, dataWindow.max.y - dataWindow.min.y + 1)
    
    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    
    images = []
    for channel in channels:
        buffer = exr_file.channel(channel, float_type)
        image = np.frombuffer(buffer, dtype=np.float32)
        image = image.reshape(size[1], size[0])
        images.append(image)
    
    exr_file.close()
    
    return np.stack(images, axis=-1)

# 步驟1: HDR圖像預處理
def preprocess(hdr_image):
    # 1.1 提取亮度分量Y並轉換到LogLuv域
    Y = 0.2126 * hdr_image[..., 0] + 0.7152 * hdr_image[..., 1] + 0.0722 * hdr_image[..., 2]
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
    for idx, direction_idx in enumerate(selected_indices):
        coeffs = block_dct[:,direction_idx][:C]
        coeffs_norm = (coeffs - coeffs.min()) / (coeffs.max() - coeffs.min())
        
        # 提取對應於當前方向的水印信息
        watermark_direction = watermark[idx*C:(idx+1)*C]
        
        # 3.4 使用QIM嵌入水印
        delta = alpha * coeffs.max()
        quantized = (coeffs_norm / delta).astype(int)
        watermarked = (quantized + watermark_direction) * delta
        block_dct[:,direction_idx][:C] = watermarked
        
    return block_dct

# 步驟4: 水印圖像重構
def combine_blocks(blocks):
    num_rows = int(np.sqrt(len(blocks)))
    num_cols = num_rows
    
    row_blocks = []
    for i in range(num_rows):
        row = np.hstack(blocks[i*num_cols:(i+1)*num_cols])
        row_blocks.append(row)
    
    combined_image = np.vstack(row_blocks)
    return combined_image

def reconstruct(watermarked_dct_blocks, theta):
    reconstructed_blocks = []
    for block_dct in watermarked_dct_blocks:
        dct_coeffs = block_dct
        sinogram = idct(dct_coeffs, axis=0, norm='ortho')
        block_recon = iradon(sinogram, theta=theta, circle=True)
        reconstructed_blocks.append(block_recon)

    # 将重建后的块拼接回图像
    reconstructed_image = combine_blocks(reconstructed_blocks)
    
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
    # 读取HDR图像
    hdr_image = read_exr_image('./Radon/images/snow.exr')
    
    # 计算图像分块后的块数
    height, width = hdr_image.shape[:2]
    B = (height // P) * (width // P)
    
    # 生成水印信息
    watermark = np.random.randint(0, 2, size=(B*H*C,))
    
    # 嵌入水印
    blocks = preprocess(hdr_image)
    dct_blocks = []
    for block in blocks:
        block_2d = block.reshape(-1, block.shape[-1])  # 将块转换为二维数组
        dct_block = rdct(block_2d)
        dct_blocks.append(dct_block)
    
    watermarked_dct_blocks = [embed(block, watermark[i*H*C:(i+1)*H*C], alpha=0.1) for i, block in enumerate(dct_blocks)]
    
    # 计算theta用于Radon逆变换
    num_projections = watermarked_dct_blocks[0].shape[0]
    theta = np.linspace(0., 180., num_projections, endpoint=False)
    watermarked_image = reconstruct(watermarked_dct_blocks, theta)
    
    # 提取水印
    extracted_watermark = extract(watermarked_image, alpha=0.1)
    
    # 评估水印提取准确率
    accuracy = np.sum(extracted_watermark == watermark) / len(watermark)
    print(f'Watermark extraction accuracy: {accuracy:.2f}')
    
if __name__ == '__main__':
    main()