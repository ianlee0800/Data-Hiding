import numpy as np
from scipy.signal import wiener
from scipy.fftpack import dct, idct
import io
from PIL import Image
import tempfile
import os

def extract_quantization_tables(image_path):
    # 創建一個臨時文件來保存JPEG版本的圖像
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        # 打開原始圖像
        img = Image.open(image_path)
        
        # 將圖像保存為JPEG格式到臨時文件
        img.save(tmp.name, format='JPEG')
        
        # 從臨時JPEG文件中提取量化表
        with open(tmp.name, 'rb') as file:
            if 'dpi' in img.info:
                dpi = img.info['dpi']
            else:
                dpi = (96, 96)
            img_byte_arr = io.BytesIO(file.read())

        quant_tables = []
        for marker in [b'\xFF\xDB', b'\xFF\xC4']:
            pos = 0
            while True:
                pos = img_byte_arr.getvalue().find(marker, pos)
                if pos < 0:
                    break
                length = img_byte_arr.getvalue()[pos+2]*256 + img_byte_arr.getvalue()[pos+3]
                quant_table = np.frombuffer(img_byte_arr.getvalue(), dtype=np.uint8,
                                            count=length, offset=pos+4)
                if len(quant_table) == 64:
                    quant_table = quant_table.reshape((8, 8))
                    quant_tables.append(quant_table)
                pos += length

    # 如果沒有找到量化表或量化表大小不是8x8,返回默認的亮度和色度量化表
    if len(quant_tables) == 0:
        quant_tables = [
            np.array([... # 默認亮度量化表
            ]),
            np.array([... # 默認色度量化表
            ])
        ]

    # 刪除臨時JPEG文件
    os.remove(tmp.name)

    return quant_tables

def stc_embed(cover, message, quantization_table):
    # 對 cover 圖片應用 DCT
    cover_dct = dct(cover)

    # 使用 quantization_table 量化 DCT 係數
    cover_quantized = np.round(cover_dct / quantization_table)

    # 在量化的 DCT 係數中嵌入 message
    stego_quantized = cover_quantized + message

    # 對量化的 stego DCT 係數進行反量化
    stego_dct = stego_quantized * quantization_table

    # 對 stego DCT 係數應用反 DCT
    stego = idct(stego_dct)

    return stego

def dct2(block):
    return dct(block.astype(np.float32))

def idct2(block):
    return idct(block.astype(np.float32))

def blkproc(im, blksize, fun):
    im = np.asarray(im)
    if im.ndim == 2:
        im = im[:, :, np.newaxis]
    elif im.ndim != 3:
        raise ValueError('Input image must be 2D or 3D')

    h, w, c = im.shape
    bh, bw = blksize
    shape = (h//bh, bh, w//bw, bw, c)
    strides = (w*bh*im.itemsize, im.itemsize, bw*im.itemsize, im.itemsize, h*w*im.itemsize)
    blocks = np.lib.stride_tricks.as_strided(im, shape=shape, strides=strides)
    return fun(blocks)

def smooth_fisher_info(FisherInformation):
    h, w = FisherInformation.shape
    tmp = np.pad(FisherInformation, ((8,8),(8,8)), mode='symmetric')
    
    for i in range(8):
        for j in range(8):
            tmp[i::8, j::8] = np.array([
                tmp[i  :i+h, j  :j+w],
                tmp[i+8:i+8+h, j  :j+w],
                tmp[i-8:i-8+h, j  :j+w],
                tmp[i  :i+h, j+8:j+8+w],
                tmp[i  :i+h, j-8:j-8+w],
                tmp[i+8:i+8+h, j+8:j+8+w],
                tmp[i-8:i-8+h, j+8:j+8+w],
                tmp[i+8:i+8+h, j-8:j-8+w],
                tmp[i-8:i-8+h, j-8:j-8+w]
            ]).mean(0)
            
    return tmp[8:-8, 8:-8]

def hBinary(probs):
    p = np.clip(probs, 1e-10, 1 - 1e-10)
    return -np.sum(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def invxlnx2_fast(y, xx, yy):
    x = np.zeros_like(y)
    i_large = y > 1000
    if np.any(i_large):
        x[i_large] = 1 / (y[i_large] / np.log(y[i_large] - 1))

    i_small = ~i_large
    if np.any(i_small):
        ind = np.sum(y[i_small, np.newaxis] >= xx[np.newaxis, :], axis=1)
        ind = np.minimum(ind, len(yy) - 2)
        x[i_small] = yy[ind] + (y[i_small] - xx[ind]) / (xx[ind + 1] - xx[ind]) * (yy[ind + 1] - yy[ind])

    return x

def TernaryProbs(FI, payload):
    xx = np.array([0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.2500, 1.5000, 1.7500, 2.0000, 2.2500, 2.5000, 2.7500, 3.0000, 3.2500, 3.5000, 3.7500, 4.0000, 4.2500, 4.5000, 4.7500, 5.0000])
    yy = np.array([0.5000, 0.4134, 0.3504, 0.3041, 0.2679, 0.2385, 0.2137, 0.1923, 0.1736, 0.1570, 0.1422, 0.1288, 0.1166, 0.1055, 0.0953, 0.0858, 0.0771, 0.0690, 0.0614, 0.0544, 0.0477])

    L, R = 5 * 10**-1.5, 5 * 10**0.5
    fL = hBinary(invxlnx2_fast(L * FI, xx, yy)) - payload
    fR = hBinary(invxlnx2_fast(R * FI, xx, yy)) - payload

    while fL * fR > 0:
        if fL > 0:
            R = 2 * R
            fR = hBinary(invxlnx2_fast(R * FI, xx, yy)) - payload
        else:
            L = L / 2
            fL = hBinary(invxlnx2_fast(L * FI, xx, yy)) - payload

    M, fM = (L + R) / 2, 1
    i = 0
    while np.abs(fM) > 0.001 and i < 20:
        fM = hBinary(invxlnx2_fast(M * FI, xx, yy)) - payload
        if fL * fM < 0:
            R, fR = M, fM
        else:
            L, fL = M, fM
        M = (L + R) / 2
        i += 1

    if i == 20:
        M = M[np.abs(fM).argmin()]

    beta = invxlnx2_fast(M * FI, xx, yy)
    return beta

def JMiPODv0(cover_y, C_STRUCT, Payload):
    quantization_tables = C_STRUCT['quant_tables']
    C_QUANT = quantization_tables[0]
    
    DCT_rounded = C_STRUCT['coef_arrays'][0]
    
    VarianceDCT = VarianceEstimationDCT2D(cover_y, BlockSize=7, Degree=3)
    
    FisherInformation = 1 / (VarianceDCT * C_QUANT**2)**2
    
    # 平滑Fisher Information矩陣
    FisherInformation = smooth_fisher_info(FisherInformation)
    
    nzAC = np.count_nonzero(DCT_rounded) - np.count_nonzero(DCT_rounded[::8, ::8])
    messageLenght = int(np.round(Payload * nzAC * np.log2(3)))
    
    FI = FisherInformation.flatten()
    beta = TernaryProbs(FI, messageLenght)
    
    pChange = beta.reshape(DCT_rounded.shape)
    
    # 將beta轉換為syndrome coding所需的costs
    rhos = -np.log(pChange / (1 - 2*pChange)) / 0.69314718055994529
    
    # 使用實際的STC進行嵌入編碼
    stego_y = stc_embed(cover_y, rhos, C_QUANT)
    
    return stego_y, pChange

def VarianceEstimationDCT2D(Image, BlockSize, Degree):
    if BlockSize % 2 != 1:
        raise ValueError('BlockSize must be odd')

    if Degree > BlockSize:
        raise ValueError('Degree must be less than or equal to BlockSize')

    h, w = Image.shape
    PadSize = BlockSize // 2
    PadImage = np.pad(Image, ((PadSize, PadSize), (PadSize, PadSize)), mode='symmetric')
    Blocks = np.lib.stride_tricks.as_strided(PadImage, shape=(h, w, BlockSize, BlockSize), strides=PadImage.itemsize * np.array([w, 1, w, 1]))

    q = (Degree + 1) * (Degree + 2) // 2
    G = np.zeros((BlockSize * BlockSize, q))
    k = 0
    for i in range(Degree + 1):
        for j in range(Degree - i + 1):
            basis = np.zeros((BlockSize, BlockSize), dtype=np.float32)
            basis[i, j] = 1
            G[:, k] = idct(idct(basis, axis=1, norm='ortho'), axis=0, norm='ortho').flatten()
            k += 1

    G_trans = G.T
    Gt_G_inv = np.linalg.inv(G_trans @ G)
    Var_est = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            Block = Blocks[i, j].flatten()[:, np.newaxis]
            Theta_est = Gt_G_inv @ G_trans @ Block
            Var_est[i, j] = np.linalg.norm(Block - G @ Theta_est)**2 / (BlockSize * BlockSize - q)

    return Var_est

