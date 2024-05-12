import numpy as np
from scipy.signal import wiener
from scipy.fftpack import dct, idct
import io
from PIL import Image
import tempfile
import os
import math

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

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
            np.array([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]
            ]),
            np.array([
                [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]
            ])
        ]
    
    # 將量化表轉換為字典格式
    quant_tables_dict = {
        'quant_tables': quant_tables
    }

    # 刪除臨時JPEG文件
    os.remove(tmp.name)

    return quant_tables_dict

def stc_embed(cover, message, quantization_table):
    BlockSize = 8
    cover_blocks = view_as_blocks(cover, (BlockSize, BlockSize))
    cover_dct_blocks = dct(cover_blocks - 128, norm='ortho')

    # 使用 quantization_table 量化 DCT 係數
    cover_quantized = np.round(cover_dct_blocks / quantization_table)

    # 在量化的 DCT 係數中嵌入 message
    stego_quantized = cover_quantized + message

    # 對量化的 stego DCT 係數進行反量化
    stego_dct_blocks = stego_quantized * quantization_table

    # 對 stego DCT 係數應用反 DCT
    stego_blocks = idct(stego_dct_blocks, norm='ortho') + 128
    stego = unview_as_blocks(stego_blocks, cover.shape)

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
    PadSize = 8
    padded_fi = np.pad(FisherInformation, ((PadSize, PadSize), (PadSize, PadSize), (0, 0), (0, 0)), mode='symmetric')

    tmp = np.zeros_like(padded_fi)
    tmp[PadSize:-PadSize, PadSize:-PadSize] = padded_fi[PadSize:-PadSize, PadSize:-PadSize]
    tmp[:PadSize, :] = tmp[PadSize:2*PadSize, :]
    tmp[-PadSize:, :] = tmp[-2*PadSize:-PadSize, :]
    tmp[:, :PadSize] = tmp[:, PadSize:2*PadSize]
    tmp[:, -PadSize:] = tmp[:, -2*PadSize:-PadSize]

    smoothed_fi = (
        tmp[:-2*PadSize, :-2*PadSize] +
        tmp[PadSize:-PadSize, :-2*PadSize] * 3 +
        tmp[2*PadSize:, :-2*PadSize] +
        tmp[:-2*PadSize, PadSize:-PadSize] * 3 +
        tmp[PadSize:-PadSize, PadSize:-PadSize] * 4 +
        tmp[2*PadSize:, PadSize:-PadSize] * 3 +
        tmp[:-2*PadSize, 2*PadSize:] +
        tmp[PadSize:-PadSize, 2*PadSize:] * 3 +
        tmp[2*PadSize:, 2*PadSize:]
    ) / 20

    return smoothed_fi

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

def RCgetJPEGmtx():
    cc, rr = np.meshgrid(range(8), range(8))
    T = np.sqrt(2 / 8) * np.cos(np.pi * (2 * cc + 1) * rr / 16)
    T[0, :] /= np.sqrt(2)
    dct8_mtx = np.zeros((64, 64))
    for i in range(64):
        dcttmp = np.zeros((8, 8))
        dcttmp[i // 8, i % 8] = 1
        TTMP = T @ dcttmp @ T.T
        dct8_mtx[:, i] = TTMP.flatten()
    return dct8_mtx

def view_as_blocks(arr, block_size):
    m, n = arr.shape
    M, N = block_size
    return arr.reshape(m//M, M, n//N, N).swapaxes(1,2)

def unview_as_blocks(blocks, arr_shape):
    M, N = blocks.shape[1], blocks.shape[2]
    return blocks.swapaxes(1,2).reshape(arr_shape)

def JMiPODv0(cover_y, C_STRUCT, Payload):
    quantization_tables = C_STRUCT['quant_tables']
    C_QUANT = quantization_tables[0]
    
    BlockSize = 8
    cover_y_blocks = view_as_blocks(cover_y, (BlockSize, BlockSize))
    num_blocks = cover_y_blocks.shape[0] * cover_y_blocks.shape[1]
    
    DCT_real = dct(cover_y_blocks - 128, norm='ortho') / C_QUANT
    DCT_rounded = np.round(DCT_real)
    
    C_STRUCT['coef_arrays'] = [DCT_rounded]
    e = DCT_rounded - DCT_real
    sgn_e = np.sign(e)
    
    rand_mask = np.random.rand(*e.shape) > 0.5
    sgn_e[e == 0] = rand_mask[e == 0] * 2 - 1
    
    change = -sgn_e
    
    WienerResidual = cover_y - wiener(cover_y, (2, 2))
    Variance = VarianceEstimationDCT2D(WienerResidual, BlockSize=3, Degree=3)
    
    VarianceDCT = view_as_blocks(Variance, (BlockSize, BlockSize))
    VarianceDCT[VarianceDCT < 1e-10] = 1e-10
    
    FisherInformation = 1 / VarianceDCT ** 2
    FisherInformation = smooth_fisher_info(FisherInformation)
    
    FI = FisherInformation * (2 * e - sgn_e) ** 4
    maxCostMat = np.zeros_like(FI, dtype=bool)
    maxCostMat[:, :, 0, 0] = True
    maxCostMat[:, :, 4, 0] = True
    maxCostMat[:, :, 0, 4] = True
    maxCostMat[:, :, 4, 4] = True
    FI[maxCostMat & (np.abs(e) > 0.4999)] = 1e10
    FI[np.abs(e) < 0.01] = 1e10
    
    nzAC = num_blocks * 63
    messageLenght = int(np.round(Payload * nzAC * np.log(2)))
    
    beta = TernaryProbs(FI, messageLenght)
    
    np.random.seed(0)
    r = np.random.rand(*DCT_rounded.shape)
    ModifPM1 = r < beta.reshape(DCT_rounded.shape)
    S_COEFFS = DCT_rounded.copy()
    S_COEFFS[ModifPM1] += change[ModifPM1]
    S_COEFFS[S_COEFFS > 1024] = 1024
    S_COEFFS[S_COEFFS < -1023] = -1023
    ChangeRate = np.count_nonzero(ModifPM1) / ModifPM1.size
    pChange = beta.reshape(DCT_rounded.shape)
    
    S_STRUCT = C_STRUCT.copy()
    S_STRUCT['coef_arrays'] = [S_COEFFS]
    Deflection = np.sum(pChange * FI)
    stego_y_dct_blocks = S_COEFFS * C_QUANT
    stego_y_blocks = idct(stego_y_dct_blocks, norm='ortho') + 128
    stego_y = unview_as_blocks(stego_y_blocks, cover_y.shape)
    
    return stego_y, pChange, ChangeRate, Deflection

def VarianceEstimationDCT2D(Image, BlockSize, Degree):
    if BlockSize % 2 != 1:
        raise ValueError('The block dimensions should be odd!')
    if Degree > BlockSize:
        raise ValueError('Number of basis vectors exceeds block dimension!')
    
    q = (Degree + 1) * (Degree + 2) // 2
    BaseMat = np.zeros((BlockSize, BlockSize))
    BaseMat[0, 0] = 1
    G = np.zeros((BlockSize ** 2, q))
    k = 0
    for xShift in range(1, Degree + 1):
        for yShift in range(1, Degree - xShift + 2):
            G[:, k] = idct(idct(np.roll(BaseMat, (xShift - 1, yShift - 1), axis=(0, 1)), norm='ortho'), norm='ortho').flatten()
            k += 1
    
    PadSize = BlockSize // 2
    padded_image = np.pad(Image, ((PadSize, PadSize), (PadSize, PadSize)), mode='symmetric')
    
    m, n = Image.shape
    M, N = BlockSize, BlockSize
    num_blocks_ver = (m + M - 1) // M
    num_blocks_hor = (n + N - 1) // N
    
    EstimatedVariance = np.zeros_like(Image, dtype=np.float64)
    for i in range(num_blocks_ver):
        for j in range(num_blocks_hor):
            block = padded_image[i*M:i*M+BlockSize, j*N:j*N+BlockSize]
            block_flat = block.flatten()
            PGorth = np.eye(BlockSize ** 2) - G @ np.linalg.pinv(G.T @ G) @ G.T
            var_est = np.sum((PGorth @ block_flat) ** 2) / (BlockSize ** 2 - q)
            EstimatedVariance[i*M:min((i+1)*M, m), j*N:min((j+1)*N, n)] = var_est
    
    return EstimatedVariance

