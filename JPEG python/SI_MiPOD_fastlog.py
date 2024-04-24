import numpy as np
from scipy.fftpack import dct, idct
from scipy.signal import wiener
from scipy.io import loadmat
import os
from jpegtbx import im2vec, vec2im
import cv2

def SI_MiPOD_fastlog(preCover, C_STRUCT, Payload):
    print(f"preCover shape: {preCover.shape}")
    """
    SI_MiPOD Embedding
    
    :param preCover: Path to the cover image or the cover image or JPEG STRUCT
    :param C_STRUCT: JPEG STRUCT
    :param Payload: Embedding payload in bits per DCT coefs (bpc).
    :return: Resulting stego jpeg STRUCT with embedded payload, FisherInfo, Deflection, pChange, ChangeRate
    """
    
    # Read the JPEG image if needed
    if isinstance(preCover, str):
        preCover = cv2.imread(preCover, flags=cv2.IMREAD_UNCHANGED)
        preCover = preCover.astype(np.float64)  # Convert to np.double
        
    if isinstance(C_STRUCT, dict) and 'quant_tables' in C_STRUCT:
        C_QUANT = C_STRUCT['quant_tables']
        if isinstance(C_QUANT, dict):
            if 0 in C_QUANT:
                C_QUANT = C_QUANT[0]
            else:
                raise ValueError("Invalid quantization table in C_STRUCT.")
        elif isinstance(C_QUANT, list):
            valid_quant_tables = [quant_table for quant_table in C_QUANT if isinstance(quant_table, np.ndarray) and quant_table.shape == (8, 8)]
            if len(valid_quant_tables) == 1:
                C_QUANT = valid_quant_tables[0]
            else:
                raise ValueError("Quantization table must be a single 8x8 array.")
        else:
            if C_QUANT.shape != (8, 8):
                raise ValueError("Quantization table must be 8x8.")
    else:
        raise ValueError("Invalid C_STRUCT. Expected a dictionary with 'quant_tables' key.")
    
    print(f"C_QUANT shape: {C_QUANT.shape}")
    
    # Get the DCT matrix
    MatDCT = RCgetJPEGmtx()
    print(f"MatDCT shape: {MatDCT.shape}")
    
    im2vec_result = im2vec(preCover - 128, [8, 8])
    print(f"im2vec(preCover - 128, [8, 8]) shape: {im2vec_result[0].shape}")
    
    dct_coeffs = np.dot(MatDCT, im2vec_result[0])
    dct_coeffs_scaled = dct_coeffs / C_QUANT.flatten()[:, None]
    print(f"dct_coeffs_scaled shape: {dct_coeffs_scaled.shape}")

    DCT_real = vec2im(dct_coeffs_scaled, [0, 0], [8, 8])
    DCT_rounded = np.round(DCT_real)

    # Initialize 'coef_arrays' key in C_STRUCT
    if 'coef_arrays' not in C_STRUCT:
        C_STRUCT['coef_arrays'] = []

    # Create a new coef_arrays list with DCT_rounded as the first element
    C_STRUCT['coef_arrays'] = [DCT_rounded]
    
    e = DCT_rounded - DCT_real
    
    # Compute rounding error
    sgn_e = np.sign(e)
    zero_indices = e.flatten() == 0
    sgn_e.flat[zero_indices] = np.random.randint(2, size=np.sum(zero_indices)) * 2 - 1
    change = -sgn_e
    
    # Compute Variance in spatial domain
    WienerResidual = preCover - wiener(preCover, [3, 3])  # Use a 3x3 window for Wiener filtering
    Variance = VarianceEstimationDCT2D(WienerResidual, 3, 3)  # Use an odd block size of 3
    
    # Apply the covariance transformation to DCT domain
    MatDCTq = MatDCT ** 2
    Qvec = C_STRUCT['quant_tables'][0].flatten() ** 2
    for idx in range(64):
        MatDCTq[idx, :] = MatDCTq[idx, :] / Qvec[idx]
    
    VarianceDCT = np.zeros_like(preCover)
    for idxR in range(0, Variance.shape[0], 8):
        for idxC in range(0, Variance.shape[1], 8):
            tmp = Variance[idxR:idxR+8, idxC:idxC+8]
            VarianceDCT[idxR:idxR+8, idxC:idxC+8] = np.reshape(np.dot(MatDCTq, tmp.flatten()), (8, 8))
    
    VarianceDCT[VarianceDCT < 1e-10] = 1e-10
    
    # Compute Fisher information and smooth it
    FisherInformation = 1 / VarianceDCT ** 2
    
    # Post Filter
    tmp = np.zeros((FisherInformation.shape[0] + 16, FisherInformation.shape[1] + 16))
    tmp[9:-8, 9:-8] = FisherInformation
    tmp[1:8, :] = tmp[9:16, :]
    tmp[-7:, :] = tmp[-15:-8, :]
    tmp[:, 1:8] = tmp[:, 9:16]
    tmp[:, -7:] = tmp[:, -15:-8]
    FisherInformation = (
        tmp[:-16, :-16] +
        tmp[9:-8, :-16] * 3 +
        tmp[17:, :-16] +
        tmp[:-16, 9:-8] * 3 +
        tmp[9:-8, 9:-8] * 4 +
        tmp[17:, 9:-8] * 3 +
        tmp[:-16, 17:] +
        tmp[9:-8, 17:] * 3 +
        tmp[17:, 17:]
    ) / 20
    
    # Compute embedding change probabilities and execute embedding
    FI = FisherInformation * (2 * e - sgn_e) ** 4
    maxCostMat = np.zeros_like(FI, dtype=bool)
    maxCostMat[::8, ::8] = True
    maxCostMat[4::8, ::8] = True
    maxCostMat[::8, 4::8] = True
    maxCostMat[4::8, 4::8] = True
    FI[maxCostMat & (np.abs(e) > 0.4999)] = 1e10
    FI[np.abs(e) < 0.01] = 1e10
    FI = FI.flatten()
    
    S_COEFFS = C_STRUCT['coef_arrays'][0]
    
    # Ternary embedding change probabilities
    nzAC = np.sum(S_COEFFS.flatten() != 0) - np.sum(S_COEFFS[::8, ::8].flatten() != 0)
    messageLenght = round(Payload * nzAC * np.log(2))
    
    beta = BinaryProbs(FI, messageLenght)
    
    # Simulate embedding
    np.random.seed(0)
    r = np.random.rand(1, S_COEFFS.size)
    ModifPM1 = r < beta
    
    # Modifying X by +-1
    S_COEFFS[ModifPM1] = S_COEFFS[ModifPM1] + change[ModifPM1]
    
    # Taking care of boundary cases
    S_COEFFS[S_COEFFS > 1024] = 1024
    S_COEFFS[S_COEFFS < -1023] = -1023
    
    ChangeRate = np.sum(ModifPM1.flatten()) / S_COEFFS.size
    pChange = np.reshape(beta, S_COEFFS.shape)
    
    S_STRUCT = C_STRUCT.copy()
    S_STRUCT['coef_arrays'][0] = S_COEFFS
    
    Deflection = np.sum(pChange.flatten() * FI.flatten())
    
    return S_STRUCT, C_STRUCT, pChange, ChangeRate, Deflection

def RCgetJPEGmtx():
    cc, rr = np.meshgrid(np.arange(8), np.arange(8))
    T = np.sqrt(2 / 8) * np.cos(np.pi * (2 * cc + 1) * rr / 16)
    T[0, :] = T[0, :] / np.sqrt(2)
    dct8_mtx = np.zeros((64, 64))
    for i in range(64):
        dcttmp = np.zeros((8, 8))
        dcttmp.flat[i] = 1
        TTMP = np.dot(np.dot(T, dcttmp), T.T)
        dct8_mtx[:, i] = TTMP.flatten()
    return dct8_mtx

def VarianceEstimationDCT2D(Image, BlockSize, Degree):
    if BlockSize % 2 == 0:
        raise ValueError('The block dimensions should be odd!!')
    
    Degree = min(Degree, BlockSize)
    
    # number of parameters per block
    q = Degree * (Degree + 1) // 2
    
    # Build G matrix
    BaseMat = np.zeros((BlockSize, BlockSize))
    BaseMat[0, 0] = 1
    G = np.zeros((BlockSize ** 2, q))
    k = 0
    for xShift in range(1, Degree + 1):
        for yShift in range(1, Degree - xShift + 2):
            tmp = idct(idct(np.roll(np.roll(BaseMat, xShift - 1, axis=0), yShift - 1, axis=1)).T).T
            G[:, k] = np.reshape(tmp, (BlockSize ** 2,))
            k += 1
    
    # Estimate the variance
    PadSize = BlockSize // 2
    padded_image = np.pad(Image, ((PadSize, PadSize), (PadSize, PadSize)), mode='symmetric')
    I2C = image2cols(padded_image, BlockSize)
    PGorth = np.eye(BlockSize ** 2) - np.dot(G, np.linalg.solve(np.dot(G.T, G), G.T))
    EstimatedVariance = np.reshape(np.sum((np.dot(PGorth, I2C)) ** 2, axis=0) / (BlockSize ** 2 - q), Image.shape)
    
    return EstimatedVariance

def BinaryProbs(FI, payload):
    data_dir = "./JPEG python/data"
    mat_file = os.path.join(data_dir, "ixlnx2_logscale.mat")
    xlog, ylog = loadmat(mat_file, squeeze_me=True)['xlog'], loadmat(mat_file, squeeze_me=True)['ylog']
    
    # Initial search interval for lambda
    L, R = 10000, 60000
    fL = hBinary(invxlnx2_fast(L * FI, xlog, ylog)) - payload
    fR = hBinary(invxlnx2_fast(R * FI, xlog, ylog)) - payload
    
    # If the range [L,R] does not cover alpha enlarge the search interval
    while fL * fR > 0:
        if fL > 0:
            L = R
            fL = fR
            R = 2 * R
            fR = hBinary(invxlnx2_fast(R * FI, xlog, ylog)) - payload
        else:
            R = L
            fR = fL
            L = L / 2
            fL = hBinary(invxlnx2_fast(L * FI, xlog, ylog)) - payload
    
    # Search for the lambda in the specified interval
    i = 0
    M = (L + R) / 2
    fM = hBinary(invxlnx2_fast(M * FI, xlog, ylog)) - payload
    while np.abs(fM) > max(2, payload / 1000.0) and i < 20:
        if fL * fM < 0:
            R = M
            fR = fM
        else:
            L = M
            fL = fM
        i += 1
        M = (L + R) / 2
        fM = hBinary(invxlnx2_fast(M * FI, xlog, ylog)) - payload
    
    # Compute beta using the found lambda
    beta = invxlnx2_fast(M * FI, xlog, ylog)
    
    return beta

def invxlnx2_fast(y, xlog, ylog):
    x = np.zeros_like(y)
    i_large = y > 1000
    if np.sum(i_large) > 0:
        yz = y[i_large]
        z = yz / np.log(yz - 1)
        for j in range(3):
            z = yz / np.log(z - 1)
        x[i_large] = 1 / z
    
    i_small = y <= 1000
    if np.sum(i_small) > 0:
        z = y[i_small]
        indlog = np.floor((np.log2(z) + 25) / 0.02).astype(int) + 1
        indlog[indlog < 1] = 1
        x[i_small] = ylog[indlog] + (z - xlog[indlog]) / (xlog[indlog + 1] - xlog[indlog]) * (ylog[indlog + 1] - ylog[indlog])
    
    x[np.isnan(x)] = 0
    
    return x

def hBinary(Probs):
    H = -Probs * np.log(Probs) - (1 - Probs) * np.log(1 - Probs)
    H[Probs < 1e-10] = 0
    H[Probs > 1 - 1e-10] = 0
    H[np.isnan(Probs)] = 0
    Ht = np.sum(H)
    return Ht

def image2cols(image, block_size):
    if isinstance(block_size, int):
        block_size = (block_size, block_size)
    else:
        block_size = tuple(block_size)

    image_rows, image_cols = image.shape
    block_rows, block_cols = block_size

    rows = (image_rows + block_rows - 1) // block_rows
    cols = (image_cols + block_cols - 1) // block_cols

    image_reshaped = np.lib.stride_tricks.as_strided(
        image,
        shape=(rows, block_rows, cols, block_cols),
        strides=(
            image.strides[0] * block_rows,
            image.strides[0],
            image.strides[1] * block_cols,
            image.strides[1],
        ),
    )

    return image_reshaped.reshape(-1, block_rows * block_cols)