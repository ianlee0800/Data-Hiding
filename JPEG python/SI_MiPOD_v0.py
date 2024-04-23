import numpy as np
from scipy.signal import wiener
from scipy.fftpack import dct, idct
from scipy.io import loadmat
import os

def SI_MiPODv0(preCover, C_STRUCT, Payload, Wie=2, BlkSz=3, Degree=3, F=None):
    if F is None:
        F = np.array([[1, 3, 1], [3, 2, 3], [1, 3, 1]])
        F = F / np.sum(F)
    
    wetConst = 1e10
    C_QUANT = C_STRUCT['quant_tables'][0]
    
    def dct2(block):
        return dct(dct(block.T).T)
    
    xi = np.apply_along_axis(dct2, 2, preCover.reshape((-1, 8, 8)) - 128)
    
    def quantize(block):
        return block / C_QUANT
    
    DCT_real = np.apply_along_axis(quantize, 2, xi)
    DCT_rounded = np.round(DCT_real)
    C_STRUCT['coef_arrays'][0] = DCT_rounded
    e = DCT_rounded - DCT_real
    
    sgn_e = np.sign(e)
    sgn_e[e == 0] = np.round(np.random.rand(np.sum(e == 0))) * 2 - 1
    change = -sgn_e
    
    MatDCT = RCgetJPEGmtx()
    nb_coul = len(C_STRUCT['coef_arrays'])
    
    VarianceDCT = np.zeros(preCover.shape)
    for cc in range(nb_coul):
        WienerResidualCC = preCover[:, :, cc] - wiener(preCover[:, :, cc], (Wie, Wie))
        VarianceCC = VarianceEstimationDCT2D(WienerResidualCC, BlkSz, Degree)
        
        if cc == 0:
            def funVar(x):
                return np.reshape(np.diag(np.dot(np.dot(MatDCT, np.diag(x.flatten())), MatDCT.T)), (8, 8)) / (C_STRUCT['quant_tables'][0] ** 2)
        else:
            def funVar(x):
                return np.reshape(np.diag(np.dot(np.dot(MatDCT, np.diag(x.flatten())), MatDCT.T)), (8, 8)) / (C_STRUCT['quant_tables'][1] ** 2)
        
        VarianceDCT[:, :, cc] = np.apply_along_axis(funVar, 2, VarianceCC.reshape((-1, 8, 8)))
    
    VarianceDCT[VarianceDCT < np.sqrt(1 / wetConst)] = np.sqrt(1 / wetConst)
    
    FisherInformation = 1 / VarianceDCT ** 2
    
    for cc in range(nb_coul):
        tmp = np.zeros((FisherInformation.shape[0] + 16, FisherInformation.shape[1] + 16))
        tmp[8:-8, 8:-8] = FisherInformation[:, :, cc]
        tmp[:8, :] = tmp[8:16, :]
        tmp[-8:, :] = tmp[-16:-8, :]
        tmp[:, :8] = tmp[:, 8:16]
        tmp[:, -8:] = tmp[:, -16:-8]
        
        FisherInformation[:, :, cc] = (
            tmp[:-16, :-16] * F[0, 0] +
            tmp[8:-8, :-16] * F[1, 0] +
            tmp[16:, :-16] * F[2, 0] +
            tmp[:-16, 8:-8] * F[0, 1] +
            tmp[8:-8, 8:-8] * F[1, 1] +
            tmp[16:, 8:-8] * F[2, 1] +
            tmp[:-16, 16:] * F[0, 2] +
            tmp[8:-8, 16:] * F[1, 2] +
            tmp[16:, 16:] * F[2, 2]
        )
    
    FI = FisherInformation * (2 * e - sgn_e) ** 2
    maxCostMat = np.zeros(FI.shape, dtype=bool)
    maxCostMat[::8, ::8] = True
    maxCostMat[4::8, ::8] = True
    maxCostMat[::8, 4::8] = True
    maxCostMat[4::8, 4::8] = True
    FI[maxCostMat & (np.abs(e) > 0.4999)] = wetConst
    FI[np.abs(e) < 0.01] = wetConst
    FI = FI.flatten()
    
    S_COEFFS = np.zeros(VarianceDCT.shape)
    for cc in range(nb_coul):
        S_COEFFS[:, :, cc] = C_STRUCT['coef_arrays'][cc]
    
    nzAC = np.sum(S_COEFFS != 0) - np.sum(S_COEFFS[::8, ::8, :] != 0)
    messageLenght = round(Payload * nzAC * np.log(2))
    
    beta = TernaryProbs(FI, messageLenght)
    
    r = np.random.rand(*S_COEFFS.shape)
    ModifPM1 = r < beta
    S_COEFFS[ModifPM1] = S_COEFFS[ModifPM1] + change[ModifPM1]
    S_COEFFS[S_COEFFS > 1024] = 1024
    S_COEFFS[S_COEFFS < -1023] = -1023
    
    ChangeRate = np.sum(ModifPM1) / S_COEFFS.size
    pChange = beta.reshape(S_COEFFS.shape)
    
    S_STRUCT = C_STRUCT.copy()
    for cc in range(nb_coul):
        S_STRUCT['coef_arrays'][cc] = S_COEFFS[:, :, cc]
    
    Deflection = np.sum(pChange.flatten() * FI.flatten())
    
    return S_STRUCT, C_STRUCT, pChange, ChangeRate, Deflection

def RCgetJPEGmtx():
    cc, rr = np.meshgrid(np.arange(8), np.arange(8))
    T = np.sqrt(2 / 8) * np.cos(np.pi * (2 * cc + 1) * rr / 16)
    T[0, :] /= np.sqrt(2)
    dct8_mtx = np.zeros((64, 64))
    for i in range(64):
        dcttmp = np.zeros((8, 8))
        dcttmp.flat[i] = 1
        TTMP = np.dot(np.dot(T, dcttmp), T.T)
        dct8_mtx[:, i] = TTMP.flatten()
    return dct8_mtx

def RCdecompressJPEG(imJPEG):
    nb_coul = len(imJPEG['coef_arrays'])
    cc, rr = np.meshgrid(np.arange(8), np.arange(8))
    T = np.sqrt(2 / 8) * np.cos(np.pi * (2 * cc + 1) * rr / 16)
    T[0, :] /= np.sqrt(2)
    dct8_mtx = np.zeros((64, 64))
    for i in range(64):
        dcttmp = np.zeros((8, 8))
        dcttmp.flat[i] = 1
        TTMP = np.dot(np.dot(T, dcttmp), T.T)
        dct8_mtx[:, i] = TTMP.flatten()
    
    imDecompress = np.zeros((imJPEG['coef_arrays'][0].shape + (nb_coul,)))
    for cc in range(nb_coul):
        DCTcoefs = imJPEG['coef_arrays'][cc]
        QM = imJPEG['quant_tables'][0] if cc == 0 else imJPEG['quant_tables'][1]
        
        def funIDCT(x):
            return np.dot(np.dot(T.T, x * QM), T)
        
        imDecompress[:, :, cc] = np.apply_along_axis(funIDCT, 2, DCTcoefs.reshape((-1, 8, 8)))
    
    return imDecompress, dct8_mtx

def VarianceEstimationDCT2D(Image, BlockSize, Degree):
    if BlockSize % 2 != 0:
        raise ValueError('The block dimensions should be odd!!')
    if Degree > BlockSize:
        raise ValueError('Number of basis vectors exceeds block dimension!!')
    
    q = Degree * (Degree + 1) // 2
    BaseMat = np.zeros((BlockSize, BlockSize))
    BaseMat[0, 0] = 1
    G = np.zeros((BlockSize ** 2, q))
    k = 0
    for xShift in range(1, Degree + 1):
        for yShift in range(1, Degree - xShift + 2):
            G[:, k] = np.reshape(idct(idct(np.roll(np.roll(BaseMat, xShift - 1, axis=0), yShift - 1, axis=1)).T).T, (BlockSize ** 2, 1))
            k += 1
    
    PadSize = BlockSize // 2
    I2C = image2cols(np.pad(Image, ((PadSize, PadSize), (PadSize, PadSize)), mode='symmetric'), (BlockSize, BlockSize))
    PGorth = np.eye(BlockSize ** 2) - np.dot(G, np.linalg.solve(np.dot(G.T, G), G.T))
    EstimatedVariance = np.reshape(np.sum(np.dot(PGorth, I2C) ** 2, axis=0) / (BlockSize ** 2 - q), Image.shape)
    
    return EstimatedVariance

def TernaryProbs(FI, payload):
    data_dir = "./JPEG python/data"
    mat_file = os.path.join(data_dir, "ixlnx2.mat")
    xx, yy = loadmat(mat_file)['xx'], loadmat(mat_file)['yy']
    
    L, R = 5 * 10 ** -1.5, 5 * 10 ** 0.5
    fL = hBinary(invxlnx2_fast(L * FI, xx, yy)) - payload
    fR = hBinary(invxlnx2_fast(R * FI, xx, yy)) - payload
    
    while fL * fR > 0:
        if fL > 0:
            R *= 2
            fR = hBinary(invxlnx2_fast(R * FI, xx, yy)) - payload
        else:
            L /= 2
            fL = hBinary(invxlnx2_fast(L * FI, xx, yy)) - payload
    
    i, fM, TM = 0, 1, np.zeros((20, 2))
    while abs(fM) > 0.001 and i < 20:
        M = (L + R) / 2
        fM = hBinary(invxlnx2_fast(M * FI, xx, yy)) - payload
        if fL * fM < 0:
            R, fR = M, fM
        else:
            L, fL = M, fM
        i += 1
        TM[i - 1] = [fM, M]
    
    if i == 20:
        M = TM[np.argmin(np.abs(TM[:, 0])), 1]
    
    beta = invxlnx2_fast(M * FI, xx, yy)
    
    return beta

def invxlnx2_fast(y, xx, yy):
    x = np.zeros_like(y)
    i_large = y > 1000
    
    if np.sum(i_large) > 0:
        z = y[i_large] / np.log(y[i_large] - 1)
        for j in range(20):
            z = y[i_large] / np.log(z - 1)
        x[i_large] = 1 / z
    
    i_small = y <= 1000
    
    if np.sum(i_small) > 0:
        z = y[i_small]
        N = xx.size
        M = z.size
        comparison = np.dot(z.reshape((-1, 1)), np.ones((1, N))) >= np.dot(np.ones((M, 1)), xx.reshape((1, -1)))
        ind = np.sum(comparison, axis=1)
        x[i_small] = yy[ind] + (z - xx[ind]) / (xx[ind + 1] - xx[ind]) * (yy[ind + 1] - yy[ind])
    
    return x

def hBinary(Probs):
    p0 = 1 - Probs
    P = np.vstack((p0.flatten(), Probs.flatten()))
    H = -(P * np.log(P))
    H[P < np.finfo(float).eps] = 0
    Ht = np.nansum(H)
    return Ht

def image2cols(image, block_size):
    rows, cols = image.shape
    return image.reshape((rows // block_size[0], block_size[0], -1, block_size[1])).swapaxes(1, 2).reshape((-1, block_size[0] * block_size[1]))