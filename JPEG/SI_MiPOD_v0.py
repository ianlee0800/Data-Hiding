import numpy as np
from scipy.signal import wiener, dct, idct, loadmat

def SI_MiPODv0(preCover, C_STRUCT, Payload, Wie=2, BlkSz=3, Degree=3, F=None):
    if F is None:
        F = np.array([[1, 3, 1], [3, 2, 3], [1, 3, 1]])
        F = F / F.sum()

    wetConst = 1e10

    C_QUANT = C_STRUCT['quant_tables'][0]
    xi = dct2(preCover.astype(float) - 128, norm='ortho')
    DCT_real = xi / C_QUANT
    DCT_rounded = np.round(DCT_real)

    C_STRUCT['coef_arrays'][0] = DCT_rounded

    e = DCT_rounded - DCT_real
    sgn_e = np.sign(e)
    sgn_e[e == 0] = np.round(np.random.rand(np.sum(e == 0))) * 2 - 1
    change = -sgn_e

    nb_coul = len(C_STRUCT['coef_arrays'])
    VarianceDCT = np.zeros_like(preCover)
    for cc in range(nb_coul):
        WienerResidualCC = preCover[:, :, cc] - wiener(preCover[:, :, cc], [Wie, Wie])
        VarianceCC = VarianceEstimationDCT2D(WienerResidualCC, BlkSz, Degree)
        if cc == 0:
            VarianceDCT[:, :, cc] = blkproc(VarianceCC, [8, 8], lambda x: np.diag(dct8_mtx @ np.diag(x.flatten()) @ dct8_mtx.T).reshape(8, 8) / (C_STRUCT['quant_tables'][0] ** 2))
        else:
            VarianceDCT[:, :, cc] = blkproc(VarianceCC, [8, 8], lambda x: np.diag(dct8_mtx @ np.diag(x.flatten()) @ dct8_mtx.T).reshape(8, 8) / (C_STRUCT['quant_tables'][1] ** 2))

    VarianceDCT = np.maximum(VarianceDCT, np.sqrt(1 / wetConst))
    FisherInformation = 1 / VarianceDCT ** 2

    for cc in range(nb_coul):
        tmp = np.zeros((FisherInformation.shape[0] + 16, FisherInformation.shape[1] + 16, FisherInformation.shape[2]))
        tmp[9:-8, 9:-8] = FisherInformation
        tmp[:8, :] = tmp[8:16, :]
        tmp[-7:, :] = tmp[-15:-8, :]
        tmp[:, :8] = tmp[:, 8:16]
        tmp[:, -7:] = tmp[:, -15:-8]
        FisherInformation[:, :, cc] = (
            tmp[:-16, :-16] * F[0, 0] + tmp[8:-8, :-16] * F[1, 0] + tmp[16:, :-16] * F[2, 0] +
            tmp[:-16, 8:-8] * F[0, 1] + tmp[8:-8, 8:-8] * F[1, 1] + tmp[16:, 8:-8] * F[2, 1] +
            tmp[:-16, 16:] * F[0, 2] + tmp[8:-8, 16:] * F[1, 2] + tmp[16:, 16:] * F[2, 2]
        )

    FI = FisherInformation * (2 * e - sgn_e) ** 2
    maxCostMat = np.zeros_like(FI, dtype=bool)
    maxCostMat[::8, ::8] = True
    maxCostMat[4::8, ::8] = True
    maxCostMat[::8, 4::8] = True
    maxCostMat[4::8, 4::8] = True
    FI[maxCostMat & (np.abs(e) > 0.4999)] = wetConst
    FI[np.abs(e) < 0.01] = wetConst
    FI = FI.flatten()

    S_COEFFS = np.zeros_like(VarianceDCT)
    for cc in range(nb_coul):
        S_COEFFS[:, :, cc] = C_STRUCT['coef_arrays'][cc]

    nzAC = np.count_nonzero(S_COEFFS) - np.count_nonzero(S_COEFFS[::8, ::8])
    messageLenght = int(np.round(Payload * nzAC * np.log2(3)))

    beta = TernaryProbs(FI, messageLenght)

    r = np.random.rand(*S_COEFFS.shape)
    ModifPM1 = (r < beta.reshape(S_COEFFS.shape))
    S_COEFFS[ModifPM1] = S_COEFFS[ModifPM1] + change[ModifPM1]
    S_COEFFS = np.clip(S_COEFFS, -1023, 1024)
    ChangeRate = np.count_nonzero(ModifPM1) / ModifPM1.size
    pChange = beta.reshape(S_COEFFS.shape)

    S_STRUCT = C_STRUCT.copy()
    for cc in range(nb_coul):
        S_STRUCT['coef_arrays'][cc] = S_COEFFS[:, :, cc]

    Deflection = np.sum(pChange * FI.reshape(pChange.shape))

    return S_STRUCT, pChange, ChangeRate, Deflection

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

def hBinary(probs):
    p = np.clip(probs, 1e-10, 1 - 1e-10)
    return -np.sum(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

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

def TernaryProbs(FI, payload):
    data = loadmat('ixlnx2.mat')
    xx = data['xx'][0]
    yy = data['yy'][0]

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
            G[:, k] = idct(idct(np.eye(BlockSize)[:, :, np.newaxis] * (i, j), norm='ortho', axis=0), norm='ortho', axis=1).flatten()
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
