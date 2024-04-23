import numpy as np
from scipy.signal import convolve2d
from scipy.fft import dct, idct

def SI_UNIWARD(precover, C_STRUCT, payload):
    """
    Embedding function for SI-UNIWARD steganography.
    
    :param precover: The precover image in spatial domain.
    :param C_STRUCT: The DCT coefficients and quantization tables of the JPEG compressed image.
    :param payload: The payload size in bits.
    :return: The stego image and the corresponding cost map.
    """
    # Get quantization table and DCT coefficients
    C_QUANT = C_STRUCT.quant_tables[0]
    
    # Compute DCT coefficients from precover
    def dct2(block):
        return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    xi = np.apply_along_axis(dct2, 2, precover.reshape((-1, 8, 8)) - 128)
    
    # Quantization
    def quantize(block):
        return block / C_QUANT
    
    DCT_real = np.apply_along_axis(quantize, 2, xi)
    DCT_rounded = np.round(DCT_real)
    C_STRUCT.coef_arrays[0] = DCT_rounded
    
    # Compute embedding change probabilities
    e = DCT_rounded - DCT_real
    sgn_e = np.sign(e)
    sgn_e[e == 0] = np.round(np.random.rand(np.sum(e == 0))) * 2 - 1
    change = -sgn_e
    
    # Compute costs
    MAX_COST = 1e13
    SIGMA = 2**(-6)
    
    # Get 2D wavelet filters - Daubechies 8
    hpdf = np.array([-0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837, 0.0158291053, 
                     -0.2840155430, -0.0004724846, 0.1287474266, 0.0173693010, -0.0440882539, 
                     -0.0139810279, 0.0087460940, 0.0048703530, -0.0003917404, -0.0006754494, 
                     -0.0001174768])
    lpdf = (-1)**np.arange(len(hpdf)) * hpdf[::-1]
    
    F = [np.outer(lpdf, hpdf), np.outer(hpdf, lpdf), np.outer(hpdf, hpdf)]
    
    # Precompute impact in spatial domain when a DCT coefficient is changed by 1
    spatialImpact = np.zeros((8, 8, 8, 8))
    for bcoord_i in range(8):
        for bcoord_j in range(8):
            testCoeffs = np.zeros((8, 8))
            testCoeffs[bcoord_i, bcoord_j] = 1
            spatialImpact[bcoord_i, bcoord_j] = idct(idct(testCoeffs, axis=0, norm='ortho'), axis=1, norm='ortho') * C_QUANT[bcoord_i, bcoord_j]
    
    # Precompute impact on wavelet coefficients when a DCT coefficient is changed by 1
    waveletImpact = np.zeros((len(F), 8, 8, 22, 22))
    for Findex in range(len(F)):
        for bcoord_i in range(8):
            for bcoord_j in range(8):
                waveletImpact[Findex, bcoord_i, bcoord_j] = convolve2d(spatialImpact[bcoord_i, bcoord_j], F[Findex], mode='full')
    
    # Create reference cover wavelet coefficients (LH, HL, HH)
    padSize = max(F[0].shape + F[1].shape)
    precover_padded = np.pad(precover, ((padSize, padSize), (padSize, padSize)), mode='symmetric')
    
    R_PC = [convolve2d(precover_padded, f, mode='valid') for f in F]
    
    k, l = precover.shape
    rho = np.zeros((k, l))
    
    for row in range(k):
        for col in range(l):
            sub_e = e[row, col]
            modRow = (row - 1) % 8
            modCol = (col - 1) % 8
            subRows = slice(row - modRow - 6 + padSize, row - modRow + 16 + padSize)
            subCols = slice(col - modCol - 6 + padSize, col - modCol + 16 + padSize)
            
            C_xi = np.zeros(3)
            S_xi = np.zeros(3)
            for fIndex in range(3):
                R_PC_sub = R_PC[fIndex][subRows, subCols]
                coeff_power = (bcoord_i, bcoord_j) = (modRow, modCol)
                wavCoverStegoDiff = waveletImpact[fIndex][coeff_power]
                C_xi[fIndex] = np.abs(wavCoverStegoDiff * sub_e) / (np.abs(R_PC_sub) + SIGMA)
                sign_power = sgn_e[row, col]
                S_xi[fIndex] = np.abs(wavCoverStegoDiff * (sub_e - sign_power)) / (np.abs(R_PC_sub) + SIGMA)
            
            C_rho = C_xi.sum()
            S_rho = S_xi.sum()
            rho[row, col] = S_rho - C_rho
    
    rho = np.clip(rho + 1e-4, 0, MAX_COST)
    rho[np.isnan(rho)] = MAX_COST
    rho[(DCT_rounded > 1022) & (e > 0)] = MAX_COST
    rho[(DCT_rounded < -1022) & (e < 0)] = MAX_COST
    
    maxCostMat = np.zeros_like(rho, dtype=bool)
    maxCostMat[::8, ::8] = True
    maxCostMat[4::8, ::8] = True
    maxCostMat[::8, 4::8] = True
    maxCostMat[4::8, 4::8] = True
    rho[maxCostMat & (np.abs(e) > 0.4999)] = MAX_COST
    
    # Compute message length for each run
    nzAC = np.count_nonzero(DCT_rounded) - np.count_nonzero(DCT_rounded[::8, ::8])
    totalMessageLength = round(payload * nzAC)
    
    # Embedding
    perm = np.random.permutation(DCT_rounded.size)
    DCT_rounded_flat = DCT_rounded.flatten()
    rho_flat = rho.flatten()
    
    LSBs, pChange = EmbeddingSimulator(DCT_rounded_flat[perm], rho_flat[perm], totalMessageLength)
    
    pChange_reshaped = np.zeros_like(DCT_rounded_flat)
    pChange_reshaped[perm] = pChange
    pChange = pChange_reshaped.reshape(DCT_rounded.shape)
    
    LSBs_reshaped = np.zeros_like(DCT_rounded_flat)
    LSBs_reshaped[perm] = LSBs
    LSBs = LSBs_reshaped.reshape(DCT_rounded.shape)
    
    # Create stego coefficients
    temp = DCT_rounded % 2
    S_COEFFS = np.zeros_like(DCT_rounded)
    S_COEFFS[temp == LSBs] = DCT_rounded[temp == LSBs]
    S_COEFFS[temp != LSBs] = DCT_rounded[temp != LSBs] + change[temp != LSBs]
    
    S_STRUCT = C_STRUCT.copy()
    S_STRUCT.coef_arrays[0] = S_COEFFS
    
    return S_STRUCT, C_STRUCT, pChange

def EmbeddingSimulator(x, rho, m):
    """
    Embedding simulator function.
    
    :param x: The cover DCT coefficients.
    :param rho: The cost map.
    :param m: The message length.
    :return: The LSBs of the stego coefficients and the embedding change probabilities.
    """
    x = x.astype(float)
    n = x.size
    rho = rho.flatten()
    
    def calc_lambda(rho, message_length, n):
        l3 = 1e+3
        m3 = message_length + 1
        iterations = 0
        while m3 > message_length:
            l3 *= 2
            p = 1 / (1 + np.exp(-l3 * rho))
            m3 = binary_entropy(p)
            iterations += 1
            if iterations > 10:
                return l3
        
        l1 = 0
        m1 = n
        lambda_val = 0
        alpha = message_length / n
        
        while (m1 - m3) / n > alpha / 1000.0 and iterations < 30:
            lambda_val = l1 + (l3 - l1) / 2
            p = 1 / (1 + np.exp(-lambda_val * rho))
            m2 = binary_entropy(p)
            if m2 < message_length:
                l3 = lambda_val
                m3 = m2
            else:
                l1 = lambda_val
                m1 = m2
            iterations += 1
        
        return lambda_val
    
    def binary_entropy(p):
        p = p.flatten()
        eps = 2.2204e-16
        p[p <= eps] = eps
        p[p >= 1 - eps] = 1 - eps
        Hb = -(p * np.log2(p)) - ((1 - p) * np.log2(1 - p))
        Hb[np.isnan(Hb)] = 0
        return Hb.sum()
    
    lambda_val = calc_lambda(rho, m, n)
    pChange = 1 - (1 / (1 + np.exp(-lambda_val * rho)))
    randChange = np.random.rand(x.size)
    flippedPixels = randChange < pChange
    LSBs = (x + flippedPixels) % 2
    
    return LSBs, pChange

if __name__ == '__main__':
    # Add your test code here
    pass