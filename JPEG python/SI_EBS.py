import numpy as np
from scipy.fft import dct

def SI_EBS(precover, C_STRUCT, payload):
    wetConst = 10 ** 13
    C_QUANT = C_STRUCT['quant_tables'][0]

    def dct2(block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    xi = np.apply_along_axis(dct2, 2, precover.reshape((-1, 8, 8)) - 128)

    def quantize(block):
        return block / C_QUANT

    DCT_real = np.apply_along_axis(quantize, 2, xi)
    DCT_rounded = np.round(DCT_real)
    C_STRUCT['coef_arrays'][0] = DCT_rounded
    
    # Block Entropy Cost
    rho_ent = np.zeros_like(DCT_rounded)
    for row in range(DCT_rounded.shape[0] // 8):
        for col in range(DCT_rounded.shape[1] // 8):
            all_coeffs = DCT_rounded[row*8:row*8+8, col*8:col*8+8]
            all_coeffs[0, 0] = 0  # remove DC
            nzAC_coeffs = all_coeffs[all_coeffs != 0]
            nzAC_unique_coeffs = np.unique(nzAC_coeffs)
            if nzAC_unique_coeffs.size > 1:
                b = np.histogram(nzAC_coeffs, bins=nzAC_unique_coeffs)[0]
                b = b[b != 0]
                p = b / np.sum(b)
                H_block = -np.sum(p * np.log(p))
            else:
                H_block = 0
            rho_ent[row*8:row*8+8, col*8:col*8+8] = 1 / (H_block ** 2)
    
    # Rounding Error Cost
    e_ri = DCT_rounded - DCT_real
    sgn_e = np.sign(e_ri)
    sgn_e[e_ri == 0] = np.round(np.random.rand(np.sum(e_ri == 0))) * 2 - 1
    change = -sgn_e
    qi = np.tile(C_QUANT, np.array(DCT_rounded.shape) // 8)
    rho_f = ((0.5 - np.abs(e_ri)) * qi) ** 2
    
    # Final cost
    rho = rho_ent * rho_f
    rho = rho + 10 ** (-4)
    rho[rho > wetConst] = wetConst
    rho[np.isnan(rho)] = wetConst
    rho[(DCT_rounded > 1022) & (e_ri < 0)] = wetConst
    rho[(DCT_rounded < -1022) & (e_ri > 0)] = wetConst
    
    # Compute message length for each run
    nzAC = np.count_nonzero(DCT_rounded) - np.count_nonzero(DCT_rounded[::8, ::8])
    totalMessageLength = round(payload * nzAC)
    
    # Embedding
    perm = np.random.permutation(DCT_rounded.size)
    LSBs, pChange = EmbeddingSimulator(DCT_rounded.flatten()[perm], rho.flatten()[perm], totalMessageLength)
    LSBs = LSBs[np.argsort(perm)].reshape(DCT_rounded.shape)
    
    temp = DCT_rounded % 2
    S_COEFFS = np.zeros_like(DCT_rounded)
    S_COEFFS[temp == LSBs] = DCT_rounded[temp == LSBs]
    S_COEFFS[temp != LSBs] = DCT_rounded[temp != LSBs] + change[temp != LSBs]
    
    S_STRUCT = C_STRUCT.copy()
    S_STRUCT['coef_arrays'][0] = S_COEFFS
    S_STRUCT['dc_huff_tables'] = []
    S_STRUCT['ac_huff_tables'] = []
    S_STRUCT['optimize_coding'] = 1
    
    pChange = pChange[np.argsort(perm)].reshape(DCT_rounded.shape)
    
    return S_STRUCT, C_STRUCT, pChange

def EmbeddingSimulator(x, rho, m):
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
            m3 = binary_entropyf(p)
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
            m2 = binary_entropyf(p)
            if m2 < message_length:
                l3 = lambda_val
                m3 = m2
            else:
                l1 = lambda_val
                m1 = m2
            iterations += 1
        
        return lambda_val
    
    def binary_entropyf(p):
        p = p.flatten()
        Hb = (-p * np.log2(p)) - ((1 - p) * np.log2(1 - p))
        Hb[np.isnan(Hb)] = 0
        return np.sum(Hb)
    
    lambda_val = calc_lambda(rho, m, n)
    pChange = 1 - (1 / (1 + np.exp(-lambda_val * rho)))
    randChange = np.random.rand(x.size)
    flippedPixels = randChange < pChange
    LSBs = (x + flippedPixels) % 2
    
    return LSBs, pChange