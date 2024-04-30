import numpy as np

def bdct(a, n=8):
    """
    Blocked discrete cosine transform
    
    Parameters:
    a (numpy.ndarray): Input image
    n (int): Block size (default: 8)
    
    Returns:
    numpy.ndarray: DCT2 transformed image
    """
    dctm = bdctmtx(n)
    v = im2vec(a, (n, n))
    b = vec2im(np.dot(dctm, v), a.shape, (n, n))
    return b

def bdctmtx(n):
    """
    Blocked discrete cosine transform matrix

    Parameters:
    n (int): Block size

    Returns:
    numpy.ndarray: DCT2 transform matrix of size n^2 x n^2
    """
    c = np.arange(n).reshape(-1, 1)
    r = np.arange(n).reshape(1, -1)

    x = np.sqrt(2 / n) * np.cos(np.pi * (2 * c + 1) * r / (2 * n))
    x[0, :] = x[0, :] / np.sqrt(2)

    m = x * x.T

    return m

def im2vec(im, bsize, padsize=0):
    bsize = np.array(bsize)
    
    if isinstance(padsize, int):
        padsize = np.array([padsize, padsize])
    else:
        padsize = np.array(padsize)
    
    if np.any(padsize < 0):
        raise ValueError("Pad size must not be negative.")
    
    imsize = np.array(im.shape)
    y, x = bsize + padsize
    blocks_row = int(np.ceil((imsize[0] + padsize[0]) / y))
    blocks_col = int(np.ceil((imsize[1] + padsize[1]) / x))
    
    t = np.zeros((y * blocks_row, x * blocks_col))
    
    # Pad the input image with zeros
    padded_im = np.pad(im, ((padsize[0], y * blocks_row - imsize[0] - padsize[0]),
                            (padsize[1], x * blocks_col - imsize[1] - padsize[1])),
                       mode='constant', constant_values=0)
    
    t[:padded_im.shape[0], :padded_im.shape[1]] = padded_im
    
    v = t.reshape(y, blocks_row, x, blocks_col).transpose(1, 3, 0, 2).reshape(blocks_row, blocks_col, y * x)
    
    return v

def vec2im(v, imshape, bsize):
    """
    Reshape and combine a 2D array into a 2D image
    
    Parameters:
    v (numpy.ndarray): Input 2D array
    imshape (tuple): Shape of the output image
    bsize (tuple): Block size (rows, cols)
    
    Returns:
    numpy.ndarray: Output image
    """
    bsize = np.array(bsize)
    
    y, x = bsize
    rows = imshape[0] // y
    cols = imshape[1] // x
    
    t = v.reshape(y, x, rows, cols)
    t = t.transpose(0, 2, 1, 3).reshape(y * rows, x * cols)
    
    im = t[:imshape[0], :imshape[1]]
    
    return im

def ibdct(a, n=8):
    """
    Inverse blocked discrete cosine transform
    
    Parameters:
    a (numpy.ndarray): Input DCT coefficients
    n (int): Block size (default: 8)
    
    Returns:
    numpy.ndarray: Reconstructed image
    """
    dctm = bdctmtx(n)
    v = im2vec(a, (n, n))
    b = vec2im(np.dot(dctm.T, v), a.shape, (n, n))
    return b

def quantize(coef, qtable):
    """
    Quantize BDCT coefficients
    
    Parameters:
    coef (numpy.ndarray): Input DCT coefficients
    qtable (numpy.ndarray): Quantization table
    
    Returns:
    numpy.ndarray: Quantized coefficients
    """
    blksz = qtable.shape
    v, r, c = im2vec(coef, blksz)
    qcoef = vec2im(np.round(v / np.tile(qtable.ravel(), (v.shape[1], 1)).T), bsize=blksz, rows=r, cols=c)
    return qcoef

def dequantize(qcoef, qtable):
    """
    Dequantize BDCT coefficients
    
    Parameters:
    qcoef (numpy.ndarray): Input quantized coefficients
    qtable (numpy.ndarray): Quantization table
    
    Returns:
    numpy.ndarray: Dequantized coefficients
    """
    blksz = qtable.shape
    v, r, c = im2vec(qcoef, blksz)
    coef = vec2im(v * np.tile(qtable.ravel(), (v.shape[1], 1)).T, bsize=blksz, rows=r, cols=c)
    return coef

def jpeg_qtable(quality=50, tnum=0, force_baseline=False):
    """
    Generate standard JPEG quantization tables
    
    Parameters:
    quality (int): Quality factor (1-100) (default: 50)
    tnum (int): Table number (0 or 1) (default: 0)
    force_baseline (bool): Force baseline compatibility (default: False)
    
    Returns:
    numpy.ndarray: Quantization table
    """
    if quality <= 0:
        quality = 1
    if quality > 100:
        quality = 100
    if quality < 50:
        quality = 5000 / quality
    else:
        quality = 200 - quality * 2
    
    if tnum == 0:
        t = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]])
    elif tnum == 1:
        t = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                      [18, 21, 26, 66, 99, 99, 99, 99],
                      [24, 26, 56, 99, 99, 99, 99, 99],
                      [47, 66, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99]])
    else:
        raise ValueError("Table number must be 0 or 1")
    
    t = np.floor((t * quality + 50) / 100)
    t[t < 1] = 1
    t[t > 32767] = 32767
    if force_baseline:
        t[t > 255] = 255
    
    return t