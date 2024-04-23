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
    v, r, c = im2vec(a, (n, n))
    b = vec2im(np.dot(dctm, v), bsize=(n, n), rows=r, cols=c)
    return b

def bdctmtx(n):
    """
    Blocked discrete cosine transform matrix
    
    Parameters:
    n (int): Block size
    
    Returns:
    numpy.ndarray: DCT2 transform matrix of size n^2 x n^2
    """
    c, r = np.meshgrid(range(n), range(n))
    c0, r0 = np.meshgrid(c, c)
    c1, r1 = np.meshgrid(r, r)
    
    x = np.sqrt(2 / n) * np.cos(np.pi * (2 * c + 1) * r / (2 * n))
    x[0, :] = x[0, :] / np.sqrt(2)
    
    m = x[r0 + c0 * n] * x[r1 + c1 * n]
    
    return m

def im2vec(im, bsize, padsize=0):
    """
    Reshape 2D image blocks into an array of column vectors
    
    Parameters:
    im (numpy.ndarray): Input image
    bsize (tuple): Block size (rows, cols)
    padsize (tuple): Padding size (rows, cols) (default: (0, 0))
    
    Returns:
    tuple: (v, rows, cols)
        v (numpy.ndarray): Reshaped image blocks
        rows (int): Number of rows of blocks
        cols (int): Number of columns of blocks
    """
    bsize = np.array(bsize)
    padsize = np.array(padsize)
    
    if np.any(padsize < 0):
        raise ValueError("Pad size must not be negative.")
    
    imsize = np.array(im.shape)
    y, x = bsize + padsize
    rows = int(np.floor((imsize[0] + padsize[0]) / y))
    cols = int(np.floor((imsize[1] + padsize[1]) / x))
    
    t = np.zeros((y * rows, x * cols))
    imy = y * rows - padsize[0]
    imx = x * cols - padsize[1]
    t[:imy, :imx] = im[:imy, :imx]
    
    t = t.reshape(y, rows, x, cols)
    t = t.transpose(0, 2, 1, 3).reshape(y, x, rows * cols)
    
    v = t[:bsize[0], :bsize[1], :rows*cols]
    v = v.reshape(y * x, rows * cols)
    
    return v, rows, cols

def vec2im(v, padsize=0, bsize=None, rows=None, cols=None):
    """
    Reshape and combine column vectors into a 2D image
    
    Parameters:
    v (numpy.ndarray): Input vector array
    padsize (tuple): Padding size (rows, cols) (default: (0, 0))
    bsize (tuple): Block size (rows, cols) (default: square root of vector length)
    rows (int): Number of rows of blocks (default: floor(sqrt(num_vectors)))
    cols (int): Number of columns of blocks (default: ceil(num_vectors/rows))
    
    Returns:
    numpy.ndarray: Output image
    """
    m, n = v.shape
    padsize = np.array(padsize)
    
    if np.any(padsize < 0):
        raise ValueError("Pad size must not be negative.")
    
    if bsize is None:
        bsize = int(np.floor(np.sqrt(m)))
    bsize = np.array(bsize)
    
    if np.prod(bsize) != m:
        raise ValueError("Block size does not match size of input vectors.")
    
    if rows is None:
        rows = int(np.floor(np.sqrt(n)))
    if cols is None:
        cols = int(np.ceil(n / rows))
    
    y, x = bsize + padsize
    t = np.zeros((y, x, rows * cols))
    t[:bsize[0], :bsize[1], :n] = v.reshape(bsize[0], bsize[1], n)
    
    t = t.reshape(y, x, rows, cols)
    t = t.transpose(0, 2, 1, 3).reshape(y * rows, x * cols)
    
    im = t[:y*rows-padsize[0], :x*cols-padsize[1]]
    
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
    v, r, c = im2vec(a, (n, n))
    b = vec2im(np.dot(dctm.T, v), bsize=(n, n), rows=r, cols=c)
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