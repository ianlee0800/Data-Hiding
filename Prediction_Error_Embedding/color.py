"""
color.py - Color image handling module for PEE data hiding system

This module provides functions for:
1. Detecting whether an image is grayscale or color using YCbCr analysis
2. Processing color images by splitting into RGB channels and applying PEE to each channel
3. Providing utilities for color metrics calculation and channel manipulation

It extends the existing PEE system to support both grayscale and color images.
"""

import numpy as np
import cv2
import cupy as cp
from common import calculate_psnr, calculate_ssim, histogram_correlation, cleanup_memory

def is_grayscale(img, threshold=5.0):
    """
    Detect if an image is grayscale using YCbCr color space analysis.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image (3-channel BGR or single-channel grayscale)
    threshold : float, optional
        Standard deviation threshold for chroma channels (default: 5.0)
        Higher values may classify slightly colorful images as grayscale
        
    Returns:
    --------
    bool
        True if image is grayscale, False if it's color
    """
    # Check if image is already single-channel
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        return True
    
    # Convert BGR to YCbCr
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # Calculate standard deviation of Cb and Cr channels
    y, cr, cb = cv2.split(img_ycrcb)
    cb_std = np.std(cb)
    cr_std = np.std(cr)
    
    print(f"Image chroma standard deviation - Cb: {cb_std:.2f}, Cr: {cr_std:.2f}")
    
    # If standard deviation of chroma channels is very low, it's likely grayscale
    if cb_std < threshold and cr_std < threshold:
        return True
    else:
        return False

def read_image_auto(filepath):
    """
    Read an image from file and automatically determine if it's grayscale or color.
    
    Parameters:
    -----------
    filepath : str
        Path to the image file
        
    Returns:
    --------
    tuple
        (image, is_gray) - The image as numpy array and a boolean indicating if it's grayscale
    """
    # Read image in color
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError(f"Failed to read image: {filepath}")
        
    # Check if it's grayscale
    is_gray = is_grayscale(img)
    
    # If grayscale, convert to single channel
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"Detected grayscale image: {filepath}")
    else:
        print(f"Detected color image: {filepath}")
        
    return img, is_gray

def split_color_channels(img):
    """
    Split a color image into its RGB channels.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input color image (3-channel BGR format)
        
    Returns:
    --------
    tuple
        (b_channel, g_channel, r_channel) - BGR channels as separate arrays
    """
    # OpenCV uses BGR format, so split accordingly
    b_channel, g_channel, r_channel = cv2.split(img)
    return b_channel, g_channel, r_channel

def combine_color_channels(b_channel, g_channel, r_channel):
    """
    Combine separate channels back into a color image.
    
    Parameters:
    -----------
    b_channel : numpy.ndarray
        Blue channel
    g_channel : numpy.ndarray
        Green channel
    r_channel : numpy.ndarray
        Red channel
        
    Returns:
    --------
    numpy.ndarray
        Combined BGR color image
    """
    return cv2.merge([b_channel, g_channel, r_channel])

def calculate_color_metrics(original_img, embedded_img):
    """
    修正版的彩色圖像品質指標計算函數
    不包含BPP計算，BPP應該在調用處單獨計算
    """
    # 確保圖像是numpy格式
    if isinstance(original_img, cp.ndarray):
        original_img = cp.asnumpy(original_img)
    if isinstance(embedded_img, cp.ndarray):
        embedded_img = cp.asnumpy(embedded_img)
    
    # 計算PSNR for color image
    mse = np.mean((original_img.astype(np.float64) - embedded_img.astype(np.float64)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10((255.0 ** 2) / mse)
    
    # 計算SSIM for each channel and average
    b1, g1, r1 = cv2.split(original_img)
    b2, g2, r2 = cv2.split(embedded_img)
    
    ssim_b = calculate_ssim(b1, b2)
    ssim_g = calculate_ssim(g1, g2)
    ssim_r = calculate_ssim(r1, r2)
    ssim = (ssim_b + ssim_g + ssim_r) / 3.0
    
    # 計算histogram correlation for each channel and average
    hist_corr_b = histogram_correlation(
        np.histogram(b1, bins=256, range=(0, 255))[0],
        np.histogram(b2, bins=256, range=(0, 255))[0]
    )
    hist_corr_g = histogram_correlation(
        np.histogram(g1, bins=256, range=(0, 255))[0],
        np.histogram(g2, bins=256, range=(0, 255))[0]
    )
    hist_corr_r = histogram_correlation(
        np.histogram(r1, bins=256, range=(0, 255))[0],
        np.histogram(r2, bins=256, range=(0, 255))[0]
    )
    hist_corr = (hist_corr_b + hist_corr_g + hist_corr_r) / 3.0
    
    return psnr, ssim, hist_corr