import cv2
import numpy as np
import math

# import zigzag functions
from zigzag import *

# defining block size
block_size = 8

# Quantization Matrix 
QUANTIZATION_MAT = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                             [12, 12, 14, 19, 26, 58, 60, 55],
                             [14, 13, 16, 24, 40, 57, 69, 56],
                             [14, 17, 22, 29, 51, 87, 80, 62],
                             [18, 22, 37, 56, 68, 109, 103, 77],
                             [24, 35, 55, 64, 81, 104, 113, 92],
                             [49, 64, 78, 87, 103, 121, 120, 101],
                             [72, 92, 95, 98, 112, 100, 103, 99]])

def adaptive_quantization(block, quality_factor):
    # Compute the adaptive quantization matrix based on the block's characteristics
    block_variance = np.var(block)
    if block_variance < 100:
        scale_factor = 1.0
    elif block_variance < 500:
        scale_factor = 0.8
    else:
        scale_factor = 0.6
    
    quantization_matrix = np.floor((QUANTIZATION_MAT * scale_factor * (quality_factor / 100))).astype(int)
    return quantization_matrix

def get_run_length_encoding(image):
    i = 0
    skip = 0
    stream = []    
    bitstream = ""
    image = image.astype(int)
    while i < image.shape[0]:
        if image[i] != 0:            
            stream.append((image[i],skip))
            bitstream = bitstream + str(image[i])+ " " +str(skip)+ " "
            skip = 0
        else:
            skip = skip + 1
        i = i + 1

    return bitstream

# reading image in grayscale style
img = cv2.imread('./DCT/images/lena.png', cv2.IMREAD_GRAYSCALE)

# get size of the image
[h, w] = img.shape

# No of blocks needed : Calculation
height = h
width = w
h = np.float32(h) 
w = np.float32(w) 

nbh = math.ceil(h/block_size)
nbh = np.int32(nbh)

nbw = math.ceil(w/block_size)
nbw = np.int32(nbw)

# Pad the image, because sometime image size is not dividable to block size
# get the size of padded image by multiplying block size by number of blocks in height/width
H = block_size * nbh
W = block_size * nbw

# create a numpy zero matrix with size of H,W
padded_img = np.zeros((H, W))

# copy the values of img into padded_img[0:h,0:w]
padded_img[0:height,0:width] = img[0:height,0:width]

cv2.imwrite('uncompressed.bmp', np.uint8(padded_img))

# start encoding:
# divide image into block size by block size (here: 8-by-8) blocks
# To each block apply 2D discrete cosine transform
# reorder DCT coefficients in zig-zag order
# reshaped it back to block size by block size (here: 8-by-8)

# define the quality factor (0-100)
quality_factor = 80

# start encoding:
for i in range(nbh):
    # Compute start and end row index of the block
    row_ind_1 = i * block_size                
    row_ind_2 = row_ind_1 + block_size
    
    for j in range(nbw):
        # Compute start & end column index of the block
        col_ind_1 = j * block_size                       
        col_ind_2 = col_ind_1 + block_size
                    
        block = padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2]
                   
        # apply 2D discrete cosine transform to the selected block                       
        DCT = cv2.dct(block.astype(np.float64))
        
        # perform adaptive quantization
        quantization_matrix = adaptive_quantization(block, quality_factor)
        DCT_quantized = np.round(DCT / quantization_matrix).astype(np.float64)
        
        # reorder DCT coefficients in zig zag order by calling zigzag function
        reordered = zigzag(DCT_quantized)

        # keep only the first 8 coefficients
        reordered[:8] = np.round(reordered[:8]).astype(np.float64)
        reordered[8:] = 0
        
        # reshape the reordered coefficients back to a 2D block
        reshaped = inverse_zigzag(reordered, block_size, block_size)
        
        # copy reshaped matrix into padded_img on current block corresponding indices
        padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2] = reshaped

cv2.imshow('encoded image', np.uint8(padded_img))

arranged = padded_img.flatten()

# apply run-length encoding to the flattened array
rle_encoded = get_run_length_encoding(arranged)

# write the run-length encoded data to a file
with open("compressed_data.txt", "w") as file:
    file.write(str(padded_img.shape[0]) + " " + str(padded_img.shape[1]) + "\n")
    file.write(rle_encoded)

cv2.waitKey(0)
cv2.destroyAllWindows()