import cv2
import numpy as np
import math
import random

# Load the original image
imgName = input("image name: ")
fileType = "png"
origImg = cv2.imread("./DE/images/%s.%s"%(imgName, fileType), cv2.IMREAD_GRAYSCALE)
height, width = origImg.shape
size = height*width

# Ensure the image is loaded successfully
if origImg is not None:
    height, width = origImg.shape
    size = height * width

    # Initialize the jump list
    jump = []

    # Initialize the hideImg array
    hideImg = np.zeros((height, width), int)

    for y in range(height):
        for x in range(width):
            if not hideImg[y, x]:
                jump.append((y, x))

    # Ensure 'jump' has enough positions for embedding
    while len(jump) < 8:
        y = random.randint(0, height - 1)
        x = random.randint(0, width - 1)
        if not hideImg[y, x]:
            jump.append((y, x))

    # Generate a random number
    random_number = random.randint(0, 255)
    print("Random Number:", random_number)

    # Create a markedImg as a copy of the original image
    markedImg = origImg.copy()

    # Now, you can use the 'jump' list in the loop to hide data
    for i in range(8):
        # Access the elements from the 'jump' list
        y, x = jump[i]

#計算峰值訊噪比PSNR
def calculate_psnr(img1, img2):
    height, width = img1.shape
    size = height*width
    max = 255
    mse = 0
    for y in range(height):
        for x in range(width):
            bgr1 = int(img1[y,x])
            bgr2 = int(img2[y,x])
            # print(img1, img2)
            diff = bgr1 - bgr2
            mse += diff**2
    if mse == 0:
        return 0
    mse = mse / size
    psnr = 10*math.log10(max**2/mse)
    psnr = round(psnr, 2)
    return psnr
#計算結構相似性SSIM
def calculate_ssim(img1, img2):
    height, width = img1.shape
    size = height*width
    sum1 = 0
    sum2 = 0
    sd1 = 0
    sd2 = 0
    cov12 = 0.0 #共變異數
    c1 = (255*0.01)**2 #其他參數
    c2 = (255*0.03)**2
    c3 = c2/2
    for y in range(height):
        for x in range(width):
            bgr1 = int(img1[y,x])
            bgr2 = int(img2[y,x])
            sum1 += bgr1
            sum2 += bgr2
    mean1 = sum1 / size
    mean2 = sum2 / size
    for y in range(height):
        for x in range(width):
            bgr1 = int(img1[y,x])
            bgr2 = int(img2[y,x])
            diff1 = bgr1 - mean1
            diff2 = bgr2 - mean2
            sd1 += diff1**2
            sd2 += diff2**2
            cov12 += (diff1*diff2)
    sd1 = math.pow(sd1/(size-1), 0.5)
    sd2 = math.pow(sd2/(size-1), 0.5)
    cov12 = cov12/(size-1)
    light = (2*mean1*mean2+c1)/(mean1**2+mean2**2+c1) #比較亮度
    contrast = (2*sd1*sd2+c2)/(sd1**2+sd2**2+c2) #比較對比度
    structure = (cov12+c3)/(sd1*sd2+c3) #比較結構
    ssim = light*contrast*structure #結構相似性
    ssim = round(ssim, 4)
    return ssim
#兩陣列是否相等
def same_array(array1, array2):
    for x in range(len(array1)):
        if array1[x] != array2[x]:
            return 0
    return 1
#差值擴張
def difference_expansion(left, right, hide):
    l = int((left+right)/2)
    if left >= right:
        h = left-right
        h_e = 2*h+hide
        left_e = l+int((h_e+1)/2)
        right_e = l-int(h_e/2)
    elif left < right:
        h = right-left
        h_e = 2*h+hide
        left_e = l-int(h_e/2)
        right_e = l+int((h_e+1)/2)
    return left_e, right_e

# Now, you can use the 'jump' list in the loop to hide data
for i in range(8):
    # Access the elements from the 'jump' list
    y, x = jump[i]
    
    # Rest of the code for embedding the random number


#藏入亂數
for i in range(8):
    bit = (random_number >> i) & 1
    y, x = jump[i]
    left = int(markedImg[y, x])
    right = int(markedImg[y, x + 1])
    left_e, right_e = difference_expansion(left, right, bit)
    markedImg[y, x] = left_e
    markedImg[y, x + 1] = right_e
    hideImg[y, x] = 1
    hideImg[y, x + 1] = 1

# Save the stego-image with the embedded random number
cv2.imwrite("./outcome/%s_marked_with_random.%s" % (imgName, fileType), markedImg)

#隱藏
markedImg = origImg.copy()
hideImg = np.zeros((height, width), int)
payload = 0 #藏量
pr_0 = 0.7
pr_1 = 0.3
insertArray = []
jump = [] #跳過的像素位置
jumpNumber = 0
i = 0 
#找出無法隱藏的位置
for y in range(height):
    x = 0
    while x <= width-2:
        left = int(markedImg[y,x]) #左像素值
        right = int(markedImg[y,x+1]) #右像素值
        b = np.random.choice([0,1], p=[pr_0,pr_1]) #嵌入位元值
        left_e, right_e = difference_expansion(left, right, b) #變更後的左右像素值
        #挑選可隱藏像素
        if left_e >= 0 and left_e <= 255 and right_e >= 0 and right_e <= 255:
            markedImg[y,x] = left_e
            markedImg[y,x+1] = right_e
            hideImg[y,x] = 1
            hideImg[y,x+1] = 1
            insertArray.append(b)
            payload += 1
            x += 2
        else:
            jumpNumber += 1
            jump.append([y,x])
            x += 1
jump = np.array(jump)
cv2.imwrite("./outcome/%s_marked.%s"%(imgName, fileType), markedImg)
# print("隱藏值隨機嵌入影像儲存成功")
bbp = payload/size
psnr = calculate_psnr(origImg, markedImg)
ssim = calculate_ssim(origImg, markedImg)
print("payload=%s"%payload)
print("bbp=%.4f"%bbp)
print("PSNR=%s"%psnr)
print("SSIM=%s"%ssim)

if origImg is not None:
    height, width = origImg.shape
    size = height * width
    # Rest of the code
else:
    print("Failed to load the image. Check the file path.")