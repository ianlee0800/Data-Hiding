import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# 繪出影像長條圖
def draw_histogram(imgName, img):
    height, width = img.shape
    q = 256
    count = [0]*q
    for y in range(height):
        for x in range(width):
            bgr = int(img[y,x])
            for i in range(q):
                if i == bgr:
                    count[i] += 1
                    i = 256
    plt.figure()
    plt.bar(range(1,257), count)
    plt.savefig("./histogram/%s_histogram.png"%(imgName))
# 計算峰值訊噪比PSNR
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
    mse = mse / size
    psnr = 10*math.log10(max**2/mse)
    psnr = round(psnr, 4)
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
    ssim = round(ssim, 2)
    return ssim
#計算色階
def count_level(img):
    height, width= img.shape
    count = [0]*LEVELS
    for y in range(height):
        for x in range(width):
            count[img[y,x]] += 1
    return count

#讀取影像
imgName = input("image name: ")
fileType = "png"
origImg = cv2.imread("./image/%s.%s"%(imgName, fileType), cv2.IMREAD_GRAYSCALE)
#長、寬、大小
height, width = origImg.shape
size = height * width
LEVELS = 256

#輸出元影像長條圖
draw_histogram("%s_orig"%imgName, origImg)

outImg = origImg.copy()
count = count_level(outImg)#計算每色階數量
#找峰值
peak = 0
for i in range(1, LEVELS):
    if count[i] > count[peak]:
        peak = i
#選擇左移或右移
if peak == 255:
    shift = -1
else:
    shift = 1
map = np.zeros((height, width))#記錄像素格為255的位置
#隱藏嵌入值
payload = count[peak]
hideArray = [np.random.choice([0,1], p=[0.5,0.5]) for n in range(payload)]
i = 0
for y in range(height):
    for x in range(width):
        if shift == 1:
            if outImg[y,x] == peak:
                if hideArray[i] == 1:
                    outImg[y,x] += shift
                i += 1
            elif outImg[y,x] > peak and outImg[y,x] != 255:
                outImg[y,x] += shift
        if shift == -1:
            if outImg[y,x] == peak:
                if hideArray[i] == 1:
                    outImg[y,x] += shift
                i += 1
            elif outImg[y,x] < peak and outImg[y,x] != 0:
                outImg[y,x] += shift
#計算相關值與輸出嵌入後的長條圖和影像
psnr = calculate_psnr(origImg, outImg)
ssim = calculate_ssim(origImg, outImg)
bpp = payload/size
print("peak=%s"%(peak))
print("payload=%s"%payload)
print("bpp=%.2f"%(bpp))
print("PSNR=%s"%psnr)
print("SSIM=%s"%ssim)
draw_histogram("%s_markedImg"%(imgName), outImg)
cv2.imwrite("./outcome/%s_markedImg.%s"%(imgName, fileType), outImg)