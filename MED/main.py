import cv2
import numpy as np
import matplotlib as plt
import math

#計算PSNR
def calculate_psnr(img1, img2):
    height, width= img1.shape
    size = height*width
    max = 255
    mse = 0
    for y in range(height):
        for x in range(width):
            bgr1 = int(img1[y,x])
            bgr2 = int(img2[y,x])
            diff = bgr1 - bgr2
            mse += diff**2
    mse = mse / size
    if mse==0:
        psnr=float("inf")
    else:    
        psnr = 10*math.log10(max**2/mse)
        psnr = round(psnr, 2)
    return psnr
#計算SSIM
def calculate_ssim(img1, img2):
    height, width= img1.shape
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

# MED预测&隱藏
def med_predictor(img,secret):
    row, col = img.shape
    res = np.zeros_like(img)#生成和原圖大小一样的全0结果圖res
    med_image = np.zeros_like(img)#生成和原圖大小一样的全0结果圖res
    res[0,:] = img[0,:] 
    res[:,0] = img[:,0]
    med_image[0,:] = img[0,:] 
    med_image[:,0] = img[:,0]
    k=0
    
    for i in range(1,row):
        for j in range(1,col):
            a = img[i, j-1] 
            b = img[i-1, j]
            c = img[i-1, j-1]
            res[i,j] = max(a,b) if c<=min(a,b) else min(a,b) if c>=max(a,b) else a+b-c
            error=img[i,j]-res[i,j]
            if k>=len(secret):
                print("secret hide finish")
                med_image[i,j]=img[i,j]
            else:
                if error<-1:
                    error_plam= error-1
                elif error==-1:
                    error_plam=error-secret[k]
                    k+=1  
                elif error==0:
                    error_plam=error+secret[k]   
                    k+=1
                else :
                    error_plam=error+1
                med_image[i,j]=res[i,j]+error_plam     
        
    return med_image,k
#HONG 等人預測方法 ("Reversible data hiding for high quality images using modification of prediction errors)
def hong_med_predictor(img,secret):
    row, col = img.shape
    res = np.zeros_like(img)#生成和原圖大小一样的全0结果圖res
    med_image = np.zeros_like(img)#生成和原圖大小一样的全0结果圖res
    res[0,:] = img[0,:] 
    res[:,0] = img[:,0]
    med_image[0,:] = img[0,:] 
    med_image[:,0] = img[:,0]
    k=0
    
    for i in range(1,row):
        for j in range(1,col):
            a = med_image[i, j-1] 
            b = med_image[i-1, j]
            c = med_image[i-1, j-1]
            res[i,j] = max(a,b) if c<=min(a,b) else min(a,b) if c>=max(a,b) else a+b-c
            error=img[i,j]-res[i,j]
            if k>=len(secret):
                print("secret hide finish")
                med_image[i,j]=img[i,j]
            else:
                if error<-1:
                    error_plam= error-1
                elif error==-1:
                    error_plam=error-secret[k]
                    k+=1  
                elif error==0:
                    error_plam=error+secret[k]   
                    k+=1
                else :
                    error_plam=error+1
                med_image[i,j]=res[i,j]+error_plam     
        
    return med_image,k




def med_predictor_improve(img,secret):
    row, col = img.shape
    res = np.zeros_like(img)#生成和原圖大小一样的全0结果圖res
    med_image = np.zeros_like(img)#生成和原圖大小一样的全0结果圖res
    error_img=np.zeros_like(img)
    res[0,:] = img[0,:] 
    res[:,0] = img[:,0]
    med_image[0,:] = img[0,:] 
    med_image[:,0] = img[:,0]
    k=0
    m1=0 #誤差值為1的數量
    m=0 #誤差值為-1的數量
    n0=0 #位元值為0的數量
    n1=0 #位元值為1的數量
    for i in range(1,row):
        for j in range(1,col):
            a = img[i, j-1] 
            b = img[i-1, j]
            c = img[i-1, j-1]
            res[i,j] = max(a,b) if c<=min(a,b) else min(a,b) if c>=max(a,b) else a+b-c
            error_img[i,j]=img[i,j]-res[i,j]
            if error_img[i,j]== 1:
                m1+=1
            elif error_img[i,j]== -1:
                m+=1
            else:
                pass             

    for p in range(len(secret)):
        if secret[p]==0:
            n0+=1
        else:
            n1+=1    
    if m>=m1 and n0>=n1:
        print("誤差值-1個數>=誤差值1個數，隱藏位元值0較多")                
        for i in range(1,row):
            for j in range(1,col):
                
                error=error_img[i,j]
                if k>=len(secret):
                    print("secret hide finish")
                    med_image[i,j]=img[i,j]
                else:
                    if error<-1:
                        error_plam= error-1
                    elif error==-1:
                        error_plam=error-secret[k]
                        k+=1  
                    elif error==0:
                        error_plam=error+secret[k]   
                        k+=1
                    else :
                        error_plam=error+1
                    med_image[i,j]=res[i,j]+error_plam     
    elif m<m1 and n0>=n1:
        print("誤差值-1個數<誤差值1個數，隱藏位元值0較多")
        for i in range(1,row):
            for j in range(1,col):
                
                error=error_img[i,j]
                if k>=len(secret):
                    print("secret hide finish")
                    med_image[i,j]=img[i,j]
                else:
                    if error<0:
                        error_plam= error-1
                    elif error==0:
                        error_plam=error-secret[k]
                        k+=1  
                    elif error==1:
                        error_plam=error+secret[k]   
                        k+=1
                    else :
                        error_plam=error+1
                    med_image[i,j]=res[i,j]+error_plam
    elif m>=m1 and n0<n1:
        print("誤差值-1個數>=誤差值1個數，隱藏位元值1較多")
        for i in range(1,row):
            for j in range(1,col):
                
                error=error_img[i,j]
                if k>=len(secret):
                    print("secret hide finish")
                    med_image[i,j]=img[i,j]
                else:
                    if error<-1:
                        error_plam= error-1
                    elif error==-1:
                        error_plam=error+secret[k]-1
                        k+=1  
                    elif error==0:
                        error_plam=error-secret[k]+1   
                        k+=1
                    else :
                        error_plam=error+1
                    med_image[i,j]=res[i,j]+error_plam
    elif m<m1 and n0<n1:
        print("誤差值-1個數<誤差值1個數，隱藏位元值1較多")
        for i in range(1,row):
            for j in range(1,col):
                
                error=error_img[i,j]
                if k>=len(secret):
                    print("secret hide finish")
                    med_image[i,j]=img[i,j]
                else:
                    if error<0:
                        error_plam= error-1
                    elif error==0:
                        error_plam=error+secret[k]-1
                        k+=1  
                    elif error==1:
                        error_plam=error-secret[k]+1   
                        k+=1
                    else :
                        error_plam=error+1
                    med_image[i,j]=res[i,j]+error_plam                                

    return med_image,k

#MED decode
def med_decode(img):
    row, col = img.shape
    res = np.zeros_like(img)#生成和原圖大小一样的全0结果圖res
    recover_image = np.zeros_like(img)#生成和原圖大小一样的全0结果圖res
    secret=[]
    res[0,:] = img[0,:] 
    res[:,0] = img[:,0]
    recover_image[0,:] = img[0,:] 
    recover_image[:,0] = img[:,0]
    for i in range(1,row):
        for j in range(1,col):
            a = img[i, j-1] 
            b = img[i-1, j]
            c = img[i-1, j-1]
            res[i,j] = max(a,b) if c<=min(a,b) else min(a,b) if c>=max(a,b) else a+b-c
            error_plam=img[i,j]-res[i,j]
            if error_plam==-1 or 0:
                secret.append(0)
            elif error_plam==-2 or 1:
                secret.append(1)
            error_dauble_plam= error_plam+1 if error_plam<-1 else error_plam if error_plam==-1 or 0 else error_plam-1
            recover_image[i,j]=res[i,j]+error_dauble_plam
    return recover_image,secret
#main
test_image=np.array([[144,146,143,143],
                     [144,146,142,145],
                     [144,146,141,147],
                     [141,148,147,145]])
hide_data=[1,0,1]
imgName = input("image name: ")
fileType = "png"
origImg = cv2.imread("./original_image/%s.%s" % (imgName, fileType),cv2.IMREAD_GRAYSCALE)
cv2.imwrite('./original_gray/%s_gray.png' % imgName, origImg)
origImg=origImg.astype(np.int16)
#判別input是否讀取成功
if origImg is None:
    print("Error reading originalimage file")
    exit()
else:
    print("origImg reading sucessful")
              
#長、寬、大小
height, width= origImg.shape
print("Original Image Size:%s x %s"%(height,width) )
payload=origImg.size
secret=[np.random.choice([0,1], p=[0.5,0.5]) for n in range(payload)]
med_image,k=med_predictor_improve(origImg,secret)
hong_med,m=hong_med_predictor(origImg,secret)
psnr=calculate_psnr(origImg,med_image)
ssim=calculate_ssim(origImg,med_image)
bpp=k/origImg.size
hong_psnr=calculate_psnr(origImg,hong_med)
hong_ssim=calculate_ssim(origImg,hong_med)
cv2.imwrite('./med_embed/%s_embed.png' % imgName, med_image)
print("payload:",k)
print("psnr:",psnr)
print("ssim",ssim)
print("bpp:",bpp)
print("hong payload:",m)
print("hong psnr:",hong_psnr)
print("hong ssim:",hong_ssim)

recover_image,data=med_decode(med_image)
recover_psnr=calculate_psnr(origImg,recover_image)
print("recover psnr:",recover_psnr)
cv2.imwrite('./med_recover/%s_recover.png' % imgName, recover_image)

