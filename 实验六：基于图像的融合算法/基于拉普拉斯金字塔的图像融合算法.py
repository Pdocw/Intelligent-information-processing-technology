import cv2
import numpy as np
from matplotlib import pyplot as plt  
import math
from PIL import Image
img_left = cv2.imread('left.png',0)
img_right = cv2.imread('right.png',0)
img_size = (1024,1024) #统一图片大小
Series = 6 #高斯金字塔处理级数
img_left = cv2.resize(img_left, img_size)
img_right = cv2.resize(img_right, img_size)

G = img_left.copy()
guass_pyramid_left = [img_left.copy()] #保存左聚焦的高斯金字塔图片
for i in np.arange(Series):     #将左聚焦进行高斯金字塔处理，总共六级处理
    G = cv2.pyrDown(G)
    guass_pyramid_left.append(G)

G = img_right.copy()
guass_pyramid_right = [img_right.copy()] #保存右聚焦的高斯金字塔图片
for i in np.arange(Series):  #将右聚焦进行高斯金字塔处理，总共六级处理
    G = cv2.pyrDown(G)
    guass_pyramid_right.append(G)

laplace_pyramid_left = [guass_pyramid_left[Series-1]] #保存左聚焦的拉普拉斯金字塔图片             
for i in np.arange(Series-1,0,-1):    #将左聚焦进行拉普拉斯金字塔处理，总共5级处理
    GE = cv2.pyrUp(guass_pyramid_left[i])
    L = cv2.subtract(guass_pyramid_left[i-1],GE)
    laplace_pyramid_left.append(L)

laplace_pyramid_right = [guass_pyramid_right[Series-1]]#保存右聚焦的拉普拉斯金字塔图片
for i in np.arange(Series-1,0,-1):    #将右聚焦进行拉普拉斯金字塔处理，总共5级处理
    GE = cv2.pyrUp(guass_pyramid_right[i])
    L = cv2.subtract(guass_pyramid_right[i-1],GE)#两张图片相减
    laplace_pyramid_right.append(L)

LS = []
for la,lb in zip(laplace_pyramid_left,laplace_pyramid_right):
    rows,cols = la.shape
    ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))    #将两个图像的矩阵的左半部分和右半部分拼接到一起
    LS.append(ls)

#重建过程
ls_reconstruct = LS[0]   #这里LS[0]为高斯金字塔的最小图片
for i in range(1,Series):                        #第一次循环的图像为高斯金字塔的最小图片，依次通过拉普拉斯金字塔恢复到大图像
    ls_reconstruct = cv2.pyrUp(ls_reconstruct)
    ls_reconstruct = cv2.add(ls_reconstruct, LS[i]) #采用金字塔拼接方法的图像

real = np.hstack((img_left[:,:cols//2],img_right[:,cols//2:]))   #直接的拼接

def avgGradient(image):#计算平均灰度梯度
    width = image.shape[1]
    width = width - 1
    heigt = image.shape[0]
    heigt = heigt - 1
    tmp = 0.0
    for i in range(width):
        for j in range(heigt):
            dx = float(image[i, j + 1]) - float(image[i, j])
            dy = float(image[i + 1, j]) - float(image[i, j])
            ds = math.sqrt((dx * dx + dy * dy) / 2)
            tmp += ds
    imageAG = tmp / (width * heigt)
    return imageAG
def ComEntropy(img1, img2):
    width = img1.shape[0]
    hegith = img1.shape[1]
    tmp = np.zeros(img_size)
    res = 0
    for i in range(width):
        for j in range(hegith):
            val1 = img1[i][j]
            val2 = img2[i][j]
            tmp[val1][val2] = float(tmp[val1][val2] + 1)
    tmp = tmp / (width * hegith)
    for i in range(width):
        for j in range(hegith):
            if (tmp[i][j] == 0):
                res = res
            else:
                res = res - tmp[i][j] * (math.log(tmp[i][j] / math.log(2.0)))
    return res

def MI(A,B,C):
    mi = ComEntropy(A,C) + ComEntropy(B,C) - ComEntropy(B, C)
    return mi

real_mean = np.mean(real)#求均值
real_var = np.var(real)#求方差
real_std = np.std(real,ddof=1)#求标准差
real_AG = avgGradient(real)#平均灰度梯度
real_MI = MI(img_left,img_right,real)#互信息
ls_reconstruct_mean = np.mean(ls_reconstruct)#求均值
ls_reconstruct_var = np.var(ls_reconstruct)#求方差
ls_reconstruct_std = np.std(ls_reconstruct,ddof=1)#求标准差
ls_reconstruct_AG = avgGradient(ls_reconstruct)#平均灰度梯度
ls_reconstruct_MI = MI(img_left,img_right,ls_reconstruct)#互信息

print("直接拼接均值为：%f" % real_mean,"\n拉普拉斯金字塔融合均值为：%f\n" % ls_reconstruct_mean)
print("直接拼接方差为：%f" % real_var,"\n拉普拉斯金字塔融合方差为：%f\n" % ls_reconstruct_var)
print("直接拼接标准差为:%f" % real_std,"\n拉普拉斯金字塔融合标准差为:%f\n" % ls_reconstruct_std)
print("直接拼接标平均灰度梯度为:%f" % real_AG,"\n拉普拉斯金字塔融合平均灰度梯度为:%f\n" % ls_reconstruct_AG)
print("直接拼接标互信息为:%f" % real_MI,"\n拉普拉斯金字塔融合互信息为:%f\n" % ls_reconstruct_MI)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.subplot(221), plt.imshow(cv2.cvtColor(img_left,cv2.COLOR_BGR2RGB))  
plt.title("左聚焦") 
plt.subplot(222), plt.imshow(cv2.cvtColor(img_right,cv2.COLOR_BGR2RGB))  
plt.title("右聚焦")  
plt.subplot(223), plt.imshow(cv2.cvtColor(real,cv2.COLOR_BGR2RGB))  
plt.title("直接拼接")
plt.subplot(224), plt.imshow(cv2.cvtColor(ls_reconstruct,cv2.COLOR_BGR2RGB))  
plt.title("laplace_pyramid") 
plt.show()
