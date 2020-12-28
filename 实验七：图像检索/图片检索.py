from  PIL import Image
import os
from matplotlib import pyplot as plt
import numpy as np
import math
def Intersect(lh, rh):#相交系数 越大越好
    return sum(min(l,r) for l,r in zip(lh,rh))

def Histogram_intersection_method(lh, rh):#直方图相交法 越大越好
    return sum(min(l,r) for l,r in zip(lh,rh))/sum(l for l, r in zip(lh, rh))

def Relativity(lh, rh):#相关度 越大越好
    ans = sum((l - sum(ll for ll, rr in zip(lh,rh))/len(lh))*(r - sum(rr for ll, rr in zip(lh,rh))/len(rh)) for l,r in zip(lh,rh))
    ans1 = 0.0
    for l,r in zip(lh,rh):
        temp = (l - sum(ll for ll, rr in zip(lh,rh))/len(lh))*(r - sum(rr for ll, rr in zip(lh,rh))/len(rh))
        ans1 += temp*temp
    return ans/math.sqrt(ans1)

def Ba_distance(lh, rh):#巴氏距离 越小越好
    return  math.sqrt(1 - sum(math.sqrt(l*r) for l,r in zip(lh,rh))/(sum(l for l, r in zip(lh, rh))*sum(r for l, r in zip(lh, rh))))

def K_square(lh, rh):#卡方系数 越小越好
    return sum(0 if l + r == 0 else (l-r)*(l-r)/(l+r) for l, r in zip(lh, rh))

def Distance(lh, rh):#欧式距离相似度 越大越好
    return sum(1 - (0 if l == r else float(abs(l - r))/max(l, r)) for l, r in zip(lh, rh))/len(lh)

def Choose_similarity_calculation_method(lf, rf):#选择相似性计算方法
    l_resize, r_resize = Image.open(lf).resize((256,256)).convert('RGB'), Image.open(rf).resize((256,256)).convert('RGB')
    #把所有的图片都统一到特别的规格,256x256的分辨率
    return sum(Distance(l.histogram(), r.histogram()) for l, r in zip(split_image(l_resize), split_image(r_resize))) / 16.0


def split_image(img, part_size = (64, 64)):
    w, h = img.size
    pw, ph = part_size
    return [img.crop((i, j, i+pw, j+ph)).copy() for i in range(0, w, pw) for j in range(0, h, ph)]

def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)

if __name__ == '__main__':
    list_name = []
    path='D:\Desktop\Study\大三上\智能信息处理\实验七：图像检索\corel1000\\'   #文件夹路径
    listdir(path,list_name)

    img1_path = 'D:\Desktop\Study\大三上\智能信息处理\实验七：图像检索\corel1000\\400.jpg'
    img2_path = 'D:\Desktop\Study\大三上\智能信息处理\实验七：图像检索\corel1000\\666.jpg'
    similary = []
    index = []
    for i in range(1000):
        print(i)  
        similary.append(Choose_similarity_calculation_method(img1_path, list_name[i]))
        index.append(i)
    itemZip = dict(zip(similary,index))
    similary_sort = sorted(similary,reverse = True)
    for i in range(10):
        plt.figure(1)
        plt.subplot(2,5,i+1)
        plt.imshow(Image.open(list_name[itemZip[similary_sort[i]]]))
    plt.show()
    similary = []
    index = []
    for i in range(1000): 
        print(1000+i) 
        similary.append(Choose_similarity_calculation_method(img2_path, list_name[i]))
        index.append(i)
    itemZip = dict(zip(similary,index))
    similary_sort = sorted(similary,reverse = True)
    for i in range(10):
        plt.figure(2)
        plt.subplot(2,5,i+1)
        plt.imshow(Image.open(list_name[itemZip[similary_sort[i]]]))
    plt.show()
    #print(similary_sort)
    #print(itemZip[])
    #print(similary)