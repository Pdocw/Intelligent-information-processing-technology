import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
m = 2 #隶属度因子
class FCM:
    def __init__(self, data, clust_num,iter_num=10):
        self.data = data
        self.cnum = clust_num#分类数
        self.sample_num=data.shape[0]
        self.dim = data.shape[-1]  # 数据最后一维度数
        U = self.Initial_U(self.sample_num, self.cnum)
        for i in range(0, iter_num): # 迭代次数默认为10
            C = self.Cen_Iter(self.data, U, self.cnum)
            U = self.U_Iter(U, C)
            print("第%d次迭代" %(i+1) ,end="")
            print("聚类中心\n",C)
            J = self.J_calcu(self.data, U, C)  # 计算目标函数
        self.label = np.argmax(U, axis=0)  # 所有样本的分类标签
        self.Clast = C    # 最后的类中心矩阵

   
    def Initial_U(self, sample_num, cluster_n): # 初始化隶属度矩阵U
        U = np.random.rand(sample_num, cluster_n)  # sample_num为样本个数即像素点的个数, cluster_n为分类数
        row_sum = np.sum(U, axis=1)  # 按行求和 row_sum: sample_num*1
        row_sum = 1 / row_sum    # 该矩阵每个数取倒数
        U = np.multiply(U.T, row_sum)  # 确保U的每列和为1
        return U   # cluster_n*sample_num

    def Cen_Iter(self, data, U, cluster_n):# 计算类中心
        c_new = np.empty(shape=[0, self.dim])  # self.dim为样本矩阵的最后一维度
        for i in range(0, cluster_n):          # 灰度图片像素值的dim为1，如散点的dim为2，彩色图片的dim为3 加上坐标后则dim+2
            u_ij_m = U[i, :] ** m  # (sample_num,)
            sum_u = np.sum(u_ij_m)
            ux = np.dot(u_ij_m, data)  # (dim,)
            ux = np.reshape(ux, (1, self.dim))  # (1,dim)
            c_new = np.append(c_new, ux / sum_u, axis=0)   # 按列的方向添加类中心到类中心矩阵
        return c_new  # cluster_num*dim

    
    def U_Iter(self, U, c):# 隶属度矩阵迭代
        for i in range(0, self.cnum):
            for j in range(0, self.sample_num):
                sum = 0
                for k in range(0, self.cnum):
                    temp = (np.linalg.norm(self.data[j, :] - c[i, :]) /np.linalg.norm(self.data[j, :] - c[k, :])) ** (2 / (m - 1))
                    sum = temp + sum
                U[i, j] = 1 / sum
        return U
    
    def J_calcu(self, data, U, c):# 计算目标函数值
        temp1 = np.zeros(U.shape)
        for i in range(0, U.shape[0]):
            for j in range(0, U.shape[1]):
                temp1[i, j] = (np.linalg.norm(data[j, :] - c[i, :])) ** 2 * U[i, j] ** m
        J = np.sum(np.sum(temp1))
        print("目标函数值:%.2f" %J)
        return J


if __name__ == '__main__':
    data = np.loadtxt('test.txt')
    res = FCM(data,4,20)
    cluster=res.label  # 聚类结果
    center=res.Clast # 聚类中心
    for i in range(30):
        print(data[i],"第"+str(cluster[i])+"类")
