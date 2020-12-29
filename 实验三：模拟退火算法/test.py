import math
from random import random
import matplotlib.pyplot as plt
# 可以看到在温度较高时解和目标函数的变化幅度比较大，但随着温度的降低状态逐渐稳定，波动幅度减小，并且温度越低变化的频率越高。
def func(x, y):
    return math.sin(2 * math.pi * x)*math.sin(2 * math.pi * x)*math.sin(2 * math.pi * x)*math.sin(2 * math.pi * y)/((x*x*x)*(x+y))
class SimulateAnnealing:
    def __init__(self, func, iter=100, T0=100, Tf=1e-7, alpha=0.99):
        self.func = func
        self.iter = iter    #iter内循环迭代次数，目的是基于不同初值进行多次退火，避免陷入局部最优
        self.alpha = alpha  #alpha：降温系数
        self.T0 = T0    #T0：初始温度
        self.Tf = Tf #Tf：温度终值，一般可以取较小的值，目的是降低收敛速度，提高精确度
        self.T = T0
        self.x = [random() * 10 for i in range(iter)]
        self.y = [random() * 10 for i in range(iter)]
        self.history = {'f': [], 'T': []}

    def generate_new(self, x, y):   #random() - random()可以得到[-1, 1]间的随机数， 并且降低了取到0的概率
        while True:
            x_new = x + self.T * (random() - random())
            y_new = y + self.T * (random() - random())
            if (0 <= x_new <= 10) and (0 <= y_new <= 10) and x_new*x_new - y_new + 1 <= 0 and 1 - x_new*x_new + (y_new-4)*(y_new-4) <= 0:
                break
        return x_new, y_new
    def Metrospolis(self, f, f_new):
        p = math.exp((f - f_new) / self.T)
        if f_new <= f or random() < p:
            return 1
        else:
            return 0
    def get_optimal(self):  #获取最优目标函数值
        f_list = []
        for i in range(self.iter):
            f = self.func(self.x[i], self.y[i])
            f_list.append(f)
        f_best = min(f_list)
        idx = f_list.index(f_best)
        return f_best, idx
    def solve(self):
        while self.T > self.Tf:
            for i in range(self.iter):
                f = self.func(self.x[i], self.y[i])
                x_new, y_new = self.generate_new(self.x[i], self.y[i])
                f_new = self.func(x_new, y_new)
                if self.Metrospolis(f, f_new):
                    self.x[i] = x_new
                    self.y[i] = y_new
            ft, _ = self.get_optimal()
            self.history['f'].append(ft)
            self.history['T'].append(self.T)
            self.T = self.T * self.alpha
            print(self.T)
        f_best, idx = self.get_optimal()
        print(f"Function={f_best}, x={self.x[idx]}, y={self.y[idx]}")

if __name__ == '__main__':
    SA = SimulateAnnealing(func)
    SA.solve()
    xlim=[]
    ylim=[]
    plt.plot(SA.history['T'], SA.history['f'])
    plt.title('Simulate Annealing')
    plt.xlabel('Temperature')
    plt.ylabel('Function value')
    if xlim:
        plt.xlim()
    if ylim:
        plt.ylim()
    plt.gca().invert_xaxis()#将x轴反向 便于观察退火过程
    plt.show()