
import numpy as np
import random

import matplotlib.pyplot as plt 


#首先要生成一系列数据，三个参数分别是要生成数据的样本数，数据的偏差，以及数据的方差
def genData(samples, theta, bias, variance):  
    X = np.zeros(shape=samples)  #初始化X
    Y = np.zeros(shape=samples)
    Y_gt = np.zeros(shape=samples)

    for i in range(samples):
        X[i] = 0.1 * i
        Y[i] = (theta * (X[i]**3) - 2.0*theta*(X[i]**2) + 40.0*theta*(X[i]) + bias) + random.uniform(-1, 1) * variance
        Y_gt[i] = (theta * (X[i]**3) - 2.0*theta*(X[i]**2) + 40.0*theta*(X[i]) + bias)
    return X, Y, Y_gt


def gradient(x, y, order = 4, iterNum = 1000):
    # 假定实际函数为 f(x) = ax^3 + bx^2 + cx + d
    para_ = np.random.rand([order, 1])
    loss_ = 1e3
    
    x_b = np.array([len(x), 4])
    x_b[:, 0] = x**3
    x_b[:, 1] = x**2
    x_b[:, 2] = x**1
    x_b[:, 3] = 1
    
    y_b = np.array([len(y), 1])
    y_b[:,0] = y
    for i in range(iterNum):
        # 计算梯度
        gradient = np.sum(x_b.dot(para_))
        
        # 参数更新

# # 
# def gradient(x, y, order = 3, iterNum = 1000):
#     # 假定实际函数为 f(x) = ax^3 + bx^2 + cx + d
#     para_ = np.array([len(x), order])
#     para_[:, 0] = x**3
#     para_[:, 1] = x**2
#     para_[:, 2] = x**1
    
#     loss_ = 1e3
    
#     for i in range(iterNum):
        


if __name__=="__main__":
    num_, theta, bias, var_ = 100, 0.1, 10, 10
    X, Y, Y_gt = genData(num_, theta ,bias, var_)
    
    # Plot data
    ax = plt.subplot(111)   
    plt.plot(X,Y,'-o')
    plt.plot(X,Y_gt,'r')
    plt.show()
    
    
    