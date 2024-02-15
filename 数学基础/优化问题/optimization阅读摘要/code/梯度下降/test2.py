# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np
import random
from sklearn import linear_model

from scipy.optimize import leastsq

from Gradient_Descent import Gradient_Descent, Stochastic_Gradient_Descent, Momentum_Gradient_Descent, Nesterov_Momentum
from Gradient_Descent import Adagrad, RMSprop, Adam

#首先要生成一系列数据，三个参数分别是要生成数据的样本数，数据的偏差，以及数据的方差
def getdata(samples, theta, bias, variance):  
    X = np.zeros(shape=samples)  #初始化X
    Y = np.zeros(shape=samples)

    for i in range(samples):
        X[i] = 0.1 * i
        Y[i] = (theta * (X[i]**3) + 2.0*theta*(X[i]**2) + bias) + random.uniform(0, 1) * variance
    return X, Y

def normalization_(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma, sigma

def minmax_scaling(data):
    max_, min_ = np.max(data), np.min(data)
    return (data - min_) / (max_ - min_), (max_ - min_)


# 梯度下降算法来解决最优化问题，求损失函数的最小值
def gradient(X, Y, order, lr, m, iter_numbers):
    para = np.ones([order,1])
    # para = np.array([1.5, 26.2]).reshape([2,1])
    Y = Y.reshape([m, 1])
    # Y, scale = minmax_scaling(Y)  # Min-Max Normalization
    
    # scale = 1        
    X_b = np.ones([num_, order]) 
    X_b[:, 0] = X**3
    X_b[:, 1] = X**2
    
    # X_b[:,0], scale1 = minmax_scaling(X_b[:,0])  # Min-Max Normalization
    # X_b[:,1], scale2 = minmax_scaling(X_b[:,1])  # Min-Max Normalization
    
    # X_b[:,0], scale1 = normalization_(X_b[:,0])  # Z-Score Normalization
    # X_b[:,1], scale2 = normalization_(X_b[:,1])  # Z-Score Normalization
    
    X_trans=X_b.transpose()  # 转化为列向量
    last_loss = 1e5
    count = 1
    
    # Momentum_Gradient_Descent
    movement = 0.1
    
    # # adam
    # movement = 0.0
    # vel = 0
    
    sum_gt = 0.0
    rt = 0.0
    while 1:                  
        # loss, para = Gradient_Descent(X_b, Y, para, lr)
        # loss, para, lr = Stochastic_Gradient_Descent(X_b, Y, para, lr)
        loss, para, movement = Momentum_Gradient_Descent(X_b, Y, para, lr, movement, alpha=0.99)
        # loss, para, movement = Nesterov_Momentum(X_b, Y, para, lr, movement, alpha=0.99, beta=0.1)
        # loss, para, rt = Adagrad(X_b, Y, para, lr, rt)
        # loss, para, rt = RMSprop(X_b, Y, para, lr, rt)
        # loss, para, movement, vel = adam(X_b, Y, para, lr, movement, vel, beta1=0.9, beta2=0.5, epsilon=1e-8)
        
        print("iteration:%d / Cost:%f"%(count, loss)) 
        
        # 结束条件
        if abs(last_loss - loss) < 1e-8:
            break
        if count > iter_numbers:
            break
        if np.isnan(loss):
            break
        
        last_loss = loss
        count = count + 1
        
    # print(para, scale)   
    # para[0] = para[0] / scale1
    # para[1] = para[1] / scale2
    return para

# 定义目标函数和残差函数
def Fun(p,x):
    a1, a2, a3 = p
    return a1*(x**3)+a2*(x**2)+a3

def error (p,x,y):                   
    return Fun(p,x)-y 

if __name__=="__main__":
    num_, theta, bias, var_ = 100, 0.15, 10, 2
    X, Y = getdata(num_, theta ,bias, var_)

    # Plot data
    ax = plt.subplot(111)   
    ax.scatter(X, Y, c='r',marker='1') 
    
    # 梯度下降法
    lr_, iter_num_ = 0.00001, 200e4
    para = gradient(X, Y, 3, lr_, num_, iter_num_)
    print(para)
    # 打印GD得到的参数对应的函数曲线
    X_GD = X
    Y_GD = np.zeros(shape=num_)
    for i in range(num_):
        X_GD[i] = X[i]
        Y_GD[i] = para[0] * (X_GD[i]**3) + para[1] * (X_GD[i]**2) + para[2] 
    ax.plot(X_GD, Y_GD, c='b', lw=1) 
  
    # 调用scipy库函数
    p0 = [0.1, 0.1, 0.1] # 拟合的初始参数设置
    para = leastsq(error, p0, args=(X, Y))  # 进行拟合
    print(para)
    Y2 = Fun(para[0],X) 
    ax.plot(X, Y2, c='g') 
    
    # 矩阵解法
    X_b = np.ones([num_, 3])
    X_b[:,0] = X**3 # 此处体现的就是目标函数是 y=a*x^3+b
    X_b[:,1] = X**2 # 此处体现的就是目标函数是 y=a*x^3+b
    w_ = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
    print(w_)
    
    # plt.show()