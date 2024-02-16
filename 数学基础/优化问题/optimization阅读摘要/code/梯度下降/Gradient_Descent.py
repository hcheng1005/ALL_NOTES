import numpy as np

'''
参考资料
[Adam-一种随机优化算法](https://zhuanlan.zhihu.com/p/133708453)
[谈谈优化算法之一（动量法、Nesterov法、自然梯度法）](https://zhuanlan.zhihu.com/p/60088231)
'''
# 梯度下降
def Gradient_Descent(X, Y, w, lr):
    Y_hat = np.dot(X, w)  # 正向传播
    loss = (0.5 * (Y - Y_hat) ** 2).mean() # 计算loss
    gradient = X.T.dot(Y_hat - Y) / len(Y)
    w = w - lr * gradient # 梯度下降
    return loss, w

'''
name: 随机梯度下降
description:
随机梯度下降顾名思义，就是在梯度下降基础上增加了随机性，即目标函数可以是随机的。
目标函数可以使用minibatch中的样本计算得到，其中minibatch为在总样本中随机采样的样本。
随着采样的不同，目标函数也是随机变化的。这可以有效的解决梯度下降的内存问题
'''
def Stochastic_Gradient_Descent(X, Y, w, lr):
    Y_hat = np.dot(X, w)  # 正向传播
    
    # 随机选取20%数据作为mini-batch进行梯度下降
    smp_num = len(Y)
    mini_batch_num = int(np.round(smp_num * 0.2))
    idx_ = np.random.randint(1, smp_num, mini_batch_num)
    
    newY = Y[idx_]
    newY_hat = Y_hat[idx_]
    newX = X[idx_]
    
    loss = (0.5 * (newY - newY_hat) ** 2).mean() # 计算loss
    gradient = newX.T.dot(newY_hat - newY) / mini_batch_num
    w = w - lr * gradient # 梯度下降
    # lr = lr * 0.99 # 调整学习率
    return loss, w, lr

'''
name: Momentum
description: 
Momentum使用动量的方式加速了学习过程，尤其是遇到损失函数高曲率情况，梯度很小但是相对稳定，噪声梯度等情况
'''
def Momentum_Gradient_Descent(X, Y, w, lr, movement=0.1, alpha=0.9):
    Y_hat = np.dot(X, w)  # 正向传播    
    loss = (0.5 * (Y - Y_hat) ** 2).mean() # 计算loss
    gradient = X.T.dot(Y_hat - Y) / len(Y)
    movement = alpha * movement - lr * gradient
    w = w + movement
    return loss, w, movement


'''
name: Nesterov Momentum(涅斯捷罗夫Momentum)
description: 
Nesterov Momentum是Momentum算法的一个变种，该方法在梯度计算之前，使用v对参数 
进行更新，可以理解为在标注动量的基础上增加了一个修正因子。
'''
def Nesterov_Momentum(X, Y, w, lr, v=0.1, alpha=0.9, beta=0.1):
    # # way 1
    # Y_hat = np.dot(X, w + gamma_*movement)  # 正向传播
    # loss = (0.5 * (Y - Y_hat) ** 2).mean() # 计算loss
    # gradient = X.T.dot(Y_hat - Y) / len(Y)
    # movement = gamma_*movement - lr * gradient
    # w = w + movement
    
    # way2
    Y_hat = np.dot(X, w)  # 正向传播
    loss = (0.5 * (Y - Y_hat) ** 2).mean() # 计算loss
    gradient = X.T.dot(Y_hat - Y) / len(Y)
    v = alpha * v - lr * gradient
    w = w + beta**2 * v - (1 + beta) * alpha * lr * gradient
    return loss, w, v


def Adagrad(X, Y, w, lr, rt, delta=1e-7):
    Y_hat = np.dot(X, w)  # 正向传播
    loss = (0.5 * (Y - Y_hat) ** 2).mean() # 计算loss
    gradient = X.T.dot(Y_hat - Y) / len(Y)
    rt = rt + gradient**2
    lr = lr / (delta + np.sqrt(rt))
    w = w -  lr * gradient
    return loss, w, rt


def RMSprop(X, Y, w, lr, rt, delta=1e-7, rho = 0.9):
    Y_hat = np.dot(X, w)  # 正向传播
    loss = (0.5 * (Y - Y_hat) ** 2).mean() # 计算loss
    gradient = X.T.dot(Y_hat - Y) / len(Y)
    rt = rho * rt + (1 - rho) * (gradient**2)
    lr = lr / np.sqrt(rt + delta)
    w = w -  lr * gradient
    return loss, w, rt


'''
name: 
description: 
算法主要是在REMSprop的基础上增加了momentum，并进行了偏差修正
'''
def Adam(X, Y, w, lr, m, v, beta1=0.9, beta2=0.999, epsilon=1e-8):
    Y_hat = np.dot(X, w)  # 正向传播
    loss = (0.5 * (Y - Y_hat) ** 2).mean() # 计算loss
    gradient = X.T.dot(Y_hat - Y) / len(Y)
    
    m = beta1 * m + (1.0 - beta1) * gradient
    v = beta2 * v + (1.0 - beta2) * (gradient**2)
    m_hat = m / (1.0 - beta1)
    v_hat = v / (1.0 - beta2)
    w = w -  lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return loss, w, m, v 