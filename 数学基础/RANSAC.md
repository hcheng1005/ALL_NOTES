# 随机采样一致性 RANSAC



- [随机采样一致性 RANSAC](#随机采样一致性-ransac)
  - [定义](#定义)
  - [RANSAC算法步骤：](#ransac算法步骤)
  - [参考代码](#参考代码)
  - [参考链接](#参考链接)

## 定义

 RANSAC是“RANdom SAmple Consensus（随机抽样一致）”的缩写。它可以从一组包含“局外点”的观测数据集中，通过迭代方式估计数学模型的参数。它是一种不确定的算法——它有一定的概率得出一个合理的结果；为了提高概率必须提高迭代次数。该算法最早由Fischler和Bolles于1981年提出。
   
RANSAC的基本假设是：
（1）数据由“局内点”组成，例如：数据的分布可以用一些模型参数来解释；
（2）“局外点”是不能适应该模型的数据；
（3）除此之外的数据属于噪声。


## RANSAC算法步骤： 

    1. 随机从数据集中随机抽出4个样本数据 (此4个样本之间不能共线)，计算出变换矩阵H，记为模型M；

    2. 计算数据集中所有数据与模型M的投影误差，若误差小于阈值，加入内点集 I ；

    3. 如果当前内点集 I 元素个数大于最优内点集 I_best , 则更新 I_best = I，同时更新迭代次数k ;

    4. 如果迭代次数大于k,则退出 ; 否则迭代次数加1，并重复上述步骤；


## 参考代码
```python
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# 数据量。
SIZE = 50
# 产生数据。np.linspace 返回一个一维数组，SIZE指定数组长度。
# 数组最小值是0，最大值是10。所有元素间隔相等。
X = np.linspace(0, 10, SIZE)
Y = 3 * X + 10

fig = plt.figure()
# 画图区域分成1行1列。选择第一块区域。
ax1 = fig.add_subplot(1,1, 1)
# 标题
ax1.set_title("RANSAC")


# 让散点图的数据更加随机并且添加一些噪声。
random_x = []
random_y = []
# 添加直线随机噪声
for i in range(SIZE):
    random_x.append(X[i] + random.uniform(-0.5, 0.5)) 
    random_y.append(Y[i] + random.uniform(-0.5, 0.5)) 
# 添加随机噪声
for i in range(SIZE):
    random_x.append(random.uniform(0,10))
    random_y.append(random.uniform(10,40))
RANDOM_X = np.array(random_x) # 散点图的横轴。
RANDOM_Y = np.array(random_y) # 散点图的纵轴。

# 画散点图。
ax1.scatter(RANDOM_X, RANDOM_Y)
# 横轴名称。
ax1.set_xlabel("x")
# 纵轴名称。
ax1.set_ylabel("y")

# 使用RANSAC算法估算模型
# 迭代最大次数，每次得到更好的估计会优化iters的数值
iters = 100000
# 数据和模型之间可接受的差值
sigma = 0.25
# 最好模型的参数估计和内点数目
best_a = 0
best_b = 0
pretotal = 0
# 希望的得到正确模型的概率
P = 0.99
for i in range(iters):
    # 随机在数据中红选出两个点去求解模型
    sample_index = random.sample(range(SIZE * 2),2)
    x_1 = RANDOM_X[sample_index[0]]
    x_2 = RANDOM_X[sample_index[1]]
    y_1 = RANDOM_Y[sample_index[0]]
    y_2 = RANDOM_Y[sample_index[1]]

    # y = ax + b 求解出a，b
    a = (y_2 - y_1) / (x_2 - x_1)
    b = y_1 - a * x_1

    # 算出内点数目
    total_inlier = 0
    for index in range(SIZE * 2):
        y_estimate = a * RANDOM_X[index] + b
        if abs(y_estimate - RANDOM_Y[index]) < sigma:
            total_inlier = total_inlier + 1

    # 判断当前的模型是否比之前估算的模型好
    if total_inlier > pretotal:
        iters = math.log(1 - P) / math.log(1 - pow(total_inlier / (SIZE * 2), 2))
        pretotal = total_inlier
        best_a = a
        best_b = b

    # 判断是否当前模型已经符合超过一半的点
    if total_inlier > SIZE:
        break

# 用我们得到的最佳估计画图
Y = best_a * RANDOM_X + best_b

# 直线图
ax1.plot(RANDOM_X, Y)
text = "best_a = " + str(best_a) + "\nbest_b = " + str(best_b)
plt.text(5,10, text,
         fontdict={'size': 8, 'color': 'r'})
plt.show()
```

## 参考链接
- [随机采样一致性 RANSAC](https://www.cnblogs.com/zl1991/p/7509363.html)