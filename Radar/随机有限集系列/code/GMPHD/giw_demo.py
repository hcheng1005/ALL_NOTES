import numpy as np 
import matplotlib.pyplot as plt
from giw_filer import GIW_Filter


def poissonSample(lamb):
	"Sample from a Poisson distribution. Algorithm is due to Donald Knuth - see wikipedia/Poisson_distribution"
	l = np.exp(-lamb)
	k = 0
	p = 1
	while True:
		k += 1
		p *= np.random.rand()
		if p <= l:
			break
	return k - 1


# 构造目标[x,y,vx,vy,l,w,theta]
gt_obj_list = np.array([[1, 10, 7, 7]]).T
cov = [[2, 0.5], 
       [0.5, 1]]
gt_num = 1

np.random.seed(2024)   

# 生成真值和量测
T_ = 0.1
Ffun = np.array([[1, 0, T_, 0],
                [0, 1, 0, T_],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

Hfun = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])

R_ = np.array([[1,0],
               [0,1]]) * 0.2

clutterintensitytot = 10
span = (0, 100)
slopespan = (-2, 3)  # currently only used for clutter generation / inference
obsntype = "chirp"

gt_data = gt_obj_list
meas_data = Hfun @ gt_data

# 定义gmphd滤波器
my_GMM = GIW_Filter()

plt.figure

sim_step = 100
for i in range(sim_step):
    # 真值
    gt_obj_list = Ffun @ gt_obj_list
    
    # 椭圆采样
    smp_data = np.random.multivariate_normal(gt_obj_list[:2].reshape([-1]), cov, 20) # 根据均值和协方差进行点云采样
    smp_data = smp_data.T
 
    # gmphd filter
    my_GMM.proc(smp_data)
    comps = my_GMM.getComponents()
    
    # 可视化
    plt.clf()
    plt.axis([-20, 100, -20, 100])
    plt.plot(smp_data[0, :], smp_data[1, :], 'b*')

    for comp in comps:
        # print(comp.X)
        plt.plot(comp.X[0], comp.X[1], 'ro')

    plt.ion()
    plt.pause(0.2)
    