from turtle import pd
import pandas as pd
import numpy as np
from numpy.linalg import inv  # 矩阵求逆
from numpy import dot  # 矩阵点乘
#时间 价格 菜品种类 是有使用红包/满减优惠 味道 评分
data = np.array([[11.30, 13, "面食",1,"咸",6],
        [11.30, 15.4, "快餐",1,"咸/辣",7],
        [11.30, 9.98, "快餐",1,"咸/辣",5],
        [11.30, 13.5, "快餐",1,"咸/辣",7],
        [11.30, 13.5, "快餐",1,"咸/辣",8],
        [11.30, 16, "快餐",1,"咸/辣",8],
        [11.30, 13.88, "快餐",1,"咸/甜",4],
        [17.20, 13, "快餐",1,"咸/辣",7],
        [17.20, 17.99, "快餐",1,"咸/辣",9],
        [17.20, 16, "快餐",1,"咸/辣",8],
        [17.20, 12, "快餐",1,"咸/辣",7],
        [17.20, 15.99, "米饭",1,"咸",10],
        [17.20, 16.5, "披萨",1,"甜",5],
        [17.20, 19, "快餐",1,"咸/辣",8],
        [17.20, 14.99, "快餐",0,"咸",7],
        [17.20, 21.4, "粥", 1, "咸", 7], ])

import matplotlib.pyplot as plt  # 绘图用的模块
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数

def ent(data):
    prob1 = pd.value_counts(data) / len(data)
    return sum(np.log2(prob1) * prob1 * (-1))

# 定义计算信息增益的函数
def gain(data, str1, str2):
    e1 = data.groupby(str1).apply(lambda x: ent(x[str2]))
    p1 = pd.value_counts(data[str1]) / len(data[str1])
    e2 = sum(e1 * p1)
    return ent(data[str2]) - e2

pinzhong={'pinzhong':data[:,2]}
kouwei={'kouwei':data[:,4]}

def calc_ent(x):
    """
        calculate shanno ent of x
    """

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent

def calc_condition_ent(x, y):

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent

    return ent

def calc_ent_grap(x,y):
    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent

    return ent_grap
#时间 价格 菜品种类 是有使用红包/满减优惠 味道 评分
jiage=calc_ent_grap(data[:,1],data[:,5])
print ('价格:',jiage)
zhonglei=calc_ent_grap(data[:,2],data[:,5])
print ('种类:',zhonglei)
youhui=calc_ent_grap(data[:,3],data[:,5])
print ('是有使用红包/满减优惠:',youhui)
weidao=calc_ent_grap(data[:,4],data[:,5])
print ('味道:',weidao)