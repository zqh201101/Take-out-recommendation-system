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

list1=(list(data[:,1]))
varieties=(list(data[:,2]))
print (varieties)
taste=(list(data[:,4]))
print (taste)
price=[]
for n in range(0,16):
       price.append(float(list1[n]))
print(price)
print("最高消费记录：",max(price))
print("最低消费记录：",min(price))
print("平均消费:",(sum(price)-max(price)-min(price))/15)
ave=(sum(price)-max(price)-min(price))/15

#假设今天商家推出了以下三样菜品，要素分别为：价格、菜品种类、口味  倾向消费者推荐最适合的一样产品
example1=[14,"快餐","咸/辣"]
example2=[13,"粥","咸"]
example3=[40,"粥","咸/甜"]

def valuer_ecommend(list):
    v=abs(list[0]-ave)*0.5358*(-1)+varieties.count(example1[1])/17*0.23+taste.count(example1[2])/17*0.2374
    return v
v1=valuer_ecommend(example1)
v2=valuer_ecommend(example2)
v3=valuer_ecommend(example3)

print("ex1:",v1,"ex2:",v2,"ex3:",v3)


