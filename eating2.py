import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model

#输入数据集【时间 价格 菜品种类 有无红包或满减优惠1/0 口味 评分】
#菜品种类【1：面食，2：快餐，3：米饭，4：披萨，0：粥】
#口味【0:咸，1：甜，2：咸辣，3：咸甜】
x = np.array([[11.30, 13,1,1,0,6],
        [11.30, 15.4,2,1,2,7],
        [11.30, 9.98,2,1,2,5],
        [11.30, 13.5,2,1,2,7],
        [11.30, 13.5,2,1,2,8],
        [11.30, 16,2,1,2,8],
        [11.30, 13.88,2,1,3,4],
        [17.20, 13,2,1,2,7],
        [17.20, 17.99,2,1,2,9],
        [17.20, 16,2,1,2,8],
        [17.20, 12,2,1,2,7],
        [17.20, 15.99,3,1,0,10],
        [17.20, 16.5,4,1,1,5],
        [17.20, 19,2,1,2,8],
        [17.20, 14.99,2,0,0,7],
        [17.20, 21.4,0,1,0,7]])
x_trans = []
x_trans1=[]
for i in range(len(x)):
    x_trans.append({'x1':str(x[i][2])})
    
vec = DictVectorizer()

for i in range(len(x)):
    x_trans1.append({'x2':str(x[i][4])})
    
vec1 = DictVectorizer()

#数据处理
dummyX = vec.fit_transform(x_trans).toarray()#菜品种类
dummyX1 = vec1.fit_transform(x_trans1).toarray()#口味
x = np.concatenate((x[:,1].reshape(len(x),1),dummyX[:,:],x[:,3].reshape(len(x),1),dummyX1[:,:],x[:,-1].reshape(len(x),1)),axis=1)
x = x.astype(float)
X = x[:,:-1]
Y = x[:,-1]
print(x)

# 训练数据
regr = linear_model.LinearRegression()
regr.fit(X,Y)
print('coefficients(w1,w2...):',regr.coef_)
print('intercept(w0):',regr.intercept_)

# 预测
x_test = np.matrix([[14,0,0,1,0,0,1,0,0,1,0],
[13,0,0,0,0,0,1,0,0,0,0],
[40,0,0,0,0,0,1,0,0,0,1]])
y_test = regr.predict(x_test)
print(y_test)




