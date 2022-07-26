import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 数据加载
data = pd.read_csv("task1_data.csv")
data.head()
# x y赋值
x = data.loc[:, '面积']
y = data.loc[:, '房价']
# 数据可视化
fig1 = plt.figure()
plt.scatter(x, y)
plt.xlabel("size(x)")
plt.ylabel("price(y)")
plt.show()
# 数据预处理
x = np.array(x)
y = np.array(y)
# 维度转换
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
# 建立模型（线性回归模型）
model = LinearRegression()
# print(model)
model.fit(x, y)  # 模型训练
# 获取线性回归模型的参数a,b
a = model.coef_
b = model.intercept_
# print(a)
# print(b)
# 结果预测
# y_predict = 8905.69177214*x + 53690.91547905
# print(y_predict)
# print(y)
y_predict2 = model.predict(x)
# print(y_predict2)

# 测试样本x_test = 100计算y
x_test = np.array([[100]])
y_test_predict = model.predict(x_test)
print(y_test_predict)

# 结果数据可视化
fig2 = plt.figure()
plt.scatter(x, y)
plt.plot(x, y_predict2, label='y_predict')
plt.xlabel("size(x)")
plt.ylabel("price(y)")
plt.legend()
plt.show()

# 模型评估
MSE = mean_squared_error(y, y_predict2)
R2 = r2_score(y, y_predict2)
print(MSE)
print(R2)
