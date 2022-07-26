#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 数据加载
import pandas as pd
import numpy as np

data = pd.read_csv("task2_data.csv")
data.head(10)

# In[7]:


from matplotlib import pyplot as plt

fig = plt.figure(figsize=(20, 5))
fig1 = plt.subplot(131)
plt.scatter(data.loc[:, '面积'], data.loc[:, '价格'])
plt.title('Price VS Size')

fig2 = plt.subplot(132)
plt.scatter(data.loc[:, '人均收入'], data.loc[:, '价格'])
plt.title('Price VS Income')

fig3 = plt.subplot(133)
plt.scatter(data.loc[:, '平均房龄'], data.loc[:, '价格'])
plt.title('Price VS House_age')
plt.show()

# In[11]:


# x y赋值
X = data.loc[:, '面积']
y = data.loc[:, '价格']
print(X, y)

# In[19]:


# 数据预处理
X = np.array(X)
y = np.array(y)
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
print(X.shape, y.shape)

# In[20]:


# 模型建立与训练
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

# In[21]:


# 模型预测
y_predict = model.predict(X)
print(y_predict)

# In[23]:


# 模型评估
from sklearn.metrics import mean_squared_error, r2_score

MSE = mean_squared_error(y, y_predict)
R2 = r2_score(y, y_predict)
print(MSE)
print(R2)

# In[26]:


fig2 = plt.figure(figsize=(8, 5))
plt.scatter(data.loc[:, '面积'], data.loc[:, '价格'])
plt.plot(X, y_predict, 'r')
plt.title('Price VS Size')

# In[31]:


# X y再次赋值
X = data.drop(['价格'], axis=1)
y = data.loc[:, '价格']
X.head()

# In[32]:


print(X.shape)

# In[33]:


# 建立多因子回归模型并训练
model_multi = LinearRegression()
model_multi.fit(X, y)

# In[34]:


# 多因子模型的预测
y_predict_multi = model_multi.predict(X)
print(y_predict_multi)

# In[36]:


# 模型评估
MSE_multi = mean_squared_error(y, y_predict_multi)
R2 = r2_score(y, y_predict_multi)
print(MSE_multi)
print(R2)

# In[38]:


# 可视化预测结果
fig3 = plt.figure(figsize=(8, 5))
plt.scatter(y, y_predict_multi)
plt.xlabel('real price')
plt.ylabel('predict price')
plt.show()

# In[39]:


fig4 = plt.figure(figsize=(8, 5))
plt.scatter(y, y_predict)
plt.xlabel('real price')
plt.ylabel('predict price')
plt.show()

# In[48]:


# 预测面积=160 人均收入=70000，平均房龄=5年
X_test = np.array([[160, 70000, 5]])
y_test_predict = model_multi.predict(X_test)
print(y_test_predict)

# In[ ]:
