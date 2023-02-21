import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

path='./data.csv'

# 使用pandas加载csv
df = pd.read_csv(path)

# 建立一列为1的dataframe，起名为b作为偏置项
df["b"] = 1
print(df.info())

# 分离X和Y
X = df[["population", "b"]]
Y = df[["profit"]]
print(type(X), type(Y))

# 将dataframe转为numpy矩阵
X = np.mat(X)
Y = np.mat(Y)
# print(type(X), type(Y))

# 使用正规方程法求解模型参数
theta = (X.T*X).I*X.T*Y
# theta = np.linalg.inv(np.dot(X.T,X)).dot(np.dot(X.T,Y))
print(theta)
plt.figure(figsize=(16, 8), dpi=80)
plt.scatter(df["population"], df["profit"])
plt.title("Scatter plot of training data")
plt.xlabel("population of city")
plt.ylabel("profit")
print(type(df["population"].max()))
X_ = np.linspace(round(df["population"].min()), round(df["population"].max()))
# 将numpy矩阵转为numpy数组
theta = np.asarray(theta)
Y_ = X_*theta[0]+theta[1]
plt.plot(X_, Y_, color='red')
plt.show()

print("当人口为时7.5，利润为=", 7.5*theta[0]+theta[1]) # 查看预测结果