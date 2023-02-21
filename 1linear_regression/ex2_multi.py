import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

path='./data_2.csv'

# 使用pandas加载csv
df = pd.read_csv(path)
size = df["size"]
num = df["num"]
price = df["price"]

# 对数据集增加偏置，并去除掉不需要的列
df = df[["size", "num", "price"]]

# 建立一列为1的dataframe，起名为b作为偏置项
df["b"] = 1

# # 异常数据处理  (只要有特征为空，就进行删除操作)
# df = df.replace("?", np.nan).dropna()

print(df.info())
# 将字符串转为浮点
# df.Global_active_power = df.Global_active_power.astype(np.float64)

# 分离X和Y
# 获取"Global_active_power","Global_reactive_power","b"为X
X = df[["size", "num", "b"]]
Y = df[["price"]]
print(X, type(X), Y, type(Y))

# 将dataframe转为numpy矩阵
X = np.mat(X)
Y = np.mat(Y)
# print(type(X), type(Y))

# 使用正规方程法求解模型参数
# theta = (X.T*X).I*X.T*Y
theta = np.linalg.inv(np.dot(X.T, X)).dot(np.dot(X.T, Y))
print(theta)

# 预测
print("2100,3房价为", 2100*theta[0] + 3*theta[1] + theta[2])