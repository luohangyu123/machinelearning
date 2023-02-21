import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./data.csv")
population = df["population"]
profit = df["profit"]
# print(df, type(df))

population_ = population.values.reshape(len(population), 1)
# print(population_, type(population_), profit, type(profit))
# 建立线性回归模型
regression = linear_model.LinearRegression()
# 拟合
regression.fit(population_, profit)
# 不难得到直线的斜率、截距
a, b = regression.coef_, regression.intercept_

plt.figure(figsize=(16, 8), dpi=80)
plt.scatter(population, profit)
plt.title("Scatter plot of training data")
plt.xlabel("population of city")
plt.ylabel("profit")
plt.plot(population, regression.predict(population_), color='red')
plt.show()

# 给出预测
popu = np.array([[7.5]]).reshape(-1, 1)
result = regression.predict(popu)
print("当人口为时7.5，利润为=", result) # 查看预测结果
