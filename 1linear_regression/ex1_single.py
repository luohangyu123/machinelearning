import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./data.csv")
population = df["population"]
profit = df["profit"]
# print(df, type(df))
print(df)


c = 0
theta = [0, 0]
m = len(df["population"])
for j in range(m):
   c += 1.0 / (2 * m) * pow(theta[0] + theta[1] * population[j] - profit[j], 2)

print(c)
alpha = 0.01 # 学习速率
iterations = 1500 # 梯度下降的迭代轮数

# 绘制回归函数
t = []
cost = []
theta0 = []
theta1 = []

for i in range(iterations):
    t.append(i)
    temp0 = theta[0]
    temp1 = theta[1]
    for j in range(m):
        temp0 -= (alpha / m) * (theta[0] + theta[1] * population[j] - profit[j])
        temp1 -= (alpha / m) * (theta[0] + theta[1] * population[j] - profit[j]) * population[j]
    theta[0] = temp0
    theta[1] = temp1
    c = 0
    for j in range(m):
        c += 1.0 / (2 * m) * pow(theta[0] + theta[1] * population[j] - profit[j], 2)
    cost.append(c)
    theta0.append(temp0)
    theta1.append(temp1)

plt.plot(t, cost, color="red")
plt.show()

x = [5.0, 22.5]
y = [5.0 * theta[1] + theta[0], 22.5 * theta[1] + theta[0]]
plt.plot(x, y, color="red")
plt.scatter(population, profit)
plt.show()

print("当人口为时7.5，利润为=", 7.5 * theta[1]+theta[0]) # 查看预测结果




