import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# 这里我们使用完全向量化的实现方法
def gradient_descent(X, theta, y, alpha, iterations):
    m = X.shape[0]
    c = []  # 存储计算损失值
    for i in range(iterations):
        theta -= (alpha / m) * X.T.dot(X.dot(theta) - y)
        c.append(cost(X, theta, y))
    return theta, c


def cost(X, theta, y):
    m = X.shape[0]
    temp = X.dot(theta) - y
    return temp.T.dot(temp) / (2 * m)


if __name__ == '__main__':
    path = './data_2.csv'
    # 使用pandas加载csv
    df = pd.read_csv(path)
    # 特征归一化
    x1 = np.array(df["size"]).reshape(-1, 1)
    x2 = np.array(df["num"]).reshape(-1, 1)
    y = np.array(df["price"]).reshape(-1, 1)

    data = np.concatenate((x1, x2, y), axis=1)  # 放在一个ndarray中便于归一化处理
    mean = np.mean(data, axis=0)  # 计算每一列的均值
    ptp = np.ptp(data, axis=0)  # 计算每一列的最大最小差值
    normal_data = (data - mean) / ptp  # 归一化

    # print(normal_data)
    X = np.insert(normal_data[..., :2], 0, 1, axis=1)  # 添加x0=1
    y = normal_data[..., -1]

    # print(X, y)
    theta = [0, 0, 0]
    alpha = 0.1
    iterations = 2000
    theta, costs = gradient_descent(X, theta, y, alpha, iterations)
    print(theta)

    # 可视化下降过程
    plt.plot()
    plt.title("Visualizing J(θ)")
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.plot([i for i in range(iterations)], costs, color="red")
    plt.show()

    # 预测
    x1_ = (2100 - mean[0]) / ptp[0]
    x2_ = (3 - mean[1]) / ptp[1]
    y_pred = theta[0] + x1_ * theta[1] + x2_ * theta[2]
    print(y_pred)
    data_ = [x1_, x2_, y_pred]
    pred = data_*ptp + mean
    print("2100,3房价为", pred[2])

