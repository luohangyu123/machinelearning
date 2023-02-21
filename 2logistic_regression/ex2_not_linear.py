import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, l):
    m = X.shape[0]
    part1 = np.sum(-y * np.log(sigmoid(X.dot(theta))) - (1 - y) * np.log(1 - sigmoid(X.dot(theta))))
    part2 = (l / (2 * m)) * np.sum(np.delete((theta * theta), 0, axis=0))
    return part1 / m + part2


def gradient(theta, X, y, alpha, iters, l):
    m = X.shape[0]
    costs = []

    for i in range(iters):
        part1 = X.T.dot((sigmoid(X.dot(theta)) - y)) / m
        reg = (l / m) * theta
        reg[0] = 0
        theta = theta - (part1 + reg) * alpha
        cost_ = cost(theta, X, y, l)
        costs.append(cost_)

    return theta, costs


def predict(theta, X):
    return [1 if i > 0.5 else 0 for i in sigmoid(X.dot(theta))]


def features_mapping(x1, x2, power):
    m = len(x1)
    features = np.zeros((m, 1))
    for sum_power in range(power):
        for x1_power in range(sum_power + 1):
            x2_power = sum_power - x1_power
            features = np.concatenate((features, (np.power(x1, x1_power) * np.power(x2, x2_power)).reshape(m, 1)), axis=1)
    return np.delete(features, 0, axis=1)


if __name__ == '__main__':
    path = './data2.csv'
    # 使用pandas加载csv
    data = pd.read_csv(path)
    data1 = data
    # print(data.head())
    # print(data.info())
    # 展示数据

    # plt.legend()
    # plt.show()
    data = data.values

    # 特征映射
    features = features_mapping(data[..., 0], data[..., 1], 6)
    X = features
    y = data[..., -1]
    theta = np.zeros(features.shape[-1])
    # 测试损失函数
    print(cost(theta, features, y, 1))

    # 梯度下降
    alpha = 0.02
    iterations = 20000
    theta, costs = gradient(theta, X, y, alpha, iterations, l=0.1)
    # print("使用梯度下降,最后的Theta:", theta)

    plt.plot(range(iterations), costs, color="red")
    plt.show()
    # # 优化
    # res = opt.minimize(fun=cost, x0=theta, args=(features, y, 1), method='TNC', jac=gradient)
    # print(classification_report(y, predict(res.x, features)))

    # 画出决策边界
    x = np.linspace(-1, 1.2, 100)
    x1, x2 = np.meshgrid(x, x)
    z = features_mapping(x1.ravel(), x2.ravel(), 6)
    z = z.dot(theta).reshape(x1.shape)

    plt.contour(x1, x2, z, 0, colors="cyan")
    plt.scatter(data1[data1["accepted"] == 0]["exam1"], data1[data1["accepted"] == 0]["exam2"], color="red", marker='x',
                label='y=0')
    plt.scatter(data1[data1["accepted"] == 1]["exam1"], data1[data1["accepted"] == 1]["exam2"], color="blue", label='y=1')
    plt.legend()
    plt.show()

    # 自行修改lambda的值去观察，完成额外的练习
    print("准确率=", np.mean(predict(theta, X) == y))
    x = np.array([[0.51267, 0.722]])
    x = features_mapping(x[..., 0], x[..., 1], 6)
    print(predict(theta, x))
