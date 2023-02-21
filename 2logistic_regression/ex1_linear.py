import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import scipy.optimize as opt

# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 代价函数
def Cost(theta, X, y):
    return np.mean((-y) * np.log(sigmoid(X.dot(theta))) - (1 - y) * np.log(1 - sigmoid(X.dot(theta))))


# 4.梯度下降
def gradient(theta, X, y):
    return X.T.dot(sigmoid(X.dot(theta)) - y) / X.shape[0]


# 普通的梯度下降实现
def gradient_descent(theta, X, y, alpha, iterations):
    costs = []
    for i in range(iterations):
        theta -= alpha * gradient(theta, X, y)
        costs.append(Cost(theta, X, y))
    return costs, theta


def predict(theta, X):
    return [1 if i > 0.5 else 0 for i in sigmoid(X.dot(theta))]


if __name__ == '__main__':
    path = './data.csv'

    # 使用pandas加载csv
    data = pd.read_csv(path)
    data1 = data
    print(data.head())
    print(data.info())

    data = data.values
    X = np.insert(data[..., :2], 0, 1, axis=1)  # 记得添加x0
    y = data[..., -1]
    theta = np.zeros((3,))
    # print(Cost(theta, X, y))  # 0.6931471805599453

    # 特征归一化
    mean = np.mean(X[..., 1:], axis=0)
    std = np.std(X[..., 1:], axis=0, ddof=1)  # 计算标准差
    X[..., 1:] = (X[..., 1:] - mean) / std

    # 梯度下降
    alpha = 0.02
    iterations = 20000
    costs, theta = gradient_descent(theta, X, y, alpha, iterations)
    print("使用梯度下降,最后的Theta:", theta)

    plt.plot(range(iterations), costs, color="red")
    plt.show()

# 使用scipy中的高级优化算法
#     res = opt.minimize(fun=Cost, x0=theta, args=(X, y), method='TNC', jac=gradient)
#     theta = res.x
#     print("使用scipy.optimize.minimize,最后的Theta:", res.x)

    # 画出决策边界
    # plt.subplot(2, 2, 2)
    x1 = np.arange(20, 110, 0.1)
    # 因为进行了特征缩放，所以计算y时需要还原特征缩放
    x2 = mean[1] - std[1] * (theta[0] + theta[1] * (x1 - mean[0]) / std[0]) / theta[2]
    db = plt.plot(x1, x2, c='cyan', label="decision boundary")
    plt.scatter(data1[data1["accepted"] == 0]["exam1"], data1[data1["accepted"] == 0]["exam2"], color="red", marker='x',
                label='y=0')
    plt.scatter(data1[data1["accepted"] == 1]["exam1"], data1[data1["accepted"] == 1]["exam2"], color="blue",
                label='y=1')
    plt.legend()
    plt.show()

    # 测试优化结果
    test_x = np.array([40, 84.5])
    test_x = (test_x - mean) / std
    test_x = np.insert(test_x, 0, 1)
    print(sigmoid(test_x.dot(theta)))  # 0.7763928918272246   这个值为h_\theta(x)

    print("准确率=", np.mean(predict(theta, X) == y))
    # 评价
    print(classification_report(y, predict(theta, X)))
    # 预测
    x = np.array([[41, 84]])
    x = np.insert(x[..., :2], 0, 1, axis=1)
    print(predict(theta, x))



