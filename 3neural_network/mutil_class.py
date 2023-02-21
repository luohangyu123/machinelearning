
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import classification_report


def show_img(X):
    img = []

    for i in range(100):

        index = np.random.randint(X.shape[0])
        img.append(X[index].reshape(20, 20))

        plt.subplot(10, 10, i + 1)
        plt.imshow(img[i].T)  # 画图
        plt.axis('off')  # 关闭坐标轴

    plt.show()  # 显示图片


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, l):
    m = X.shape[0]
    part1 = np.mean((-y) * np.log(sigmoid(X.dot(theta))) - (1 - y) * np.log(1 - sigmoid(X.dot(theta))))
    part2 = (l / (2 * m)) * np.sum(theta * theta)
    return part1 + part2


def gradient(theta, X, y, l):
    m = X.shape[0]
    part1 = X.T.dot(sigmoid(X.dot(theta)) - y)
    part2 = (l / m) * theta
    part2[0] = 0
    return part1 + part2


def convert(y):
    n = len(np.unique(y))
    res = False
    for i in y:
        temp = np.zeros((1, n))
        temp[0][i[0] % 10] = 1
        if type(res) == bool:
            res = temp
        else:
            res = np.concatenate((res, temp), axis=0)
    return res


def predict(theta, X):
    p = sigmoid(X.dot(theta.T))
    res = False
    for i in p:
        index = np.argmax(i)
        temp = np.zeros((1, 10))
        temp[0][index] = 1
        if type(res) == bool:
            res = temp
        else:
            res = np.concatenate((res, temp), axis=0)
    return res


if __name__ == '__main__':
    data = sio.loadmat("ex3data1.mat")
    print(data.keys())  # 查看其中包含的key-value键值对
    X = data['X']  # 通过字典的访问方式得到数据
    y = data['y']
    print(X, type(X), y, type(y))

    # show_img(X)

    y = convert(y)
    # print(y)
    X = np.insert(X, 0, 1, axis=1)
    m = X.shape[0]
    n = X.shape[1] - 1
    theta = np.zeros((n + 1,))
    trained_theta = False
    for i in range(y.shape[-1]):
        res = opt.minimize(fun=cost, x0=theta, args=(X, y[..., i], 1), method="TNC", jac=gradient)
        if type(trained_theta) == bool:
            trained_theta = res.x.reshape(1, n + 1)
        else:
            trained_theta = np.concatenate((trained_theta, res.x.reshape(1, n + 1)), axis=0)

    print(classification_report(y, predict(trained_theta, X), digits=4))
