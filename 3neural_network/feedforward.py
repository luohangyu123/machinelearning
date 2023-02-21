import numpy as np
import scipy.io as sio


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    data = sio.loadmat("ex3data1.mat")
    theta = sio.loadmat("ex3weights.mat")
    print(data.keys())  # 查看其中包含的key-value键值对
    X = data['X']  # 通过字典的访问方式得到数据
    y = data['y']
    # print(X, type(X), y, type(y))

    X = np.insert(X, 0, 1, axis=1)
    y = y.flatten()
    theta1 = theta["Theta1"]
    theta2 = theta["Theta2"]

    a1 = X
    a2 = sigmoid(X.dot(theta1.T))
    a2 = np.insert(a2, 0, 1, axis=1)
    a3 = sigmoid(a2.dot(theta2.T))

    predict_y = np.argmax(a3, axis=1)
    predict_y += 1
    print("准确率为:", np.mean(predict_y == y))


