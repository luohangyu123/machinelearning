import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

def gaussian_distribution(X, mu, sigma2):

    p = (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-(X - mu) ** 2 / (2 * sigma2))
    return np.prod(p, axis=1)  # 横向累乘


def visualize_contours(mu, sigma2):

    x = np.linspace(5, 25, 100)
    y = np.linspace(5, 25, 100)
    xx, yy = np.meshgrid(x, y)
    X = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)
    z = gaussian_distribution(X, mu, sigma2).reshape(xx.shape)
    cont_levels = [10 ** h for h in range(-20, 0, 3)]  # 当z为当前列表的值时才绘出等高线
    plt.contour(xx, yy, z, cont_levels)


# yp预测的y,yt真实的y,两个参数都为(m,)
def error_analysis(yp, yt):
    # 混淆矩阵计算四个值，然后计算f1,得分越高越好
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(yp)):
        if yp[i] == yt[i]:
            if yp[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if yp[i] == 1:
                fp += 1
            else:
                fn += 1
    precision = tp / (tp + fp) if tp + fp else 0  # 防止除以0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return f1


# 传入的两个参数都为(m,)
def select_threshold(y_value, pval):

    epsilons = np.linspace(min(pval), max(pval), 1000)
    f1_score = np.zeros((1, 2))
    for epsilon in epsilons:
        y_predict = (pval < epsilon).astype(float)  # 计算出的p与阈值相比，小于阈值预测为1(异常)
        f1 = error_analysis(y_predict, y_value)     # 计算f1分数
        f1_score = np.concatenate((f1_score, np.array([[epsilon, f1]])), axis=0)
    index = np.argmax(f1_score[..., 1])
    return f1_score[index, 0], f1_score[index, 1]


def detection(X, epsilon, mu, sigma2):

    p = gaussian_distribution(X, mu, sigma2)
    anomaly_points = np.array([X[i] for i in range(len(p)) if p[i] < epsilon])
    return anomaly_points


def circle_anomaly_points(X):
    plt.scatter(X[..., 0], X[..., 1], s=80, facecolors='none', edgecolors='r', label='anomaly point')


if __name__ == "__main__":
    data = sio.loadmat("ex8data1.mat")
    X = data['X']  # (307,2)
    plt.scatter(X[..., 0], X[..., 1], marker='x', label='point')

    mu = np.mean(X, axis=0)  # 计算方向因该是沿着0，遍历每组数据
    sigma2 = np.var(X, axis=0)  # N-ddof为除数,ddof默认为0

    p = gaussian_distribution(X, mu, sigma2)
    visualize_contours(mu, sigma2)

    X_value = data['Xval']  # (307,2)
    y_value = data['yval']  # (307,1)
    epsilon, f1 = select_threshold(y_value.ravel(), gaussian_distribution(X_value, mu, sigma2))
    print('best choice of epsilon is ', epsilon, ',the F1 score is ', f1)
    anomaly_points = detection(X, epsilon, mu, sigma2)  # 利用选择出来的阈值完善模型，对异常进行预测

    circle_anomaly_points(anomaly_points)
    plt.title('anomaly detection')
    plt.legend()
    plt.show()

    # High dimensional dataset
    data2 = sio.loadmat("ex8data2.mat")
    X = data2['X']
    Xval = data2['Xval']
    yval = data2['yval']
    mu = np.mean(X, axis=0)  # 计算方向因该是沿着0，遍历每组数据
    sigma2 = np.var(X, axis=0)  # N-ddof为除数,ddof默认为0
    e, f1 = select_threshold(yval.ravel(), gaussian_distribution(Xval, mu, sigma2))
    anomaly_points = detection(X, e, mu, sigma2)
    print('\n\nfor this high dimensional dataset \nbest choice of epsilon is ', e, ',the F1 score is ', f1)
    print('the number of anomaly points is', anomaly_points.shape[0])
