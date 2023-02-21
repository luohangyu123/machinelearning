import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random


def find_closet_centroids(X, init_centroids):
    centroid_index = np.zeros((1,))  # 记录各点距离哪个centroids最近 把索引记下来
    print(centroid_index)
    for x in X:
        centroid_index = np.append(centroid_index, np.argmin(np.sqrt(np.sum((init_centroids - x) ** 2, axis=1))))
    return centroid_index[1:]


def compute_centroids(X, centroid_index):
    next_centroids = np.zeros([3, 2])
    counts = np.zeros([3, 2])
    for i in range(X.shape[0]):
        next_centroids[int(centroid_index[i])] += X[i]
        counts[int(centroid_index[i])] += 1
    return next_centroids / counts


def random_initialization(X, K):
    res = np.zeros((1, X.shape[-1]))
    m = X.shape[0]
    rl = []
    while True:
        index = random.randint(0, m)
        if index not in rl:
            rl.append(index)
        if len(rl) >= K:
            break
    for index in rl:
        res = np.concatenate((res, X[index].reshape(1, -1)), axis=0)
    return res[1:]


def cost(X, centroid_index, centriods):
    c = 0
    for i in range(len(X)):
        c += np.sum((X[i] - centriods[int(centroid_index[i])]) ** 2)
    c /= len(X)
    return c


def k_means(X, cols):
    centroids = random_initialization(X, cols)

    # centroids = np.array([[3, 3], [6, 2], [8, 5]])
    centroid_index = np.zeros((1,))
    last_cost = -1
    now_cost = -2
    costs = []
    while now_cost != last_cost:  # 当收敛时结束算法，或者可以利用指定迭代轮数
        centroid_index = find_closet_centroids(X, centroids)
        last_cost = now_cost
        now_cost = cost(X, centroid_index, centroids)
        centroids = compute_centroids(X, centroid_index)
        costs.append(now_cost)
    return centroid_index, centroids, costs


if __name__ == "__main__":
    data = sio.loadmat("data\\ex7data2.mat")
    X = data['X']  # (300,2)
    #
    # init_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    # centroid_index = find_closet_centroids(X, init_centroids)
    #
    # print(compute_centroids(X, centroid_index))
    centroid_index, centroids, costs = k_means(X, 3)

    plt.scatter(X[..., 0], X[..., 1], c=centroid_index)
    plt.scatter(centroids[..., 0], centroids[..., 1], c='r', marker='+')
    plt.show()

    # 代价函数可视化
    # print(len(costs))
    # plt.plot(range(len(costs)), costs, 'r--')
    # plt.xticks(range(len(costs)), [i+1 for i in range(len(costs))])
    # plt.xlabel("iterations")
    # plt.show()

    # visualizing(X, idx, centroids_all)
