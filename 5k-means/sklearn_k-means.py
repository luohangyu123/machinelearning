from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import scipy.io as sio

# 构造数据样本点集X，并计算K-means聚类
data = sio.loadmat("data\\ex7data2.mat")
X = data['X']  # (300,2)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# 输出及聚类后的每个样本点的标签（即类别），预测新的样本点所属类别
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_
plt.scatter(X[..., 0], X[..., 1], c=labels)
print(cluster_centers)
plt.scatter(cluster_centers[..., 0], cluster_centers[..., 1], c='r', marker='+')
plt.show()
print(kmeans.predict([[0, 0], [4, 4], [5, 7]]))
