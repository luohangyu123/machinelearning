import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

k_range = range(1, 31)
k_error = []
#循环，取k=1到k=31，查看误差效果
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    #cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
    scores = cross_val_score(knn, x_train, y_train, cv=6, scoring='accuracy')
    k_error.append(1 - scores.mean())

#画图，x轴为k值，y值为误差值
plt.plot(k_range, k_error)
plt.xlabel('Value of K for KNN')
plt.ylabel('Error')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)

knn = KNeighborsClassifier(11)  # 调用KNN分类器
knn.fit(X_train, y_train)  # 训练KNN分类器
print(knn.predict(X_test))  # 预测值
print(y_test)  # 真实值
print("准确率=", np.mean(knn.predict(X_test) == y_test))
