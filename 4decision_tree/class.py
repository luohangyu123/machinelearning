import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# path = './data.csv'
# # 使用pandas加载csv
# dataset = pd.read_csv(path)
# data = dataset.values
# X = data[:-20, :2]
# y = data[:-20, 2]
# X_test = data[-20:, :2]
# y_test = data[-20:, 2]
# model = DecisionTreeClassifier(random_state=0)
# model.fit(X, y)
#
# print(model.predict([[34, 78]]))
# print(model.score(X_test, y_test))

# from sklearn import datasets
# import numpy as np
#
# # 导入训练集 以及训练集的标签
# x_train = datasets.load_iris().data[:130, ...]
# y_train = datasets.load_iris().target[:130, ...]
# np.random.seed(116)
# np.random.shuffle(x_train)  # 打乱数据集
# np.random.seed(116)
# np.random.shuffle(y_train)
#
#
# model = DecisionTreeClassifier(criterion="entropy", random_state=0)
# model.fit(x_train, y_train)
# #
# print(model.predict([[6, 3, 4, 1.4]]))
# print(model.score(datasets.load_iris().data[130:, ...], datasets.load_iris().target[130:, ...]))

import tensorflow as tf  # 导入模块


mnist = tf.keras.datasets.mnist  # 导入mnist数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 分别配置训练集和测试集的输入和标签

model = DecisionTreeClassifier(criterion="entropy", random_state=0)
model.fit(x_train.reshape(x_train.shape[0], -1), y_train)

print(model.score(x_test.reshape(x_test.shape[0], -1), y_test))
