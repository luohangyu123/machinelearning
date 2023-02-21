# from sklearn import tree
# import tensorflow as tf
#
# # X = [[0, 0], [2, 2]]
# # y = [0.5, 0.5]
# # clf = tree.DecisionTreeRegressor()
# # clf = clf.fit(X, y)
# # print(clf.predict([[1, 2]]))
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()
#
# clf = tree.DecisionTreeRegressor(criterion='mse', random_state=30, splitter='random')
# clf = clf.fit(X_train, y_train)
#
# print(clf.score(X_test, y_test))

from xgboost import XGBClassifier
import tensorflow as tf  # 导入模块


mnist = tf.keras.datasets.mnist  # 导入mnist数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 分别配置训练集和测试集的输入和标签

model = XGBClassifier()
model.fit(x_train.reshape(x_train.shape[0], -1), y_train)

print(model.score(x_test.reshape(x_test.shape[0], -1), y_test))

