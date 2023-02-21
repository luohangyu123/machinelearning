import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()  # 加载数据

# 对数据进行标准化预处理，方便神经网络更好的学习  归一化
mean = X_train.mean(axis=0)
X_train -= mean
std = X_train.std(axis=0)
X_train /= std
X_test -= mean
X_test /= std


# # 构建神经网络模型
# def build_model():
#     # 这里使用Sequential模型
#     model = tf.keras.models.Sequential()
#     # 进行层的搭建，注意第二层往后没有输入形状(input_shape)，它可以自动推导出输入的形状等于上一层输出的形状
#     model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
#     model.add(tf.keras.layers.Dense(64, activation='relu'))
#     model.add(tf.keras.layers.Dense(1))
#     # 编译网络
#     model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
#     return model


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mse', metrics=['mae'])
predicts = model.predict(X_test)

from sklearn import metrics
print(metrics.mean_absolute_error(y_test, predicts))
