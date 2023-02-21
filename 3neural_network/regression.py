import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

x_data = np.linspace(-1, 1, 200)[np.newaxis, :]
bias = 0.5
y_data = np.square(x_data) + bias + np.random.normal(0, 0.05, x_data.shape)

plt.scatter(np.arange(x_data.shape[1]), y_data[0])  # squeeze将shape为1的单维度删除 np.squeeze(y_data)
plt.show()


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])
print(x_data.shape, y_data.shape)
model.compile(loss='mse', metrics=['mae'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

