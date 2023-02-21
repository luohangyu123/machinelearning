import pandas as pd
import tensorflow as tf

path = './data2.csv'
# 使用pandas加载csv
data = pd.read_csv(path)
data = data.values
x = data[..., :2]
y = data[..., -1]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

model.fit(x, y, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)


