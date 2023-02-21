import tensorflow as tf  # 导入模块


mnist = tf.keras.datasets.mnist  # 导入mnist数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 分别配置训练集和测试集的输入和标签
print(x_train, y_train)
# x_train, x_test = x_train / 255.0, x_test / 255.0  # 把输入特征做归一化处理，让值在0-1之间，更容易让神经网络吸收

model = tf.keras.models.Sequential([  # model.Sequential()搭建神经网络
    tf.keras.layers.Flatten(),  # 把数据集变成一位数组
    tf.keras.layers.Dense(128, activation="relu"),  # 构建128个神经元，激活函数为relu的全连接层
    tf.keras.layers.Dense(10, activation="softmax")  # 构建10个神经元，激活函数为softmax的全连接层
])

model.compile(  # model.compile()配置训练方法
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']  # 标注网络评价指标为sparse_categorical_accuracy
)

model.fit(  # model.fit()用来执行训练过程
    x_train,  # 训练集的输入
    y_train,  # 训练集的标签
    batch_size=32,  # 每一次喂入的数据是32
    epochs=5,  # 迭代数是5
    validation_data=(x_test, y_test),  # 测试集的输入特征和标签
    validation_freq=1  # 测试的间隔次数
)

model.summary()  # 输入神经网络的网络参数
