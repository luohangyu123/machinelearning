import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

path = './data.csv'

# 使用pandas加载csv
dataset = pd.read_csv(path)

# 分割数据集训练集，测试集
X_train, X_test, Y_train, Y_test = train_test_split(dataset[["exam1", "exam2"]], dataset[["accepted"]], random_state=0)

# 训练模型
# 设置最大迭代次数为3000，默认为1000.不更改会出现警告提示
log_reg = LogisticRegression(max_iter=3000)
# 给模型喂入数据
clm = log_reg.fit(X_train, Y_train)

# 使用模型对测试集分类预测,并打印分类结果
print(clm.predict(X_test))
# 最后使用性能评估器，测试模型优良，用测试集对模型进行评分
print(clm.score(X_test, Y_test))

x = pd.DataFrame(np.array([[41, 84.5]]), columns=["exam1", "exam2"])
print(clm.predict(x))
