from sklearn.decomposition import PCA

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# 读取数据集
x = datasets.load_iris().data
y = datasets.load_iris().target
# print(x,y)
# 数据标准化
std = StandardScaler()
x_std = std.fit_transform(x)
# 第二种方法：使用sklearn包
pca = PCA(n_components=2)  # 保留特证数目
Y = pca.fit_transform(x_std)
print(Y)
