"""
利用PCA来实现降维
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition

def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target

def test_PCA(*data):
    x, y = data
    pca = decomposition.PCA(n_components = 2)
    pca.fit(x)
    x_r = pca.transform(x)  # 将原始数据集转换到二维
    #绘制二维数据
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),\
            (0.5,0.5,0),(0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),\
            (0.5,0.3,0.2),) #颜色集合，不同标记的样本，染不同的色
    for label,color in zip(np.unique(y),colors):
        position=y==label
        ax.scatter(x_r[position,0],x_r[position,1],\
                label="target=%d" % label, color = color)
    ax.set_xlabel("x[0]")
    ax.set_ylabel("y[0]")
    ax.legend(loc="best")
    ax.set_title("PCA")
    plt.show()


if __name__ == "__main__":
    x, y = load_data()
    test_PCA(x,y)
