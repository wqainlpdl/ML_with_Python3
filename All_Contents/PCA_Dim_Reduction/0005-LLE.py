import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,manifold

def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target

def test_LocallyLinearEmbedding(*data):
    """
    测试局部线性嵌入的效果
    """
    x,y = data
    for n in [4, 3, 2, 1]:
        lle = manifold.LocallyLinearEmbedding(n_components=n)
        lle.fit(x)
        print("reconstruction_error(n_components=%d):%s" %\
                (n, lle.reconstruction_error_))

def plot_LocallyLinearEmbedding_k(*data):
    """
    测试LocallyLinearEmbedding中参数n_neighbors的影响
    其中，降维至二维
    """
    x, y = data
    Ks = [1, 5, 25, y.size - 1] # n_neighbors参数的候选值的集合
    fig = plt.figure()
    for i,k in enumerate(Ks):
        lle = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=k)
        x_r = lle.fit_transform(x)
        ax = fig.add_subplot(2, 2, i+1)
        colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0),\
                (0.5, 0, 0.5), (0, 0.5, 0.5), (0.6, 0.4, 0),\
                (0.4, 0.6, 0), (0, 0.6, 0.4), (0.5, 0.3, 0.2))
        for label, color in zip(np.unique(y), colors):
            position = y == label
            ax.scatter(x_r[position, 0], x_r[position, 1],\
                    label = "target=%d" % label, color = color)
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.legend(loc="best")
        ax.set_title("k=%d" % k)
    plt.suptitle("LocallyLinearEmbedding")
    plt.show()
def plot_LocallyLinearEmbedding_k_d1(*data):
    x,y = data
    Ks = [1, 5, 25, y.size - 1]  # n_neighbors 参数的候选值的集合
    fig = plt.figure()
    for i, k in enumerate(Ks):
        lle = manifold.LocallyLinearEmbedding(n_components = 1,\
                n_neighbors = k)
        x_r = lle.fit_transform(x)   # 将原始数据集降到1维
        ax = fig.add_subplot(2, 2, i+1)
        colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0),\
                (0.5, 0, 0.5), (0, 0.5, 0.5), (0.4, 0.6, 0),\
                (0.6, 0.4, 0), (0, 0.6, 0.4), (0.5, 0.3, 0.2))
        for label, color in zip(np.unique(y), colors):
            position = y == label
            ax.scatter(x_r[position], np.zeros_like(x_r[position]),\
                    label = "target = %d" % label, color = color)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc = "best")
        ax.set_title("k = %d" % k)
    plt.suptitle("LocallyLinearEmbedding")
    plt.show()


def main():
    x, y = load_data()
    #test_LocallyLinearEmbedding(x, y)
    #plot_LocallyLinearEmbedding_k(x, y)
    plot_LocallyLinearEmbedding_k_d1(x, y)


if __name__ == "__main__":
    main()


