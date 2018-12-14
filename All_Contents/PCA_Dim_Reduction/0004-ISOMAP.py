import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, manifold

def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target

def test_Isomap(*data):
    """
    测试等度量映射降维
    """
    x,y = data
    for n in [4, 3, 2, 1]:
        isomap = manifold.Isomap(n_components = n)
        isomap.fit(x)
        print("reconstruction_error(n_components=%d): %s" %\
                (n, isomap.reconstruction_error()))

def plot_Isomap_k(*data):
    """
    测试Isomap算法的性能随参数n_neighbors变化的情况，
    其中降维至二维
    """
    x,y = data
    Ks = [1, 5, 25, y.size - 1]  # n_neighbors参数的候选值的集合
    fig = plt.figure()
    for i,k in enumerate(Ks):
        isomap = manifold.Isomap(n_components = 2, n_neighbors = k)
        x_r = isomap.fit_transform(x)   # 将原始数据转换到二维
        ax = fig.add_subplot(2, 2, i+1)
        colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0),\
                (0.5, 0, 0.5), (0, 0.5, 0.5), (0.4, 0.6, 0),\
                (0.6, 0.4, 0), (0, 0.6, 0.4), (0.5, 0.3, 0.2))
        for label, color in zip(np.unique(y), colors):
            position = y == label
            ax.scatter(x_r[position, 0], x_r[position, 1],\
                    label = "target = %d" % label, color = color)
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.legend(loc = "best")
        ax.set_title("k = %d" % k)
    plt.suptitle("Isomap")
    plt.show()

def plot_Isomap_k_d1(*data):
    """
    测试Isomap中n_neighbors参数的影响，其中降维至1维
    """
    x, y = data
    Ks = [1, 5, 25, y.size - 1]
    fig = plt.figure()
    for i,k in enumerate(Ks):
        isomap = manifold.Isomap(n_components = 1, n_neighbors = k)
        x_r = isomap.fit_transform(x)  # 将原始数据降到1维
        ax = fig.add_subplot(2, 2, i+1)
        colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0),\
                (0.5, 0, 0.5), (0, 0.5, 0.5), (0.4, 0.6, 0),\
                (0.6, 0.4, 0), (0, 0.6, 0.4), (0.5, 0.3, 0.2))
        for label, color in zip(np.unique(y), colors):
            position = y == label
            ax.scatter(x_r[position], np.zeros_like(x_r[position]),\
                    label = "target = %d" % label, color = color)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc = "best")
        ax.set_title("k = %d" % k)
    plt.suptitle("Isomap")
    plt.show()


if __name__ == "__main__":
    x, y = load_data()
    #test_Isomap(x,y)
    #plot_Isomap_k(x,y)
    plot_Isomap_k_d1(x,y)
