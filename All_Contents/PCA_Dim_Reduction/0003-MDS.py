import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,manifold

def load_data():
    """
    加载用于降维的数据集
    """
    iris = datasets.load_iris()
    return iris.data, iris.target

def test_MDS(*data):
    x,y = data
    for n in [4, 3, 2, 1]:     # 依次制定维度目标为4,3,2,1
        mds = manifold.MDS(n_components = n)
        mds.fit(x)
        print("stress(n_components=%d):%s" % (n, str(mds.stress_)))


def plot_MDS(*data):
    """
    绘制经过使用MDS降维到二维之后的样本点
    """
    x,y = data
    mds = manifold.MDS(n_components=2)
    x_r = mds.fit_transform(x)  # 将原始数据集转换到二维
    # 绘制二维图形
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ((1, 0, 0),(0, 1, 0),(0, 0, 1),(0.5,0.5,0),(0.5,0,0.5),\
            (0,0.5,0.5),(0.4, 0.6, 0),(0.6,0.4,0),(0,0.6,0.4),\
            (0.5, 0.3, 0.2))
    for label, color in zip(np.unique(y), colors):
        position = y == label
        ax.scatter(x_r[position,0], x_r[position,1],\
                label = "target = %d" % label, color = color)

    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")
    ax.legend(loc="best")
    ax.set_title("MDS")
    plt.show()

def main():
    x,y = load_data()
    #test_MDS(x,y)
    plot_MDS(x,y)

if __name__ == "__main__":
    main()


