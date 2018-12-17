# 本模块用来测试之前所有的聚类模型，包括KMeans，密度聚类(DBSCAN)
# 以及层次聚类(AgglomerativeClustering)
# 通过对这些无监督学习中聚类模型的测试，来展示效果
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs  # 聚类数据生成器
from kmeans import *  #从本文件夹中的kmeans模块中导入所有的成员
from dbscan import *  #从本文件夹中的dbscan模块中导入所有成员
from agglomerative_clustering import *

# 创建测试不同聚类算法的数据
def create_data(centers, num = 100, std = 0.5):
    """
    生成用于聚类的数据集

    :param centers:聚类的中心点组成的数组，如果中心点是二维的，则产生的
                    每个样本都是二维的。
    :param num:样本数
    :param std:每个簇中样本的标准差
    :return:用于聚类的数据集。是一个元组，第一个元素是样本集
            第二个元素是样本集的真实簇分类标记
    """
    x, labels_true = make_blobs(n_samples = num, centers = centers,\
                                cluster_std = std)
    return x, labels_true


def plot_data(*data):
    """
    绘制用于聚类的数据集

    :param data: 可变参数，是一个元组。元组的元素依次为:第一个元素为样本集
                第二个元素为样本集的真实簇分类标记
    :return: None
    """
    x, labels_true = data
    labels = np.unique(labels_true)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = "rgbyckm"  # 每个簇的样本标记不同的颜色
    for i, label in enumerate(labels):
        position = labels_true == label
        ax.scatter(x[position, 0], x[position, 1],\
                label = "cluster %d" % label,\
                color = colors[i % len(colors)])
    ax.legend(loc = "best", framealpha = 0.5)
    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")
    ax.set_title("data")
    plt.show()

# 定义主函数
def main():
    centers = [[1, 1], [2, 2], [1, 2], [10, 20]] # 中心点的维度是二维
                                    # 初始化为4个簇，根据此来生成数据
    x, labels_true = create_data(centers, 1000, 0.5) # 根据簇中心生成
                                    # 1000个样本，每个簇中样本的标准差
                                    # 为0.5
    #plot_data(x, labels_true)   # 绘制用于聚类的数据集
    #test_Kmeans(x, labels_true)
    #test_Kmeans_nclusters(x, labels_true)
    #test_KMeans_n_init(x, labels_true)
    #test_DBSCAN(x, labels_true)
    #test_DBSCAN_epsilon(x, labels_true)
    #test_DBSCAN_min_samples(x, labels_true)
    #test_AgglomerativeClustering(x, labels_true)
    #test_AgglomerativeClustering_nclusters(x, labels_true)
    #test_AgglomerativeClustering_linkage(x, labels_true)


if __name__ == "__main__":
    main()
