# 本模块用来测试密度聚类DBSCAN
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import numpy as np

import matplotlib.pyplot as plt
def test_DBSCAN(*data):
    x, labels_true = data
    clst = cluster.DBSCAN()
    predicted_labels = clst.fit_predict(x)
    print("ARI: %s" % adjusted_rand_score(labels_true, predicted_labels))
    print("Core sample num: %d" % (len(clst.core_sample_indices_),))

def test_DBSCAN_epsilon(*data):
    """
    考察邻域epsilon对聚类效果的影响
    """
    x, labels_true = data
    epsilons = np.logspace(-1, 1.5)
    ARIs = []
    Core_nums = []
    for epsilon in epsilons:
        clst = cluster.DBSCAN(eps = epsilon)
        predicted_labels = clst.fit_predict(x)
        ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
        Core_nums.append(len(clst.core_sample_indices_))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(epsilons, ARIs, marker = "+")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ARI")

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(epsilons, Core_nums, marker = "o")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("Core_Nums")

    fig.suptitle("DBSCAN")
    plt.show()


def test_DBSCAN_min_samples(*data):
    """
    测试DBSCAN的聚类结果随min_samples参数的影响
    """
    x, labels_true = data
    min_samples = range(1,100)
    ARIs = []
    Core_nums = []
    for num in min_samples:
        clst = cluster.DBSCAN(min_samples = num)
        predicted_labels = clst.fit_predict(x)
        ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
        Core_nums.append(len(clst.core_sample_indices_))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(min_samples, ARIs, marker = "+")
    ax.set_xlabel("min_samples")
    ax.set_ylabel("ARI")
    ax.set_ylim(0, 1)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(min_samples, Core_nums, marker = "o")
    ax.set_xlabel("min_samples")
    ax.set_ylabel("Core_nums")
    plt.suptitle("DBSCAN")
    plt.show()

