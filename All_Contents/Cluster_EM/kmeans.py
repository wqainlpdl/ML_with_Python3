from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

def test_Kmeans(*data):
    x,labels_true = data
    clst = cluster.KMeans()
    clst.fit(x)
    predicted_labels = clst.predict(x)
    print("ARI: %s" % adjusted_rand_score(labels_true, predicted_labels))
    print("Sum center distance %s" % (clst.inertia_,))


def test_Kmeans_nclusters(*data):
    """
    测试KMeans的聚类结果随参数n_clusters的参数的影响
    在这里，主要分别研究ARI和所有样本距离各簇中心的距离值和随簇的个数
    的变化情况
    """
    x, labels_true = data
    nums = range(1, 50)
    ARIs = []
    Distances = []
    for num in nums:
        clst = cluster.KMeans(n_clusters = num)
        clst.fit(x)
        predicted_labels = clst.predict(x)
        ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
        Distances.append(clst.inertia_)
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(nums, ARIs, marker = "+")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(nums, Distances, marker = "o")
    ax.set_xlabel("n_cluster")
    ax.set_ylabel("intertia_")
    fig.suptitle("KMeans")
    plt.show()


def test_KMeans_n_init(*data):
    """
    该函数考察KMeans算法运行的次数和选择的初始中心向量策略的影响
    """
    x, labels_true = data
    nums = range(1, 50)
    # 绘图
    fig = plt.figure()

    ARIs_k = []
    Distances_k = []
    ARIs_r = []
    Distances_r = []
    for num in nums:
        clst = cluster.KMeans(n_init = num, init = "k-means++")
        clst.fit(x)
        predicted_labels = clst.predict(x)
        ARIs_k.append(adjusted_rand_score(labels_true, predicted_labels))
        Distances_k.append(clst.inertia_)
        
        clst = cluster.KMeans(n_init = num, init = "random")
        clst.fit(x)
        predicted_labels = clst.predict(x)
        ARIs_r.append(adjusted_rand_score(labels_true, predicted_labels))
        Distances_r.append(clst.inertia_)
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(nums, ARIs_k, marker = "+", label = "k-means++")
    ax.plot(nums, ARIs_r, marker = "+", label = "random")
    ax.set_xlabel("n_init")
    ax.set_ylabel("ARI")
    ax.set_ylim(0, 1)
    ax.legend(loc = "best")
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(nums, Distances_k, marker = "o", label = "k-means++")
    ax.plot(nums, Distances_r, marker = "o", label = "random")
    ax.set_xlabel("n_init")
    ax.set_ylabel("inertia_")
    ax.legend(loc = "best")
    fig.suptitle("KMeans")
    plt.show()


