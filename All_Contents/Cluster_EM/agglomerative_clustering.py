# 测试层次聚类

from sklearn import cluster

from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

def test_AgglomerativeClustering(*data):
    """
    测试在无监督学习中密度聚类的聚类效果
    """
    x, labels_true = data
    clst = cluster.AgglomerativeClustering()
    predicted_labels = clst.fit_predict(x)
    print("ARI: %s" % adjusted_rand_score(labels_true, predicted_labels))


def test_AgglomerativeClustering_nclusters(*data):
    """
    测试agglomerativeClustering的聚类结果随参数n_clusters参数的
    影响
    """
    x, labels_true = data
    nums = range(1, 50)
    ARIs = []
    for num in nums:
        clst = cluster.AgglomerativeClustering(n_clusters = num)
        predicted_labels = clst.fit_predict(x)
        ARIs.append(adjusted_rand_score(labels_true, predicted_labels))

    # 绘制曲线图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(nums, ARIs, marker = "+")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    fig.suptitle("AgglomerativeClustering")
    plt.show()

def test_AgglomerativeClustering_linkage(*data):
    """
    测试AgglomerativeClustering的聚类结果
    随链接方式的影响
    """
    x, labels_true = data
    nums = range(1, 50)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    linkages = ["ward", "complete", "average"]
    markers = "+o*"
    for i, linkage in enumerate(linkages):
        ARIs = []
        for num in nums:
            clst = cluster.AgglomerativeClustering(n_clusters = num,\
                    linkage = linkage)
            predicted_labels = clst.fit_predict(x)
            ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
        ax.plot(nums,ARIs,marker=markers[i], label = "linkage:%s"%linkage)
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax.legend(loc = "best")
    fig.suptitle("AgglomerativeClustering")
    plt.show()

