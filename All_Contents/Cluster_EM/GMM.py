"""
本模块用来测试高斯混合模型
"""
from sklearn import mixture
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

def test_GMM(*data):
    """
    测试高斯混合模型的用法
    :param data:可变参数，元组类型：第一个元素为样本集
                第二个元素为样本集的真实簇分类标记
    :return: None
    """
    x, labels_true = data
    clst = mixture.GaussianMixture()
    clst.fit(x)
    predicted_labels = clst.predict(x)
    print("ARI:%s" % adjusted_rand_score(labels_true, predicted_labels))

def test_GMM_n_components(*data):
    """
    测试GMM的聚类结果随n_components参数的影响
    :param data:可变参数，元组类型。元组元素依次为：
            第一个元素为样本集；第二个元素为样本集的真实簇分类标记
    :return: None
    """
    x, labels_true = data
    nums = range(1, 50)
    ARIs = []
    for num in nums:
        clst = mixture.GaussianMixture(n_components = num)
        clst.fit(x)
        predicted_labels = clst.predict(x)
        ARIs.append(adjusted_rand_score(labels_true, predicted_labels))

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(nums, ARIs, marker = "+")
    ax.set_xlabel("n_components")
    ax.set_ylabel("ARI")
    fig.suptitle("GMM")
    plt.show()


def test_GMM_cov_type(*data):
    """
    测试GMM的聚类结果随协方差类型的影响

    :param data: 一个可变类型，元组。元组的第一个元素为样本集
                元组的第二个元素为样本集的真实的簇分类标记
    :return: None
    """
    x, labels_true = data
    nums = range(1, 50)
    cov_types = ["spherical", "tied", "diag", "full"]
    markers = "+o*s"
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i, cov_type in enumerate(cov_types):
        ARIs = []
        for num in nums:
            clst = mixture.GaussianMixture(n_components = num,\
                    covariance_type = cov_type)
            clst.fit(x)
            predicted_labels = clst.predict(x)
            ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
        ax.plot(nums, ARIs, marker = markers[i],\
                label = "covariance_type:%s" % (cov_type,))
    ax.set_xlabel("n_components")
    ax.legend(loc = "best")
    ax.set_ylabel("ARI")
    fig.suptitle("GMM")
    plt.show()

