"""
KNN Regression
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors,model_selection

def create_regression_data(n):
    """
    创建KNN回归模型的数据集
    

    :param n: 数据集的大小
    :return: 一个元组类型，依次为：训练数据集，测试数据集
             训练数据集标签，测试数据集标签
    """
    x = 5 * np.random.rand(n, 1)
    y = np.sin(x).ravel()
    y[::5] += 1 * (0.5 - np.random.rand(int(n/5)))
    return model_selection.train_test_split(x, y, test_size = 0.25,\
            random_state = 0)

def test_KNN_Regression(*data):
    train_x, test_x, train_y, test_y = data
    model = neighbors.KNeighborsRegressor()
    model.fit(train_x, train_y)
    print("Training score: %.2f" % (model.score(train_x, train_y),))
    print("Testing score: %.2f" % (model.score(test_x, test_y),))

def test_KNNRegression_k_w(*data):
    """
    测试模型随参数邻居的数量n_neignbors，和参数weights的
    影响
    """
    train_x, test_x, train_y, test_y = data
    Ks = np.linspace(1, train_y.size, num = 100, endpoint = False,\
            dtype = "int")
    weights = ["uniform","distance"]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # 绘制不同的weight下，score随n_neighbors变化的曲线
    for weight in weights:
        training_scores = []
        testing_scores = []
        for k in Ks:
            model = neighbors.KNeighborsRegressor(weights = weight,\
                    n_neighbors = k)
            model.fit(train_x, train_y)
            training_scores.append(model.score(train_x, train_y))
            testing_scores.append(model.score(test_x, test_y))
        ax.plot(Ks, training_scores,\
                label = "training score: weight = %s" % (weight,))
        ax.plot(Ks, testing_scores,\
                label = "testing score : weight = %s" % (weight,))
    ax.legend(loc = "best")
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("KNN Regression Model")
    plt.show()

def test_KNNRegression_k_p(*data):
    """
    测试KNN Regression Model 随参数n_neighbors和参数p的
    影响，其中p指的是距离度量的选取。
    比如p=1为曼哈顿距离，p=2为欧氏距离
    """
    train_x, test_x, train_y, test_y = data
    Ks = np.linspace(1, train_y.size, endpoint = False, dtype = "int")
    Ps = [1, 2, 10]
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # 绘制不同的p下，score随n_neighbors变化的曲线
    for p in Ps:
        training_scores = []
        testing_scores = []
        for k in Ks:
            model = neighbors.KNeighborsRegressor(p = p, n_neighbors = k)
            model.fit(train_x, train_y)
            training_scores.append(model.score(train_x, train_y))
            testing_scores.append(model.score(test_x, test_y))
        ax.plot(Ks, training_scores,\
                label = "traing score: p = %d" % (p,))
        ax.plot(Ks, testing_scores,\
                label = "testing score: p = %d" % (p,))
    ax.legend(loc = "best")
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("KNN Regressor Model")
    plt.show()


if __name__ == "__main__":
    train_x, test_x, train_y, test_y = create_regression_data(1000)
    #test_KNN_Regression(train_x, test_x, train_y, test_y)
    #test_KNNRegression_k_w(train_x, test_x, train_y, test_y)
    test_KNNRegression_k_p(train_x, test_x, train_y, test_y)
