import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets, model_selection

def load_data():
    digits = datasets.load_digits()
    train_x = digits.data
    train_y = digits.target
    return model_selection.train_test_split(train_x, train_y,\
            test_size = 0.25, random_state = 0, stratify = train_y)

def test_KNNClassifier(*data):
    train_x, test_x, train_y, test_y = data
    model = neighbors.KNeighborsClassifier()
    model.fit(train_x, train_y)
    print("Training Score: %.2f" % (model.score(train_x, train_y),))
    print("Testing Score: %.2f" % (model.score(test_x, test_y),))


def test_KNNClassifier_K_Weights(*data):
    """
    测试KNN模型随参数邻居个数k和权重参数的变化情况
    在kNN算法中，每个邻居都有相应的权重，大多数情况
    是根据距离来分配相应的权重，距离越近，一般权重越大
    """    
    train_x, test_x, train_y, test_y = data
    Ks = np.linspace(1, train_y.size, num = 100, endpoint = False,\
            dtype = 'int')
    weights = ['uniform', 'distance']

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ### 绘制不同的weight下，模型的预测得分随邻居个数的变化曲线
    for weight in weights:
        training_scores = []
        testing_scores = []
        for k in Ks:
            model = neighbors.KNeighborsClassifier(weights = weight,\
                    n_neighbors = k)
            model.fit(train_x, train_y)
            training_scores.append(model.score(train_x, train_y))
            testing_scores.append(model.score(test_x, test_y))
        ax.plot(Ks, training_scores, label = "Training Score:\
                weight = %s" % weight)
        ax.plot(Ks, testing_scores, label = "Testing Score:\
                weight = %s" % weight)
    ax.legend(loc = "best")
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("KNeighborsClassifier")
    plt.show()


def test_KNNClassifier_K_P(*data):
    """
    测试kNNModel随参数邻居的个数n_neighbors和
    参数距离度量的选取p的变化情况
    """
    train_x, test_x, train_y, test_y = data
    Ks = np.linspace(1, train_y.size, endpoint = False,\
            dtype = "int")
    Ps = [1, 2, 10]   # 当选取2时，为欧氏距离， 当选取1时，为曼哈顿
    fig = plt.figure()
    ax =fig.add_subplot(1, 1, 1)
    # 绘制不同的p值下，预测得分随邻居数n_neighbors的变化曲线
    for p in Ps:
        training_scores = []
        testing_scores = []
        for K in Ks:
            model = neighbors.KNeighborsClassifier(p = p, n_neighbors = K)
            model.fit(train_x, train_y)
            training_scores.append(model.score(train_x, train_y))
            testing_scores.append(model.score(test_x, test_y))
        ax.plot(Ks, training_scores, label = "Training Score: p =\
                %d" % p)
        ax.plot(Ks, testing_scores, label = "Testing Score: p =\
                %d" % p)
    ax.legend(loc = "best")
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("KNNClassifier-p")
    plt.show()


def main():
    train_x, test_x, train_y, test_y = load_data()
    #test_KNNClassifier(train_x, test_x, train_y, test_y)
    #test_KNNClassifier_K_Weights(train_x, test_x, train_y, test_y)
    test_KNNClassifier_K_P(train_x, test_x, train_y, test_y)

if __name__ == "__main__":
    main()
