"""
实现线性分类的支持向量机
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, model_selection, svm


def load_data_classification():
    iris = datasets.load_iris()   # 使用iris数据集
    train_x = iris.data
    train_y = iris.target
    return model_selection.train_test_split(train_x, train_y,
            test_size = 0.25, random_state = 0,
            stratify = train_y)


def test_LinearSVC(*data):
    train_x, test_x, train_y, test_y = data
    model = svm.LinearSVC()
    model.fit(train_x, train_y)
    print("Coefficients:%s, intercept %s" % (model.coef_, model.intercept_))
    print("Score: %.2f" % model.score(test_x, test_y))

def test_LinearSVC_loss(*data):
    """
    测试LinearSVC的性能随损失函数变化的影响
    """
    train_x, test_x, train_y, test_y = data
    losses = ["hinge", "squared_hinge"]
    for loss in losses:
        model = svm.LinearSVC(loss = loss)
        model.fit(train_x, train_y)
        print("Loss: %s" % loss)
        print("Cofficients: %s, intercept: %s" % (model.coef_,
            model.intercept_))
        print("Score: %.2f" % model.score(test_x, test_y))

def test_LinearSVC_L12(*data):
    """
    测试LinearSVC的预测性能随正则化形式的影响
    """
    train_x, test_x, train_y, test_y = data
    L12 = ["l1", "l2"]
    for p in L12:
        model = svm.LinearSVC(penalty = p, dual = False)
        model.fit(train_x, train_y)
        print("penalty: %s" % p)
        print("Cofficient: %s, intercept: %s" % (model.coef_,
            model.intercept_))
        print("Score: %.2f" % model.score(test_x, test_y))

def test_LinearSVC_C(*data):
    """
    测试 LinearSVC的预测性能随参数C变化的影响
    """
    train_x, test_x, train_y, test_y = data
    Cs = np.logspace(-2, 1)
    train_scores = []
    test_scores = []
    for C in Cs:
        model = svm.LinearSVC(C = C)
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Cs, train_scores, label = "Training Score")
    ax.plot(Cs, test_scores, label = "Testing Score")
    ax.set_xlabel(r"C")
    ax.set_ylabel(r"score")
    ax.set_xscale("log")
    ax.set_title("LinearSVC")
    ax.legend(loc = "best")
    plt.show()

def main():
    train_x, test_x, train_y, test_y = load_data_classification()
    #test_LinearSVC(train_x, test_x, train_y, test_y)
    #test_LinearSVC_loss(train_x, test_x, train_y, test_y)
    #test_LinearSVC_L12(train_x, test_x, train_y, test_y)
    test_LinearSVC_C(train_x, test_x, train_y, test_y)


if __name__ == "__main__":
    main()



