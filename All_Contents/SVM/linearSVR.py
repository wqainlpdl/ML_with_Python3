"""
测试线性回归的SVM
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, model_selection, svm

def load_data_regression():
    """
    加载用于回归的数据集
    """
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data, diabetes.target,
            test_size = 0.25, random_state = 0)


def test_LinearSVR(*data):
    """
    测试LinearSVR的用法
    """
    train_x, test_x, train_y, test_y = data
    model = svm.LinearSVR()
    model.fit(train_x, train_y)
    print("Coefficient: %s, intercept: %s" % (model.coef_,
        model.intercept_))
    print("Score: %.2f" % model.score(test_x, test_y))


def test_LinearSVR_loss(*data):
    """
    测试LinearSVR的预测性能随不同损失函数的影响
    """
    train_x, test_x, train_y, test_y = data
    losses = ["epsilon_insensitive", "squared_epsilon_insensitive"]
    for loss in losses:
        model = svm.LinearSVR(loss = loss)
        model.fit(train_x, train_y)
        print("loss: %s" % loss)
        print("Cofficients: %s, intercept: %s" % (model.coef_,
            model.intercept_))
        print("Score: %.2f" % model.score(test_x, test_y))

def test_LinearSVR_epsilon(*data):
    """
    测试LinearSVR的预测性能随eposilon参数的影响
    """
    train_x, test_x, train_y, test_y = data
    epsilons = np.logspace(-2, 2)
    train_scores = []
    test_scores = []
    for epsilon in epsilons:
        model = svm.LinearSVR(epsilon = epsilon,
                loss = "squared_epsilon_insensitive")
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epsilons, train_scores, label = "Training Score",
            marker = "+")
    ax.plot(epsilons, test_scores, label = "Testing Score",
            marker = "o")
    ax.set_title("LinearSVR_epsilon")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc = "best")
    plt.show()

def test_LinearSVR_C(*data):
    """
    测试LinearSVR的预测性能随罚项系数C的变化情况
    """
    train_x, test_x, train_y, test_y = data
    Cs = np.logspace(-1, 2)
    train_scores = []
    test_scores = []
    for C in Cs:
        model = svm.LinearSVR(epsilon = 0.1, 
                loss = "squared_epsilon_insensitive", C = C)
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Cs, train_scores, label = "Training Score", marker = "+")
    ax.plot(Cs, test_scores, label = "Testing Score", marker = "o")
    ax.set_title("LinearSVR_C")
    ax.set_xscale("log")
    ax.set_xlabel(r"C")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.05)
    ax.legend(loc = "best")
    plt.show()

def main():
    train_x, test_x, train_y, test_y = load_data_regression()
    #test_LinearSVR(train_x, test_x, train_y, test_y)
    #test_LinearSVR_loss(train_x, test_x, train_y, test_y)
    #test_LinearSVR_epsilon(train_x, test_x, train_y, test_y)
    test_LinearSVR_C(train_x, test_x, train_y, test_y)
if __name__ == "__main__":
    main()
