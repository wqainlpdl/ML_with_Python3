"""
测试非线性分类的SVM
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, model_selection, svm

def load_data_classification():
    iris = datasets.load_iris()
    train_x = iris.data
    train_y = iris.target
    return model_selection.train_test_split(train_x, train_y, 
            test_size = 0.25, random_state = 0, stratify = train_y)


def test_SVC_linear(*data):
    """
    测试SVC的用法，在这里使用线性核
    """
    train_x, test_x, train_y, test_y = data
    model = svm.SVC(kernel = "linear")
    model.fit(train_x, train_y)
    print("Cofficient: %s, intercept: %s" % (model.coef_, model.intercept_))
    print("Score: %.2f" % model.score(test_x, test_y))

def test_SVC_poly(*data):
    """
    测试多项式核的SVC的预测性能随 degree、gamma、coef0的影响
    """
    train_x, test_x, train_y, test_y = data
    fig = plt.figure()
    # 测试degree
    degrees = range(1, 20)
    train_scores = []
    test_scores = []
    for degree in degrees:
        model = svm.SVC(kernel = "poly", degree = degree)
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))
    ax = fig.add_subplot(1, 3, 1)  # 一行三列
    ax.plot(degrees, train_scores, label = "Training Scores", marker = "+")
    ax.plot(degrees, test_scores, label = "Testing Scores", marker = "o")
    ax.set_title("SVC_poly_degree")
    ax.set_xlabel("p")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc = "best", framealpha = 0.5)
    # 测试gamma，此时degree固定为3
    gammas = range(1, 20)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        model = svm.SVC(kernel = "poly", gamma = gamma, degree = 3)
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(gammas, train_scores, label = "Training Scores", marker = "+")
    ax.plot(gammas, test_scores, label = "Testing Scores", marker = "o")
    ax.set_title("SVC_poly_gamma")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc = "best", framealpha = 0.5)
    # 测试 r，此时将gamma固定为10，degree固定为3
    rs = range(0, 20)
    train_scores = []
    test_scores = []
    for r in rs:
        model = svm.SVC(kernel = "poly", gamma = 10, degree = 3, coef0 = r)
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(rs, train_scores, label = "Training Scores", marker = "+")
    ax.plot(rs, test_scores, label = "Tesing Scores", marker = "o")
    ax.set_title("SVC_poly_r")
    ax.set_xlabel(r"r")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc = "best", framealpha = 0.5)
    plt.show()

def test_SVC_rbf(*data):
    """
    测试rbf核的SVC的预测性能随参数gamma的变化的情况
    """
    train_x, test_x, train_y, test_y = data
    gammas = range(1, 20)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        model = svm.SVC(kernel = "rbf", gamma = gamma)
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(gammas, train_scores, label = "Training Score", marker = "+")
    ax.plot(gammas, test_scores, label = "Tesing Score", marker = "o")
    ax.set_title("SVC_rbf")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc = "best", framealpha = 0.5)
    plt.show()


def test_SVC_sigmoid(*data):
    """
    测试sigmoid核的SVC的预测性能随参数gamma，r的变化的影响
    """
    train_x, test_x, train_y, test_y = data
    fig = plt.figure()
    # 测试gamma，固定r即参数coef0为0
    gammas = np.logspace(-2, 1)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        model = svm.SVC(kernel = "sigmoid", gamma = gamma, coef0 = 0)
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(gammas, train_scores, label = "Training Score", marker = "+")
    ax.plot(gammas, test_scores, label = "Testing Score", marker = "o")
    ax.set_title("SVC_sigmoid_gamma")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc = "best", framealpha = 0.5)
    # 测试r，固定gamma为0.01
    rs = np.linspace(0, 5)
    train_scores = []
    test_scores = []

    for r in rs:
        model = svm.SVC(kernel = "sigmoid", coef0 = r, gamma = 0.01)
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(rs, train_scores, label = "Training Score", marker = "+")
    ax.plot(rs, test_scores, label = "Testing Score", marker = "o")
    ax.set_title("SVC_sigmoid_r")
    ax.set_xlabel(r"r")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc = "best", framealpha = 0.5)
    plt.show()

def main():
    train_x, test_x, train_y, test_y = load_data_classification()
    #test_SVC_linear(train_x, test_x, train_y, test_y)
    #test_SVC_poly(train_x, test_x, train_y, test_y)
    #test_SVC_rbf(train_x, test_x, train_y, test_y)
    test_SVC_sigmoid(train_x, test_x, train_y, test_y)

if __name__ == "__main__":
    main()

