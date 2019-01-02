"""
非线性回归SVR
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, model_selection,svm

def load_data_regression():
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data,
            diabetes.target, test_size = 0.25,
            random_state = 0)

def test_SVR_linear(*data):
    """
    测试SVR，使用线性核
    """
    train_x, test_x, train_y, test_y = data
    model = svm.SVR(kernel = "linear")
    model.fit(train_x, train_y)
    print("Coefficients: %s, intercept: %s" % (model.coef_,
        model.intercept_))
    print("Score: %.2f" % model.score(test_x, test_y))

def test_SVR_poly(*data):
    """
    测试SVR，使用多项式核
    """
    train_x, test_x, train_y, test_y = data
    fig = plt.figure()
    # 测试degree，固定其他项
    degrees = range(1, 20)
    train_scores = []
    test_scores = []
    for degree in degrees:
        model = svm.SVR(kernel = "poly", degree = degree, coef0 = 1)
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(degrees, train_scores, label = "Training Score", marker = "+")
    ax.plot(degrees, test_scores, label = "Testing Score", marker = "o")
    ax.set_title("SVR_poly_degree r = 1")
    ax.set_xlabel("p")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.)
    ax.legend(loc = "best", framealpha = 0.5)
    # 测试gamma，固定其他参数，其中设置degree = 3, r = 1
    gammas = range(1, 40)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        model = svm.SVR(kernel = "poly", gamma = gamma, degree = 3,
                coef0 = 1)
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(gammas, train_scores, label = "Training Score", marker = "+")
    ax.plot(gammas, test_scores, label = "Testing Score", marker = "o")
    ax.set_title("SVR_poly_gamma r = 1")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.)
    ax.legend(loc = "best")
    
    # 测试r，固定其他参数，其中设置gamma = 20, degree = 3
    rs = range(0, 20)
    train_scores = []
    test_scores = []
    for r in rs:
        model = svm.SVR(kernel = "poly", gamma = 20, degree = 3, coef0 = r)
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(rs, train_scores, label = "Training Score", marker = "+")
    ax.plot(rs, test_scores, label = "Testing Score", marker = "o")
    ax.set_title("SVR_poly_r gamma = 20, degree = 3")
    ax.set_xlabel(r"r")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.)
    ax.legend(loc = "best", framealpha = 0.5)
    plt.show()

def test_SVR_rbf(*data):
    """
    测试高斯核的SVR的预测性能随gamma参数的影响
    """
    train_x, test_x, train_y, test_y = data
    gammas = range(1, 20)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        model = svm.SVR(kernel = "rbf", gamma = gamma)
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(gammas, train_scores, label = "Training Score", marker = "+")
    ax.plot(gammas, test_scores, label = "Testing Score", marker = "o")
    ax.set_title("SVR_rbf")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.)
    ax.legend(loc = "best", framealpha = 0.5)
    plt.show()

def test_SVR_sigmoid(*data):
    """
    测试sigmoid核的SVR的预测性能随gamma、coef0的影响
    """
    train_x, test_x, train_y, test_y = data
    fig = plt.figure()
    # 测试gamma，固定r = 0.01
    gammas = np.logspace(-1, 3)
    train_scores = []
    test_scores = []
    for gamma in gammas:
        model = svm.SVR(kernel = "sigmoid", gamma = gamma, coef0 = 0.01)
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(gammas, train_scores, label = "Training Score", marker = "+")
    ax.plot(gammas, test_scores, label = "Testing Score", marker = "o")
    ax.set_title("SVR_sigmoid_gamma r = 0.01")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.)
    ax.legend(loc = "best", framealpha = 0.5)
    # 测试r，固定gamma，其中设置gamma = 10
    rs = np.linspace(0, 5)
    train_scores = []
    test_scores = []
    for r in rs:
        model = svm.SVR(kernel = "sigmoid", coef0 = r, gamma = 10)
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(rs, train_scores, label = "Training Score", marker = "+")
    ax.plot(rs, test_scores, label = "Testing Score", marker = "o")
    ax.set_title("SVR_sigmoid_r gamma = 10")
    ax.set_xlabel(r"r")
    ax.set_ylabel("score")
    ax.set_ylim(-1, 1.)
    ax.legend(loc = "best", framealpha = 0.5)
    plt.show()

def main():
    train_x, test_x, train_y, test_y = load_data_regression()
    #test_SVR_linear(train_x, test_x, train_y, test_y)
    #test_SVR_poly(train_x, test_x, train_y, test_y)
    #test_SVR_rbf(train_x, test_x, train_y, test_y)
    test_SVR_sigmoid(train_x, test_x, train_y, test_y)
if __name__ == "__main__":
    main()

