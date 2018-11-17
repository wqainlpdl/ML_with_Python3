import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn import model_selection

def load_data():
    iris = datasets.load_iris()    # 使用鸢尾花数据集
    train_X = iris.data
    train_y = iris.target
    return model_selection.train_test_split(train_X,\
        train_y, test_size = 0.25, random_state = 0,\
        stratify = train_y)

def test_LogisticRegression(*data):
    train_X, test_X, train_y, test_y = data
    model = linear_model.LogisticRegression()
    model.fit(train_X, train_y)
    print("Coefficients:%s, intercept: %s" % (model.coef_,\
            model.intercept_))
    print("Score: %.2f" % model.score(test_X, test_y))

def test_LogisticRegression_Multinomial(*data):
    train_X, test_X, train_y, test_y = data
    model = linear_model.LogisticRegression(multi_class = "multinomial",\
        solver = "lbfgs")
    model.fit(train_X, train_y)
    print("Coefficients:%s, intercept: %s" % (model.coef_,\
        model.intercept_))
    print("Score: %.2f" % model.score(test_X, test_y))

def test_LogisticRegression_C(*data):
    train_X, test_X, train_y, test_y = data
    Cs = np.logspace(-2, 4, num = 100)
    scores = []
    for C in Cs:
        model = linear_model.LogisticRegression(C = C)
        model.fit(train_X, train_y)
        scores.append(model.score(test_X, test_y))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Cs, scores)
    ax.set_xlabel(r"C")
    ax.set_ylabel(r"score")
    ax.set_xscale("log")
    ax.set_title("LogisticRegression")
    plt.show()

def main():
    train_X, test_X, train_y, test_y = load_data() # load data
    # test_LogisticRegression(train_X, test_X, train_y, test_y) # called func
    test_LogisticRegression_C(train_X, test_X, train_y, test_y)

if __name__ == "__main__":
    main()
