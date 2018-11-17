import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn import model_selection


def load_data():
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data,\
        diabetes.target, test_size = 0.25, random_state = 0)

def test_Ridge(*data):
    train_X, test_X, train_y, test_y = data
    model = linear_model.Ridge()
    model.fit(train_X, train_y)
    print("Cofficients: %s, intercept: %.2f" % (model.coef_,\
        model.intercept_))
    test_y_hat = model.predict(test_X)
    print("Residual sum of square:%.2f" % np.mean((test_y_hat - test_y) ** 2))
    print("Score: %.2f" % model.score(test_X, test_y))

def test_Ridge_alpha(*data):
    train_X, test_X, train_y, test_y = data
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100,\
            200, 500, 1000]
    scores = []
    for i, alpha in enumerate(alphas):
        model = linear_model.Ridge(alpha)
        model.fit(train_X, train_y)
        scores.append(model.score(test_X, test_y))


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"score")
    ax.set_xscale("log")
    ax.set_title("Ridge")
    plt.show()
def main():
    train_X, test_X, train_y, test_y = load_data()
    # test_Ridge(train_X, test_X, train_y, test_y)
    test_Ridge_alpha(train_X, test_X, train_y, test_y)

if __name__ == "__main__":
    main()
