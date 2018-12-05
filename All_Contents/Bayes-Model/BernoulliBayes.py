from sklearn import naive_bayes
import numpy as np
import matplotlib.pyplot as plt

def test_BernoulliNB(*data):
    train_x, test_x, train_y, test_y = data
    model = naive_bayes.BernoulliNB()
    model.fit(train_x, train_y)
    print("Training Score: %.2f" % (model.score(train_x, train_y),))
    print("Testing Score: %.2f" % (model.score(test_x, test_y),))

def test_BernoulliNB_alpha(*data):
    train_x, test_x, train_y, test_y = data
    alphas = np.logspace(-2, 5, num = 200)
    train_scores = []
    test_scores = []
    for alpha in alphas:
        model = naive_bayes.BernoulliNB(alpha = alpha)
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, train_scores, label = "Training Score")
    ax.plot(alphas, test_scores, label = "Testing Score")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("BernoulliNB")
    ax.set_xscale("log")
    ax.legend(loc = "best")
    plt.show()

def test_BernoulliNB_binarize(*data):
    train_x, test_x, train_y, test_y = data
    min_x = min(np.min(train_x.ravel()), np.min(test_x.ravel())) - 0.1
    max_x = max(np.max(train_x.ravel()), np.max(test_x.ravel())) + 0.1
    binarizes = np.linspace(min_x, max_x, endpoint = True, num = 100)
    train_scores = []
    test_scores = []
    for binarize in binarizes:
        model = naive_bayes.BernoulliNB(binarize = binarize)
        model.fit(train_x, train_y)
        train_scores.append(model.score(train_x, train_y))
        test_scores.append(model.score(test_x, test_y))

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(binarizes, train_scores, label = "Training Score")
    ax.plot(binarizes, test_scores, label = "Testing Score")
    ax.set_xlabel("binarize")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_title("BernoulliNB")
    ax.legend(loc = "best")
    plt.show()

