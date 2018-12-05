from sklearn import naive_bayes
import numpy as np
import matplotlib.pyplot as plt

def test_MultinomialNB(*data):
    train_x, test_x, train_y, test_y = data
    model = naive_bayes.MultinomialNB()
    model.fit(train_x, train_y)
    print("Traing Score: %.2f" % (model.score(train_x, train_y),))
    print("Testing Score: %.2f" % (model.score(test_x, test_y),))


def test_MultinomialNB_alpha(*data):
    train_x, test_x, train_y, test_y = data
    alphas = np.logspace(-2,5, num = 200)
    train_scores = []
    test_scores = []
    for alpha in alphas:
        model = naive_bayes.MultinomialNB(alpha = alpha)
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
    ax.set_title("MultinomoalNB")
    ax.set_xscale("log")
    plt.show()

