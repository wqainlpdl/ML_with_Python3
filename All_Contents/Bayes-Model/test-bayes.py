"""
naive_bayes test
"""
from sklearn import datasets, model_selection, naive_bayes
import matplotlib.pyplot as plt

from GaussianBayes import test_GaussianNB
from BernoulliBayes import test_BernoulliNB
from BernoulliBayes import test_BernoulliNB_alpha
from BernoulliBayes import test_BernoulliNB_binarize
from MultinomialBayes import test_MultinomialNB
from MultinomialBayes import test_MultinomialNB_alpha

def load_data():
    digits = datasets.load_digits()
    return model_selection.train_test_split(digits.data,\
            digits.target, test_size = 0.25, random_state = 0,\
            stratify = digits.target)

def show_digits():
    digits = datasets.load_digits()
    fig = plt.figure()
    print("vector from images 0:",digits.data[0])
    for i in range(25):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.imshow(digits.images[i], cmap = plt.cm.gray_r,\
                interpolation = "nearest")
    plt.show()


if __name__ == "__main__":
    train_x, test_x, train_y, test_y = load_data()
    # show_digits()  #用来展示数字图片
    # test_GaussianNB(train_x, test_x, train_y, test_y)
    # test_MultinomialNB(train_x, test_x, train_y, test_y)
    # test_MultinomialNB_alpha(train_x, test_x, train_y, test_y)
    # test_BernoulliNB(train_x, test_x, train_y, test_y)
    # test_BernoulliNB_alpha(train_x, test_x, train_y, test_y)
    test_BernoulliNB_binarize(train_x, test_x, train_y, test_y)
