"""
Decision Tree Model about classifier
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.tree import DecisionTreeClassifier

from sklearn import model_selection

def load_data():
   """
   加载iris数据集
   """
   iris = datasets.load_iris()
   train_X = iris.data
   train_y = iris.target
   return model_selection.train_test_split(train_X, train_y,\
        test_size = 0.25, random_state = 0,\
        stratify = train_y)

def test_DecisionTreeClassifier(*data):
    train_X, test_X, train_y, test_y = data
    model = DecisionTreeClassifier()
    model.fit(train_X, train_y)

    print("training score: %f" % (model.score(train_X, train_y)))
    print("testing score: %f" % (model.score(test_X, test_y)))

def test_DecisionTreeClassifier_criterion(*data):
    """
    测试DecisionTreeClassifier的预测性能随criterion参数的影响
    """
    train_X, test_X, train_y, test_y = data
    criterions = ["gini", "entropy"]
    for criterion in criterions:
        model = DecisionTreeClassifier(criterion = criterion)
        model.fit(train_X, train_y)
        print("criterion: %s" % criterion)
        print("Training score: %f" % (model.score(train_X, train_y)))
        print("Testing score: %f" % (model.score(test_X, test_y)))
def test_DecisionTreeClassifier_splitter(*data):
    train_X, test_X, train_y, test_y = data
    splitters = ["best", "random"]
    for splitter in splitters:
        model = DecisionTreeClassifier(splitter = splitter)
        model.fit(train_X, train_y)
        print("splitter: %s" % splitter)
        print("Training score: %f" % (model.score(train_X, train_y)))
        print("Testing score: %f" % (model.score(test_X, test_y)))

def test_DecisionTreeClassifier_depth(*data, maxdepth):
    """
    测试DecisionTreeClassifier模型的预测性能随参数max_depth的变化的
    影响
    """
    train_X, test_X, train_y, test_y = data
    depths = np.arange(1, maxdepth)
    training_scores = []
    testing_scores = []
    for depth in depths:
        model = DecisionTreeClassifier(max_depth = depth)
        model.fit(train_X, train_y)
        training_scores.append(model.score(train_X, train_y))
        testing_scores.append(model.score(test_X, test_y))
    # draw
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(depths, training_scores, label = "traing score", marker = "o")
    ax.plot(depths, testing_scores, label = "testing score", marker = "*")
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("DecisionTreeClassification")
    ax.legend(framealpha = 0.5, loc = "best")
    plt.show()

def main():
    train_X, test_X, train_y, test_y = load_data()
    #test_DecisionTreeClassifier(train_X, test_X, train_y, test_y)
    #test_DecisionTreeClassifier_criterion(train_X, test_X, train_y,\
    #    test_y)
    #test_DecisionTreeClassifier_splitter(train_X, test_X, train_y, test_y)
    test_DecisionTreeClassifier_depth(train_X, test_X, train_y,\
        test_y, maxdepth = 100)

if __name__ == "__main__":
    main()
