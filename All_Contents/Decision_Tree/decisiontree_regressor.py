"""
DecisionTreeRgressor 模型
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor

def create_data(n):
    """
    create some smaple 
    """
    np.random.seed(0)
    X = 5 * np.random.rand(n, 1)
    y = np.sin(X).ravel()
    noise_num = (int)(n / 5)
    y[::5] += 3 * (0.5 - np.random.rand(noise_num)) #每第5个样本，就在
                # 该样本上添加噪声
    return model_selection.train_test_split(X, y, test_size = 0.25,\
        random_state = 1)

def test_DecisionTreeRegressor(*data):
    train_X, test_X, train_y, test_y = data
    model = DecisionTreeRegressor()
    model.fit(train_X, train_y)
    print("training score: %f" % (model.score(train_X, train_y)))
    print("testing score: %f" % (model.score(test_X, test_y)))
    # drawing
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    X = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    Y = model.predict(X)
    ax.scatter(train_X, train_y, label = "train sample", c = "g")
    ax.scatter(test_X, test_y, label = "test sample", c = "r")
    ax.plot(X, Y, label = "predict_value", linewidth = 2, alpha = 0.5)
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title("Decision Tree Regressor")
    ax.legend(framealpha = 0.5)
    plt.show()

def test_DecisionTreeRegressor_splitter(*data):
    train_X, test_X, train_y, test_y = data
    splitters = ["best", "random"]
    for splitter in splitters:
        model = DecisionTreeRegressor(splitter = splitter)
        model.fit(train_X, train_y)
        print("splitter: %s" % splitter)
        print("training score: %f" % (model.score(train_X, train_y)))
        print("testing score: %f" % (model.score(test_X, test_y)))

def test_DecisionTreeRegressor_depth(*data, maxdepth):
    """
    测试DecisionTreeRegressor模型的预测性能随max_depth参数
    变化的影响
    """
    train_X, test_X, train_y, test_y = data
    depths = np.arange(1, maxdepth)
    training_scores = []
    testing_scores = []
    for depth in depths:
        model = DecisionTreeRegressor(max_depth = depth)
        model.fit(train_X, train_y)
        training_scores.append(model.score(train_X, train_y))
        testing_scores.append(model.score(test_X, test_y))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(depths, training_scores, label = "traing score")
    ax.plot(depths, testing_scores, label = "testing score")
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Regressor")
    ax.legend(framealpha = 0.5)
    plt.show()

def main():
    train_X, test_X, train_y, test_y = create_data(100)
    #test_DecisionTreeRegressor(train_X, test_X, train_y, test_y)
    #test_DecisionTreeRegressor_splitter(train_X, test_X, train_y, test_y)
    test_DecisionTreeRegressor_depth(train_X, test_X, train_y,\
        test_y, maxdepth = 20)

if __name__ == "__main__":
    main()

