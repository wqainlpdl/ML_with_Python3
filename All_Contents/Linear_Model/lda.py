'''
此处的lda为线性判别分析，是一种监督学习算法，其核心思想为：
 在训练时：设法将训练样本投影到一条直线上，使得同类样本的
 的投影点尽可能地接近、异类样本的投影点尽可能远离。
 要学习的就是这样一条直线
 在预测时：将带预测样本投影到学到的直线上，根据它的投影点
 的位置来判定它的类别
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import discriminant_analysis
from sklearn import model_selection

def load_data():
    iris = datasets.load_iris()
    train_X = iris.data
    train_y = iris.target
    return model_selection.train_test_split(train_X,\
        train_y, test_size = 0.25, random_state = 0,\
        stratify = train_y)

def test_LinearDiscriminantAnalysis(*data):
    train_X, test_X, train_y, test_y = data
    model = discriminant_analysis.LinearDiscriminantAnalysis()
    model.fit(train_X,train_y)
    print("Cofficients: %s, intercept: %s" % \
        (model.coef_, model.intercept_))
    print("Score: %.2f" % model.score(test_X, test_y))

def plot_LDA(converted_X, y):
    """
    绘制经过LDA转换后的数据
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = 'rgb'
    markers = 'o*s'
    for target, color, marker in zip([0, 1, 2], colors, markers):
        pos = (y == target).ravel()
        X = converted_X[pos, :]
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color = color,\
            marker = marker, label = "Label %d" % target)
    ax.legend(loc = "best")
    fig.suptitle("Iris After LDA")
    plt.show()

def run_plot_LDA():
    train_X, test_X, train_y, test_y = load_data()
    X = np.vstack((train_X, test_X))
    Y = np.vstack((train_y.reshape(train_y.size, 1),\
        test_y.reshape(test_y.size, 1)))
    model = discriminant_analysis.LinearDiscriminantAnalysis()
    model.fit(X, Y)
    converted_X = np.dot(X, np.transpose(model.coef_)) + model.intercept_
    plot_LDA(converted_X, Y)

def test_LinearDiscriminantAnalysis_solver(*data):
    """
    测试LDA模型的预测性能随参数solver选择的影响
    """
    train_X, test_X, train_y, test_y = data
    solvers = ["svd", "lsqr", "eigen"]
    for solver in solvers:
        if solver == "svd":
            model = discriminant_analysis.LinearDiscriminantAnalysis(\
                solver = solver)
        else:
            model = discriminant_analysis.LinearDiscriminantAnalysis(\
                solver = solver, shrinkage = None)
        model.fit(train_X, train_y)
        print("Score at solver = %s: %.2f" % \
            (solver, model.score(test_X, test_y)))

def test_LinearDiscriminantAnalysis_shrinkage(*data):
    """
    测试LDA模型的预测性能随shrinkage参数变化的影响
    """
    train_X, test_X, train_y, test_y = data
    shrinkages = np.linspace(0.0, 1.0, num = 20)
    scores = []
    for shrinkage in shrinkages:
        model = discriminant_analysis.LinearDiscriminantAnalysis(\
            solver = "lsqr", shrinkage = shrinkage)
        model.fit(train_X, train_y)
        scores.append(model.score(test_X, test_y))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(shrinkages, scores)
    ax.set_xlabel(r"shrinkage")
    ax.set_ylabel(r"score")
    ax.set_ylim(0, 1.05)
    ax.set_title("LinearDiscriminantAnalysis")
    plt.show()

def main():
    train_X, test_X, train_y, test_y = load_data()
    #test_LinearDiscriminantAnalysis(train_X, test_X, train_y, test_y)
    #run_plot_LDA()
    #test_LinearDiscriminantAnalysis_solver(train_X, test_X, train_y,\
    #    test_y)
    test_LinearDiscriminantAnalysis_shrinkage(train_X, test_X,\
        train_y, test_y)


if __name__ == "__main__":
    main()
