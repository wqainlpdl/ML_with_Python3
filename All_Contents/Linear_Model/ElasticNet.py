import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn import model_selection

def load_data():
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data,\
        diabetes.target, test_size = 0.25, random_state = 0)

def test_ElasticNet(*data):
    train_X, test_X, train_y, test_y = data
    model = linear_model.ElasticNet()
    model.fit(train_X, train_y)
    print("Cofficients: %s, intercept: %.2f" % \
        (model.coef_, model.intercept_))
    test_y_hat = model.predict(test_X)
    print("Residual sum of squares: %.2f" % \
        np.mean((test_y_hat - test_y) ** 2))
    print("score: %.2f" % model.score(test_X, test_y))

def test_ElasticNet_alpha_rho(*data):
    train_X, test_X, train_y, test_y = data
    alphas = np.logspace(-2, 2)
    rhos = np.linspace(0.01, 1)
    scores = []
    for alpha in alphas:
        for rho in rhos:
            model = linear_model.ElasticNet(alpha = alpha,\
                l1_ratio = rho)
            model.fit(train_X, train_y)
            scores.append(model.score(test_X, test_y))
    alphas, rhos = np.meshgrid(alphas, rhos)
    scores = np.array(scores).reshape(alphas.shape)
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(alphas, rhos, scores, rstride = 1,\
        cstride = 1, cmap = cm.jet, linewidth = 0,\
            antialiased = False)
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\rho$")
    ax.set_zlabel("score")
    ax.set_title("ElasticNet")
    plt.show()

def main():
    train_X, test_X, train_y, test_y = load_data()
    #test_ElasticNet(train_X, test_X, train_y, test_y)
    test_ElasticNet_alpha_rho(train_X, test_X, train_y, test_y)

if __name__ == "__main__":
    main()
