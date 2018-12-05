from sklearn import naive_bayes

def test_GaussianNB(*data):
    train_x, test_x, train_y, test_y = data
    model = naive_bayes.GaussianNB()
    model.fit(train_x, train_y)
    print("Training Score: %.2f" % (model.score(train_x, train_y),))
    print("Testing Score: %.2f" % (model.score(test_x, test_y)))
