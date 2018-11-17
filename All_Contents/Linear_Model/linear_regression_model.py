"""
使用scikit-learn来完成linear_Regression_Model，在这里：
该线性模型属于广义的线性模型，属于最为典型的
一种情况，没有惩罚项，在这个模块里，包括两个普通函数
和一个主函数，分别来实现数据加载，模型测试，运行


"""
# import 相关的包
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn import model_selection

# 定义相关的函数
def load_data():
    '''
    加载回归问题所用到的数据集，函数的返回值
    是一个元组，元组的元素分别是训练数据集、测试数据集、
    训练数据集所对应的目标值，测试数据集所对应的目标值
    数据集应用sklearn.datasets中自带的糖尿病病人数据集
    '''
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data, diabetes.target, test_size = 0.25, random_state = 0)

def test_LinearRegression(*data):
    """
    data是一个可变参数，是一个元组类型，其分别指定了训练数据集、测试数据集、训练数据集对应的目标值和测试数据集所对应的目标值


    """
    train_X, test_X, train_y, test_y = data
    model = linear_model.LinearRegression()  # 定义模型
    model.fit(train_X, train_y)              # 模型训练
    print("Cofficients: %s, intercept: %.2f" % (model.coef_,\
        model.intercept_))       # 输出模型的相关的特征值
    model_value = model.predict(test_X)  # 模型得到的理论值
    print("Residual sum of squares: %.2f" % (np.mean((model_value -\
        test_y) ** 2)))            # 残差平方和
    print("Score: %.2f" % (model.score(test_X, test_y)))  # 模型的得分

def main():
    train_X, test_X, train_y, test_y = load_data() # 得到数据集
    test_LinearRegression(train_X, test_X, train_y, test_y)

if __name__ == "__main__":
    main()
