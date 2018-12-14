import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition

def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target

def test_KPCA(*data):
    x, y = data
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    for kernel in kernels:
        kpca = decomposition.KernelPCA(n_components=None,\
                kernel=kernel)  #依次测试4中核函数
        kpca.fit(x)
        print("kernel=%s-->lambda:%s" % (kernel, kpca.lambdas_))

def plot_KPCA(*data):
    """
    绘制经过核PCA将原始数据降到二维后的样本点
    """
    x, y = data
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    fig = plt.figure()
    colors = ((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0.5,0,0.5),\
            (0,0.5,0.5),(0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),\
            (0.5,0.3,0.2))     #颜色集合，不同标记的样本染不同的色
    for i,kernel in enumerate(kernels):
        kpca = decomposition.kernelPCA(n_components=2,kernel=kernel)
        kpca.fit(x)
        x_r = kpca.transform(x) #将原始数据降到二维
        ax = fig.add_subplot(2,2,i+1) #两行两列，每个单元显示一种核函数降维
                                      # 后的效果图
        for label, color in zip(np.unique(y), colors):
            position=y==label
            ax.scatter(x_r[position,0],x_r[position,1],\
                    label="target=%d" % label,color=color)
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.legend(loc="best")
        ax.set_title("kernel=%s" % kernel)
    plt.suptitle("KPCA")
    plt.show()

def plot_KPCA_poly(*data):
    """
    绘制经过使用poly核的kernelPCA降维到二维后的样本点
    """
    x,y = data
    fig = plt.figure()
    colors = ((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0.5,0,0.5),\
            (0,0.5,0.5),(0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),\
            (0.5,0.3,0.2),)     # 颜色集合，不同标记的样本染不同的色
    """
    注意下面的poly核参数构成的列表
    p取得值为:3,10
    gamma取值为:1,10
    r取值为1,10
    所以，排列组合一共为8种组合
    注意多项式核函数为K(x,z)=(gamma*(x.dot(z)+1)+r)**p
    """
    Params = [(3,1,1),(3,10,1),(3,1,10),(3,10,10),(10,1,1),\
            (10,10,1),(10,1,10),(10,10,10)]  #poly核的参数组成的列表
    for i,(p,gamma,r) in enumerate(Params):
        kpca = decomposition.KernelPCA(n_components=2,kernel="poly",\
                gamma=gamma,degree=p,coef0=r) #poly核，目标是2维
        kpca.fit(x)
        x_r = kpca.transform(x) # 原始数据集转换到二维
        ax = fig.add_subplot(2,4,i+1)  #两行四列，每个单元显示核函数为
                               #poly的kernelPCA一组参数的效果图
        for label,color in zip(np.unique(y),colors):
            position=y==label
            ax.scatter(x_r[position,0],x_r[position,1],\
                    label = "target=%d" % label,color=color)
        ax.set_xlabel("x[0]")
        ax.set_xticks([])   #隐藏x轴的刻度
        ax.set_yticks([])   #隐藏y轴的刻度
        ax.set_ylabel("x[1]")
        ax.legend(loc="best")
        ax.set_title(r"$(%s(x \cdot z+1)+%s)^{%s}$"%(gamma,r,p))
    plt.suptitle("KPCA-Poly")
    plt.show()


def plot_KPCA_rbf(*data):
    """
    绘制经过使用rbf核的kernelPCA降维到二维之后的样本点
    """
    x,y = data
    fig = plt.figure()
    colors = ((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0.5,0,0.5),\
            (0,0.5,0.5),(0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),\
            (0.5,0.3,0.2),)
    Gammas=[0.5,1,4,10] #rbf核的参数组成的列表，每个参数就是gamma值
    for i,gamma in enumerate(Gammas):
        kpca = decomposition.KernelPCA(n_components=2,kernel="rbf",\
                gamma=gamma)
        kpca.fit(x)
        x_r = kpca.transform(x)  #将原始数据降到二维
        ax = fig.add_subplot(2,2,i+1)
        for label,color in zip(np.unique(y),colors):
            position=y==label
            ax.scatter(x_r[position,0],x_r[position,1],\
                    label="target=%d" % label,color=color)
        ax.set_xlabel("x[0]")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("x[1]")
        ax.legend(loc="best")
        ax.set_title(r"$\exp(-%s||x-z||^2)$"%gamma)
    plt.suptitle("KPCA-rbf")
    plt.show()

def plot_KPCA_sigmoid(*data):
    """
    绘制经过使用sigmoid核的KernelPCA降维到二维之后的样本点
    """
    x,y = data
    fig = plt.figure()
    colors = ((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0.5,0,0.5),\
            (0,0.5,0.5),(0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),\
            (0.5,0.3,0.2),)
    """
    下面的Params列表中的每一个元素是sigmoid核参数的排列组合
    gamma的取值为:0.01,0.1,0.2
    coef0的取值为:0.1,0.2
    因此，排列组合数为6
    """
    Params=[(0.01,0.1),(0.01,0.2),(0.1,0.1),(0.1,0.2),(0.2,0.1),\
            (0.2,0.2)]
    for i,(gamma,r) in enumerate(Params):
        kpca = decomposition.KernelPCA(n_components=2,kernel="sigmoid",\
                gamma=gamma,coef0=r)
        kpca.fit(x)
        x_r = kpca.transform(x)
        ax = fig.add_subplot(3,2,i+1)
        for label,color in zip(np.unique(y),colors):
            position=y==label
            ax.scatter(x_r[position,0],x_r[position,1],\
                    label="target=%d" % label,color=color)
        ax.set_xlabel("x[0]")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("x[1]")
        ax.legend(loc="best")
        ax.set_title(r"$\tanh(%s(x\cdot z)+%s)$"%(gamma,r))
    plt.suptitle("KPCA-sigmoid")
    plt.show()


if __name__=="__main__":
    x,y = load_data()
    #test_KPCA(x,y)
    #plot_KPCA(x,y)
    #plot_KPCA_poly(x,y)
    #plot_KPCA_rbf(x,y)
    plot_KPCA_sigmoid(x,y)










                                                                                                     

