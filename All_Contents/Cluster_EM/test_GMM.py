"""
本模块用来测试高斯混合模型的用法
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs
from GMM import *       # 从本文件夹中的GMM模块中导入所有成员

# 创建测试GMM算法的数据
def create_data(centers, num = 100, std = 0.5):
    x, labels_true = make_blobs(n_samples = num, centers = centers,\
            cluster_std = std)
    return x, labels_true

def main():
    centers = [[1, 1], [2, 2], [1, 2], [10, 20]]
    x, labels_true = create_data(centers, 1000, 0.5)
    #test_GMM(x, labels_true)
    #test_GMM_n_components(x, labels_true)
    test_GMM_cov_type(x, labels_true)



if __name__ == "__main__":
    main()

