# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 12:50:37 2024

@author: zyq

这个代码用于简单实现HSI函数化拟合参数特征提取（代码没有精简），然后使用拟合参数实现SVM分类的方法
"""
import numpy as np
import time,skfda,os
from scipy.io import loadmat
from tqdm import tqdm
from skfda.preprocessing.smoothing import KernelSmoother
from skfda.misc.hat_matrix import ( NadarayaWatsonHatMatrix,)
import matplotlib.pyplot as plt
def achieve_data():
    #这个函数把高光谱数据转换成数组，gt为真实样本
    current_directory = os.getcwd() #相对路径，即代码和数据放在一个文件夹下
    input_data = loadmat("xuzhou.mat")['xuzhou']
    gt = loadmat("xuzhou_gt.mat")['xuzhou_gt']
    data = []
    for i in range(input_data.shape[0]):
        for p in range(input_data.shape[1]):
            data.append(input_data[i,p,:])
    data_arr=np.array(data)
    return data_arr,gt


def plot_data():
    #画个图，一个是原始光谱矢量一个是拟合的曲线
    X_basis.plot(label='Spectral curve')
    plt.plot(t,data_line,label='Original spectral vector')
    plt.legend()
    plt.show()
    
    
    
if __name__ == '__main__':
    #获取代码开始运行的时间
    #t_start = time.time()
    hsi_data , hsi_gt = achieve_data()
    data_arr_hang , data_arr_lie = hsi_data.shape
    gt_change=hsi_gt.reshape(data_arr_hang)
    n_basis=15  #n_basis是基函数数量，可调
    coeff = []
    for i in tqdm(range(data_arr_hang)):
        if gt_change[i]==0:
            coeff.append(np.full((n_basis,), -9999))
        else:
            data_line = hsi_data[i]
            t = np.linspace(0, 1, len(data_line))
            a=skfda.FDataGrid(data_line, t)
            fd_os = KernelSmoother(kernel_estimator=NadarayaWatsonHatMatrix(bandwidth=0.01),).fit_transform(a)#平滑方法可以更换，详情看官方文档
            #data_line=fd_os.data_matrix[0][:,0]
            basis = skfda.representation.basis.MonomialBasis(n_basis=n_basis)#注意基函数可以换
            X_basis = fd_os.to_basis(basis)#这个是函数型数据拟合后的对象，注意FDA不是简单的平滑+拟合
            coefficients = X_basis.coefficients.flatten()#获取每个基函数的系数
            coeff.append(coefficients)
    
            #plot_data()#注意绘图的时候最好只用一条曲线，清晰一点，可以将for循环中改成range(1)

    '''
    #下面代码用于实现SVM计算分类精度
    
    print('开始SVM分类')
    #下面开始做SVM特征分类
    t_start = time.time()
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    coeff_array = np.array(coeff)
    # 假设 coeff_array 是你的特征向量，每一行代表一个样本的基函数系数
    
    # 划分数据集为训练集和测试集，10%训练,这里用的是整个图像，但实际中图像是存在0值的，gt中的0值像素不应该参与精度分类，这个自己尝试改一下
    X_train, X_test, y_train, y_test = train_test_split(hsi_data, gt_change, test_size=0.9, random_state=42,stratify=gt_change)
    #X_train_index, X_test_index, y_train_index, y_test_index = train_test_split(np.arange(len(hsi_data)), gt_change, test_size=0.9, random_state=42,stratify=gt_change)
    
    # 创建 SVM 分类器
    svm_classifier = SVC(kernel='sigmoid')  # 也可以尝试其他核函数
    
    # 在训练集上训练 SVM 分类器
    svm_classifier.fit(X_train, y_train)
    
    # 在测试集上进行预测
    y_pred = svm_classifier.predict(X_test)
    
    # 计算分类器的准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:{}，耗时{}秒".format(accuracy,time.time()-t_start))
    
    '''