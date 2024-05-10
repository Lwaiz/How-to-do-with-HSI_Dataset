# -*- coding: utf-8 -*-
import os
import scipy.io
import scipy.ndimage
import numpy as np
from random import shuffle
from utils import pca, pad, standartize, patch


# 定义全局变量
PATCH_SIZE = 11  # 切片尺寸为11*11
OUTPUT_CLASSES = 9  # 输出9类地物
TEST_FRAC = 0.20  # 用来测试数据的百分比
NEW_DATA_PATH = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "DataSet", "patch_train100")  # 存放数据路径 patch是文件夹名称


# 加载数据
def loadData(flieName, dataName, labelName):
    # 原始数据路径
    DATA_PATH = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "DataSet", flieName)

    data = scipy.io.loadmat(os.path.join(DATA_PATH, dataName))
    data = data[list(data.keys())[-1]]
    label = scipy.io.loadmat(os.path.join(DATA_PATH, labelName))
    label = np.int32(label[list(label.keys())[-1]])
    data = np.transpose(data, (2, 0, 1))  # 将通道数提前，便于数组处理操作
    return data, label


# 生成切片数据并存储
def createdData(X, label):
    for c in range(OUTPUT_CLASSES):
        PATCH, LABEL, TEST_PATCH, TRAIN_PATCH, TEST_LABEL, TRAIN_LABEL = [], [], [], [], [], []
        for h in range(X.shape[1]-PATCH_SIZE+1):
            for w in range(X.shape[2]-PATCH_SIZE+1):
                gt = label[h, w]
                if(gt == c+1):
                    img = patch(X, PATCH_SIZE, h, w)
                    PATCH.append(img)
                    LABEL.append(gt-1)
        # 打乱切片
        shuffle(PATCH)
        # 划分测试集与训练集
        # split_size = int(len(PATCH)*TEST_FRAC)
        split_size = 100
        TRAIN_PATCH.extend(PATCH[:split_size])  # 0 ~ split_size
        TEST_PATCH.extend(PATCH[split_size:])  # split_size ~ len(class)
        TRAIN_LABEL.extend(LABEL[:split_size])
        TEST_LABEL.extend(LABEL[split_size:])
        # 写入文件夹
        train_dict, test_dict = {}, {}
        train_dict["train_patches"] = TRAIN_PATCH
        train_dict["train_labels"] = TRAIN_LABEL
        file_name = "Training_class(%d).mat" % c
        if not os.path.exists(NEW_DATA_PATH):
            os.makedirs(NEW_DATA_PATH)
        scipy.io.savemat(os.path.join(NEW_DATA_PATH, file_name), train_dict)
        test_dict["testing_patches"] = TEST_PATCH
        test_dict["testing_labels"] = TEST_LABEL
        file_name = "Testing_class(%d).mat" % c
        scipy.io.savemat(os.path.join(NEW_DATA_PATH, file_name), test_dict)


data, label = loadData("PaviaU", "PaviaU.mat", "PaviaU_gt.mat")
# data = pca(data, 50)
data = standartize(data)

data = pad(data, int((PATCH_SIZE-1)/2))
createdData(data, label)


