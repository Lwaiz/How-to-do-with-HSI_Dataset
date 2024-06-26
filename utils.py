# -*- coding: utf-8 -*-

from __future__ import print_function, division
import torch
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn import preprocessing

def device():
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    return device

def train(Epoch, data_loader, model, criterion, optimizer, scheduler):
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(Epoch):
        total_loss = 0.0
        correct_count = 0
        total_count = 0


        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()
            model = model.to('cuda:0')

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算当前batch的损失和准确率
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_count += (predicted == labels).sum().item()
            total_count += labels.size(0)

        # 更新学习率
        scheduler.step()

        # 计算平均损失和准确率
        avg_loss = total_loss / len(data_loader)
        accuracy = correct_count / total_count

        print(f"Epoch {epoch + 1}/{Epoch}, Avg Loss: {avg_loss}, Accuracy: {accuracy}")


# 定义测试函数
def test(data_loader, model, criterion):
    model.eval()  # 将模型设置为评估模式，这会关闭dropout等操作
    test_loss = 0.0
    correct_count = 0
    total_count = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():  # 在评估模式下不需要计算梯度
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 计算当前batch的损失和准确率
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_count += (predicted == labels).sum().item()
            total_count += labels.size(0)

    # 计算平均损失和准确率
    avg_test_loss = test_loss / len(data_loader)
    test_accuracy = correct_count / total_count

    print(f"Test Loss: {avg_test_loss}, Test Accuracy: {test_accuracy}")

"""DataSet类"""
# 加载输入
class DataSet(Dataset):
    def __init__(self, path, train, transform=None):
        if(train):
            select = "Training"
            patch_type = "train"
        else:
            select = "Testing"
            patch_type = "testing"
        self.tensors = []

        self.labels = []
        self.transform = transform
        # 迭代每类tensor 并添加patch和labels
        # 对应list
        for file in os.listdir(path):
            if(os.path.isfile(os.path.join(path, file)) and select in file):
                temp = scipy.io.loadmat(os.path.join(
                    path, file))  # 加载 mat dictionary
                # 过滤 dictionary 有下划线 "_" 的保留
                temp = {k: v for k, v in temp.items() if k[0] != '_'}

                for i in range(len(temp[patch_type+"_patches"])):
                    self.tensors.append(temp[patch_type+"_patches"][i])
                    self.labels.append(temp[patch_type+"_labels"][0][i])
        self.tensors = np.array(self.tensors)
        self.labels = np.array(self.labels)
        # print(np.shape(temp[patch_type+"_patches"]))

    def __len__(self):
        try:
            if len(self.tensors) != len(self.labels):
                raise Exception(
                    "Lengths of the tensor and labels list are not the same")
        except Exception as e:
            print(e.args[0])
        return len(self.tensors)


# 返回一个patch 和label
    def __getitem__(self, idx):
        sample = (self.tensors[idx], self.labels[idx])
       # print(self.labels)
        sample = (torch.from_numpy(self.tensors[idx]), torch.from_numpy(
            np.array(self.labels[idx])).long())
        return sample
    # 包含patch image和相应label的元组

"""数据预处理方法"""

# 标准化
def standartize(X):
    newX = np.reshape(X, (-1, 1))
    scaler = preprocessing.StandardScaler().fit(newX)
    newX = scaler.transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], X.shape[2]))
    return newX


# 预处理 主成分分析
def pca(X, k):  # k是要保留的特征数量
    newX = np.reshape(X, (-1, X.shape[0]))
    pca = PCA(n_components=k, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX.T, (k, X.shape[1], X.shape[2]))
    return newX



# 归一化
def normalized(data):
    data = data.astype(float)
    data -= np.min(data)
    data /= np.max(data)
    return data


# 填充边
def pad(X, margin):
    newX = np.zeros((X.shape[0], X.shape[1]+margin*2, X.shape[2]+margin*2))
    newX[:, margin:X.shape[1]+margin, margin:X.shape[2]+margin] = X
    return newX


def xZero(X):
    X = X.detach().cpu().numpy()
    newX = np.zeros(
        (X.shape[0], X.shape[1], X.shape[2], X.shape[3]))
    newX[:, :, int((X.shape[2]+1)/2), int((X.shape[3]+1)/2)] = X[:,
                                                                 :, int((X.shape[2]+1)/2), int((X.shape[3]+1)/2)]
    newX = torch.from_numpy(newX).float()
    if torch.cuda.is_available():
        newX = newX.cuda()
    return newX


# 生成patch，并且中心化
def patch(X, patch_size, height_index, width_index):
    # slice函数用来切片
    height_slice = slice(height_index, height_index + patch_size)
    width_slice = slice(width_index, width_index + patch_size)
    patch = X[:, height_slice, width_slice]  # patch包含所有波段
    for i in range(X.shape[0]):
        mean = np.mean(patch[i, :, :])
        patch[i] = patch[i]-mean
    return patch


def validate(net, data_loader, set_name, classes_name):
    """
    对一批数据进行预测，返回混淆矩阵以及Accuracy
    :param net:
    :param data_loader:
    :param set_name:  eg: 'valid' 'train' 'tesst
    :param classes_name:
    :return:
    """
    net.eval()
    cls_num = len(classes_name)
    conf_mat = np.zeros([cls_num, cls_num])

    for data in data_loader:
        images, labels = data
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        outputs.detach_()
        _, predicted = torch.max(outputs.data, 1)

        # 统计混淆矩阵
        for i in range(len(labels)):
            cate_i = labels[i]
            pre_i = predicted[i]
            conf_mat[cate_i, pre_i] += 1.0

    for i in range(cls_num):
        print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
            classes_name[i], np.sum(
                conf_mat[i, :]), conf_mat[i, i], conf_mat[i, i] / (1 + np.sum(conf_mat[i, :])),
            conf_mat[i, i] / (1 + np.sum(conf_mat[:, i]))))

    print('{} set Accuracy:{:.2%}'.format(
        set_name, np.trace(conf_mat) / np.sum(conf_mat)))

    return conf_mat, '{:.2}'.format(np.trace(conf_mat) / np.sum(conf_mat))


# 生成图像
def show_confMat(confusion_mat, classes, set_name, out_dir):

    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    cmap = plt.cm.get_cmap('Greys')
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(
                confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix' + set_name + '.png'))
    plt.close()

