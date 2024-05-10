# -*- coding: utf-8 -*-
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from resnet18 import RestNet18
from net import Net
from pytorch.utils import DataSet
from utils import train, test, device

# 定义预训练epoch
TRAIN_EPOCH = 10
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = RestNet18()
# model.init_weights()

model = model.float()
model = model.to(device)
print("Run with the device:", device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=int(TRAIN_EPOCH / 2), gamma=0.5)  # 设置学习率下降策略

# --------------------加载数据---------------------
# .mat文件路径(每个文件都是一个单独的类)   patch1是8/2分   patch是5/5分
path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "DataSet", "patch_pca50")
training_dataset = DataSet(path=path, train=True)
testing_dataset = DataSet(path=path, train=False)

train_loader = DataLoader(
    dataset=training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # num_workers=4
)

test_loader = DataLoader(
    dataset=testing_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # num_workers=4
)

# Pretrain the model
print("Pretrain the model")
train(TRAIN_EPOCH, train_loader, model, criterion, optimizer, scheduler)

# 保存模型
cnn_save_path = os.path.join(os.getcwd(), "model_save", 'pretrained_resnet18_10_pca50.pth')
torch.save(model, cnn_save_path)

print("Test the model")
test(test_loader, model, criterion)
