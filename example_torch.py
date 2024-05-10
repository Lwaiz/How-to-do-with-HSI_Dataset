import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from utils import DataSet
from utils import train, test


# 定义超参
TRAIN_EPOCH = 10
EMBED_EPOCH = 5
BATCH_SIZE = 64
classes_name = [str(c) for c in range(9)]  # 分类地物数量
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --------------------加载数据---------------------
# .mat文件路径(每个文件都是一个单独的类)
path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "DataSet", "patch_pca50")
training_dataset = DataSet(path=path, train=True)
testing_dataset = DataSet(path=path, train=False)
# Data Loaders
train_loader = torch.utils.data.DataLoader(
    dataset=training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    dataset=testing_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,

)

# 加载预训练模型
cnn_save_path = os.path.join(os.getcwd(), "model_save", 'pretrained_resnet18_10_pca50.pth')
model = torch.load(cnn_save_path).to(device)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=int(TRAIN_EPOCH / 2), gamma=0.5)  # 设置学习率下降策略

# 预训练模型的测试损失与测试精度
print("Pretrained_model")
# print("Test Loss: 0.0019592674451366712, Test Accuracy: 0.9996259060088847")
test(test_loader, model, criterion)



