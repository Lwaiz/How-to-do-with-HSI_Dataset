模型在pretrain.py中进行预训练

输入数据为11*11的切片，pavia_U数据含有103个波段
通过data.py生成patch

# `data.py` 数据集和切片生成


这个项目用于生成地物分类的数据集，并将数据集切片用于训练和测试模型。

## 数据集说明

- 数据集：PaviaU
- 数据格式：MATLAB格式 (.mat)
- 数据集路径：`DataSet/PaviaU`
- 数据集包括：\
地物光谱数据 (`PaviaU.mat`) \
地物标签 (`PaviaU_gt.mat`)

## 切片生成

- 切片尺寸：11x11 (`PATCH_SIZE = 11`)
- 输出类别：9类地物 (`OUTPUT_CLASSES = 9`)
- 训练数据： 每一类地物100个训练样本

## 数据处理

- 数据处理步骤：
  1. 加载原始数据
  2. 对数据进行标准化处理
  3. 对数据进行填充以便于切片
  4. 生成切片数据并保存到文件夹 `DataSet/patch_train100`
  5. 切片数据包括训练集和测试集

## 使用方法

1. 运行 `loadData` 函数加载数据集
2. 对数据集进行预处理，如标准化
3. 运行 `createdData` 函数生成切片数据并保存到文件夹

## 文件说明

- `loadData` 函数：用于加载数据集
- `createdData` 函数：用于生成切片数据并保存到文件夹
- `NEW_DATA_PATH`：存放生成的切片数据的路径

# `example_torch.py`高光谱数据集加载示例


## 超参数设置

- 训练周期：10 (`TRAIN_EPOCH = 10`)
- 嵌入周期：5 (`EMBED_EPOCH = 5`)
- 批量大小：64 (`BATCH_SIZE = 64`)

## 数据加载

- 数据集路径：`DataSet/patch_pca50`
- 数据集加载：使用自定义的`DataSet`类加载数据集
- 数据加载器：`train_loader`和`test_loader`，用于训练和测试

## 模型加载与优化器设置

- 预训练模型路径：`model_save/pretrained_resnet18_10_pca50.pth`
- 模型加载：使用`torch.load`加载预训练模型
- 损失函数：交叉熵损失函数 (`nn.CrossEntropyLoss()`)
- 优化器：随机梯度下降 (`optim.SGD`)，学习率为0.01，动量为0.9
- 学习率调整器：每2个周期将学习率减半 (`torch.optim.lr_scheduler.StepLR`)

## 使用方法

1. 设置超参数
2. 加载数据
3. 加载预训练模型

## 注意事项

- 可根据需要修改超参数和模型路径
- 请确保数据集路径正确

# `pretrain.py` 分类模型训练示例

HSI分类模型预训练示例