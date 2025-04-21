---
title: 使用Pytorch搭建深度学习模型的基本框架——以COVID-19 Cases Prediction为例
tags:
    - Machine Learning
    - Programming
---

## toc

&emsp;&emsp;自2012年Alexnet在Imagenet图像分类竞赛中一鸣惊人，以神经网络算法为主体的深度学习技术在人工智能领域兴起，而后诸如卷积神经网络(CNN)，残差网络(Resnet)，生成式对抗网络(GAN)，自注意力模型(Transformer)等众多性能强大的算法模型被提出，使得人工智能领域的研究与应用进入了一个蓬勃发展的阶段。2016年，DeepMind推出围棋人工智能AlphaGo，其以4:1战胜围棋世界冠军李世石，让世人认识到了深度学习技术的强大。2023年，基于Transformer的LLM模型ChatGPT横空出世，这场来自AI领域的技术革命第一次距离我们如此之近。无人知晓深度学习未来能给人类世界带来多大的变革，它能否最终实现强人工智能，带领我们进入科幻电影中的世界，但至少目前，深度学习仍牢牢占据学术与工业界研究的主流。  
&emsp;&emsp;在众多用于实现深度学习技术的框架中，Pytorch与TensorFlow目前被使用得最为广泛，而Pyotrch以其简洁的语法与强大的生态颇受学者的青睐，成为目前学术研究最流行的深度学习框架，本节会以COVID-19 Cases为例，使用Pytorch搭建一个深度学习模型。这是一个非常简单的回归任务，在完成这个任务的过程中，我们将会了解使用Pytorch搭建深度学习模型的流程、各种函数的作用以及训练模型的步骤。  

## Task Description  

- **Objectives**

  - Solve a regression problem with deep neural networks(DNN)  
  - Understand basic DNN training steps.
  - Get familiar with PyTorch.  
  
- **Task**  
&emsp;&emsp;**COVID-19 Cases Prediction:** Given survey results in the past 5 days in a specific states in U.S., then predict the percentage of new tested positive cases in the 5th day.  

- **Data**  
  - Training Data: 2699 samples  
  - Testing Data: 1078 samples  
  - Feature Infactors(117):  
    - States(37, encoded to one-hot vector)
    - COVID-like illness(4*5)
      - cli、ili ...
    - Behavior Indicators(8*5)
      - wearing_mask、travel_outside_state ...
    - Mental Health Indicators(3*5)
      - anxious、depressed ...
    - Tested Positive Cases(1*5)
      - tested_positive(**this is what we want to predict**)   


## Download Data  

&emsp;&emsp;本案例的数据集来自于Kaggle，可以访问以下网站下载数据集.  

- Data Source: https://www.kaggle.com/competitions/ml2022spring-hw1/overview  

## Import Packages  
&emsp;&emsp;在开始任务前首先导入我吗需要使用到的Python库 

```python
# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter
```

&emsp;&emsp;一些库的主要功能：  

- `tqdm`: 用于在循环中显示进度条，以增强用户对程序运行进度的可视化体验。主要功能有**进度条显示、时间估算、定制输出、支持多种数据结构、并发安全.**  
- `torch.nn`: 用于构建神经网络模型。它提供了各种用于构建神经网络层、损失函数、优化器等的类和函数，使用户能够方便地创建、训练和部署各种类型的神经网络模型。主要功能有**神经网络层的构建、损失函数的定义、优化器、自定义模型、数据转换层、模型的保存和加载.**
- `torch.utils.data`: 用于处理和管理数据加载、预处理以及批量处理等任务。它提供了一组工具和类，帮助用户有效地加载、处理和传输数据到神经网络模型中，从而方便地进行训练、验证和测试。主要功能有**数据加载与管理、数据预处理、数据批处理、并行加载、迭代加载.**  
- `torch.utils.tensorboard`: 用于在训练过程中可视化模型训练和性能指标，可以帮助深度学习研究人员和工程师实时监视、分析和优化他们的模型训练过程。主要功能有**训练过程的可视化、模型结构的可视化、嵌入向量的可视化、分布的可视化、图像和音频的可视化.**  

## Set Random Seed  

&emsp;&emsp;由于深度学习的训练过程包含一定的随机性，例如网络参数初始化、随机梯度下降、Dropout等，在学术研究中为了使得结果可以复现，我们通常需要事先设置随机数种子以固定模型的随机性，其代码如下:  

```python
def same_seed(seed): 
    # Fixes random number generator seeds for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
```
&emsp;&emsp;一些主要函数的功能：  

- `torch.bankends.cudnn.deterministic`: 用于控制在使用CUDA进行深度学习训练时是否启用确定性计算。在深度学习中，训练过程中使用CUDA可以显著加速计算，但由于浮点数的不精确性和优化算法的随机性，相同的代码在不同的运行环境中可能会产生不同的结果。但需要注意的是，启用确定性计算会带来一定的计算效率损失，因为一些优化策略可能会被禁用，从而降低了性能。
- `torch.backends.cudnn.benchmark`: 用于控制CuDNN（CUDA Deep Neural Network library，NVIDIA深度神经网络库）在使用CUDA加速时是否自动寻找最适合当前硬件环境的优化算法。当设置为True时，将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速，然而，不同的卷积算法可能在计算精度和数值稳定性方面有微小差异，这可能导致每次前向传播的结果略微不同。当设置为False时，CuDNN不再搜寻最佳算法，而是选择一个固定的卷积算法。这个固定的算法在相同的输入数据和参数情况下产生相同的输出，因为它不受运行时的微小变化影响。  
- `torch.manual_seed()`: 用于为CPU设置随机数生成器的种子，从而控制生成的随机数序列。
- `torch.cuda.is_available()`: 用于检查当前系统是否支持CUDA，以及是否安装了可用的GPU设备。
- `torch.cuda.manual_seed()`: 为GPU设置随机数种子，从而控制生成的随机数序列。
- `torch.cuda.manual_seed_all()`: 如果使用的是多GPU模型，其可以设置用于在所有GPU上生成随机数的种子。  

## Dataset  

&emsp;&emsp;在PyTorch中，我们需要将原始数据集定义为一个Dataset实例，定义Dataset实例的作用是将数据加载和预处理封装成一个可迭代的对象，以便在训练过程中有效地加载和使用数据。其代码如下：  

```python
class COVID19Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
```

&emsp;&emsp;Dataset类中一些函数的功能：  

- `__init__()`: Dataset类的构造函数，用于初始化数据集的属性和参数。
- `__getitem__()`: 用于根据索引获取数据集中的一个样本。在训练过程中，DataLoader会通过迭代访问数据集，调用__getitem__来获取样本。
- `__len__()`: 返回数据集中样本的数量，通常通过获取数据的长度来实现。  

## Feature Selection  
&emsp;&emsp;原始数据集包含众多的特征，但如果将所有的特征作为训练数据，可能会造成训练时间过长，模型过拟合等问题。实际上特征之间往往存在多重共线性，对于这个任务，我们并不需要数量非常庞大的特征，这时我们需要进行特征工程方面的工作，这里不展开解释特征工程的方法，有兴趣可以自行查阅相关资料。总得来说，当我们的数据集包含众多特征时，我们需要保有进行特征工程的意识。在本案例为了简单起见，省略掉了特征筛选的过程，以人为设置的特征作为筛选结果。  
```python
def select_feat(train_data, valid_data, test_data, select_all=True):
    '''Selects useful features to perform regression'''
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0,1,2,3,4] # TODO: Select suitable feature columns.
        
    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid
```

## Dataloader  

&emsp;&emsp;在定义了Dataset实例后，我们通常需要将Dataset实例传递给Dataloader类。Dataloader是一个迭代器，用于加载和处理训练数据，其可以将数据集划分成小批量的数据，并可以自动进行数据预处理、洗牌和GPU加速等操作。在定义Dataloader时，我们需要确定Training Data、Testing Data、代码如下：  
```python
# Set seed for reproducibility
same_seed(config['seed'])

def train_valid_split(data_set, valid_ratio, seed):
    #Split provided training data into training set and validation set
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

# train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days) 
# test_data size: 1078 x 117 (without last day's positive rate)
train_data, test_data = pd.read_csv('./covid.train.csv').values, pd.read_csv('./covid.test.csv').values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

# Print out the data size.
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# Select features
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

# Print out the number of features.
print(f'number of features: {x_train.shape[1]}')

train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train),COVID19Dataset(x_valid, y_valid), COVID19Dataset(x_test)

# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)  
```

&emsp;&emsp;在这段代码中，我们首先通常前文定义的`same_seed()`函数固定随机数种子，使得之后的一系列操作的结果可复现。然后我们定义了`train_valid_split()`函数，这个函数用于将Training Data进行再次划分，分为真正用于训练的数据与用于检验模型泛化能力的数据，通过`valid_ratio`参数，我们可以设置train data和valid data的划分比例。定义好函数后，我们读取原始csv文件，得到原始的Training Data和Testing Data，利用`train_valid_split()`函数得到划分好的train data与valid data。此时，这些数据集仍包含着所有的特征，我们需要通过前文定义的`select_feat()`函数进行特征筛选，得到正在用于本次任务的数据。最后我们使用这些数据定义Dataset实例，并将定义好的Dataset传递给`Dataloader`用于之后训练模型。  

&emsp;&emsp;`Dataloader`的一些主要参数及其作用：  

- `dataset`: 指定要加载的Dataset实例，这是DataLoader的必需参数，用于提供数据样本。
- `batch_size`: 指定每个批次中包含的样本数量。将数据划分成小批次可以在训练时提高内存的使用效率。
- `shuffle`: 设置为True时，在每次返回一批次前会对数据进行随机洗牌。这有助于提高模型的泛化能力。默认值为False。
- `pin_memory`：如果在GPU上训练，可以将此参数设置为True，以便在加载数据时将数据置于CUDA固定内存中，从而加速数据传输。
- `drop_last`：如果数据样本数量不能整除批次大小，设置为True时，会丢弃最后一个不完整的批次。默认值为False。
- `num_workers`：指定用于数据加载的并行线程数。通过并行加载数据可以加快速度，特别是在数据集较大时。  

## Neural Network Model  

&emsp;&emsp;准备好数据后，我们便需要构建本次任务所用的深度学习模型，这里我构建了一个简单的三层前馈神经网络模型，这个神经网络包含一个输入层，一个隐藏层，一个输出层，其代码如下:  
```python
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x
```

&emsp;&emsp;一些函数及类的主要功能：  

- `nn.Module`: 提供了一种组织和管理神经网络组件的方式，使得模型的构建、参数管理和前向传播等过程更加简洁和可控。我们在实际应用中定义的模型`My_Model`需要继承`nn.Module`类。  
- `nn.Sequrntial()`: PyTorch中的一个模型容器，用于按顺序组合多个神经网络模块，从而构建一个序列式的神经网络模型。它可以在不需要自定义模型类的情况下，方便地定义简单的神经网络结构。  
- `nn.Linear()`: 用于定义线性变换（全连接层）。它将输入数据与权重矩阵相乘，并添加一个偏置，从而实现线性变换。nn.Linear()主要用于神经网络中的全连接层，将输入特征映射到输出特征。
&emsp;&emsp;除了使用`nn.Linear()`进行线性神经网络层，我们还可以根据任务的不同使用诸如卷积神经网络层`nn.convXd`、循环神经网络层`nn.RNN()`等。  

- `nn.ReLU()`: 用于实现激活函数 ReLU（Rectified Linear Activation）。ReLU 是深度学习中常用的激活函数之一，它对输入进行非线性变换，将负值变为零，保持正值不变。ReLU 激活函数在神经网络中引入非线性性质，有助于模型学习复杂的特征和表示。  
&emsp;&emsp;除了ReLU之外，还有一些常用的激活函数，例如`nn.Sigmoid()`(Sigmoid函数)、`nn.Tanh()`(双曲正切激活函数)。我们可以根据训练数据的特点，选择合适的激活函数，或者通过实验确定最佳激活函数。  
- `forward()`: 用于将训练数据通过网络进行前向传播。  

## Training Loop  

&emsp;&emsp;在准备好数据与模型后，下一步就是训练模型，训练模型的主要流程如下：  
```markdown
设置损失函数
定义优化器
定义tensorboard
总训练轮次、当前训练轮次、当前最佳误差、无效训练次数等参数初始化

for 每一训练轮次 in 总训练轮次：
    设置模型为训练模式
    定义一个空的训练误差列表
    将训练数据传递给tqdm
    for 每一批次特征数据、标签数据 in tqdm(训练数据):
        初始化优化器梯度值
        将数据移动到GPU中
        将特征数据输入模型，通过正向传播得到标签数据的预测值
        通过标签数据的预测值与真实值得到误差
        误差反向传播，得到误差对于网络参数的梯度
        利用梯度下降算法更新网络参数
        step += 1
        将本批次的误差添加到训练误差列表中
        自定义训练进度条内容
    通过对训练误差列表求平均得到这一轮次的平均训练误差
    将setp与平均训练误差添加到tensorboard中
    将模型设置为评估模式
    定义一个空的评估误差列表
    for 每一批次特征数据、标签数据 in 评估数据:
        将数据移动到GPU中
        with 初始化优化器梯度值:
            将特征数据输入当前模型，通过正向传播得到标签数据的预测值
            通过标签数据的预测值与真实值得到误差
        将本批次的误差添加到评估误差列表
    通过对评估误差列表求平均得到这一轮次的平均评估误差
    print(当前训练轮次/平均训练误差/平均评估误差)
    将setp与平均评估误差添加到tensorboard中
    if 平均评估误差 < 当前最佳误差:
        将当前最佳误差设置为平均评估误差
        保存当前模型
        初始化无效训练次数
    else:
        无效训练次数 += 1
    if 无效训练次数 > 所设置的无效训练次数上限:
        停止训练
```

&emsp;&emsp;这一流程的代码如下：  

```python
def trainer(train_loader, valid_loader, model, config, device):

    criterion = nn.MSELoss(reduction='mean') # Define your loss function, do not modify this.

    # Define your optimization algorithm. 
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9) 

    writer = SummaryWriter() # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models') # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            x, y = x.to(device), y.to(device)   # Move your data to device. 
            pred = model(x)             
            loss = criterion(pred, y)
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())
            
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval() # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return
```

&emsp;&emsp;训练流程中一些函数的作用及参数:  

- `nn.MSELoss()`: PyTorch 中的一个损失函数模块，用于计算均方误差损失。
&emsp;&emsp;常用的一些损失函数: 交叉熵损失`nn.CrossEntropyLoss()`，负对数似然损失`nn.NLLLoss()`，KL散度损失`nn.KLDivLoss()`等，可以更具任务特点与所用模型选择合适的损失函数，例如均方误差损失多用于回归任务，交叉熵损失多用于分类任务，KL散度损失一般用于GAN模型。  
- `torch.optim.SGD()`: 用于实现随机梯度下降（Stochastic Gradient Descent，SGD）优化算法。SGD 是深度学习中最基本和常用的优化算法之一，用于调整模型的参数以最小化损失函数。其重要参数: 
  - `params`：这是一个模型参数的可迭代对象，指定了需要进行优化的参数。一般通过`model.parameters()`来获取模型中的参数列表。
  - `lr`：学习率，控制参数更新的步长。它决定了每次参数更新的幅度，过大可能导致不稳定的训练，过小可能导致收敛速度缓慢。  
  - `momentum`: 动量，用于加速梯度下降过程。设置一个介于 0 到 1 之间的值，代表在更新参数时考虑前一次的动量。较大的动量值可以帮助跳出局部最小值。  
  
&emsp;&emsp;除开基础的SGD优化方法，深度学习中还有Adam`torch.optim.Adam()`，RMSprop`torch.optim.RMSprop()`，Adagrad`torch.optim.Adagrad()`等优化算法，它们各种适应不同的数据特点。想进一步了解深度学习中的优化算法可以查阅相关资料。  

&emsp;&emsp;完成模型训练的流程后，下一步便是设置训练步骤中所需要的超参数。

## Configurations  

&emsp;&emsp;设置我们在整个任务中所需要用到的参数：  
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,      # random seed
    'select_all': True,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 3000,     # Number of epochs.            
    'batch_size': 256, 
    'learning_rate': 1e-5,              
    'early_stop': 400,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}
```

## Start training!  

&emsp;&emsp;万事俱备，开始训练我们的模型吧！  
```python
model = My_Model(input_dim=x_train.shape[1]).to(device) # put your model and data on the same computation device.
trainer(train_loader, valid_loader, model, config, device)
```

&emsp;&emsp;由于深度学习模型的参数量一般较大，训练可能会花费一定的时间，待训练完成后我们便得到了已更新好参数的神经网络模型。  

## Plot learning curves with tensorboard  
&emsp;&emsp;通过使用`tensorboard`，我们可以得到损失曲线与学习曲线，便于我们更好地理解模型训练的过程。  
```python
%reload_ext tensorboard
%tensorboard --logdir=./runs/
```

## Testing  
&emsp;&emsp;最后，我们可以将Testing Data输入到已经更新好参数的模型，得到相应标签数据的预测值，我们可以通过比较真实值与预测值之间的差异评价模型训练的结果。  
```python
def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds

def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device) 
save_pred(preds, 'pred.csv')      
```

&emsp;&emsp;预测的结果保存在`pred.csv`文件中。

## Reference  

https://www.bilibili.com/video/BV1Wv411h7kN?p=11&vd_source=234cf2ac075a1558881a6956450ddf89

