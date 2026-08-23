# NADE-MNIST

## 项目介绍

NADE-MNIST 是一个基于 PyTorch 的生成模型项目，采用神经自回归分布估计器（Neural Autoregressive Distribution Estimator，NADE）学习图像数据的联合概率分布。
本项目支持两种 NADE 模型：
- Bernoulli NADE：面向二值离散数据，通过伯努利分布对像素进行建模与生成。
- Categorical NADE：面向多类别离散数据，基于类别分布和 Softmax 函数完成概率建模与样本生成。

除图像生成外，该方法还可进一步扩展至文本建模、序列生成等离散数据任务。

## 功能特性
- 支持二值图像和灰度图像
- 包含 MNIST、Fashion-MNIST 和 KMNIST 多种数据集
- 使用共享隐藏层实现 NADE 自回归条件分布
- 使用类别嵌入和 Softmax 实现 Categorical NADE
- 使用 NLL 和 bits per dimension 评价概率建模效果
- 生成真实图像与模型样本的左右对比图
- 固定数据划分和随机种子，便于复现实验

## 安装

环境要求：Python 3.10、3.11 或 3.12。

可以先创建虚拟环境，在安装相关依赖

```bash
conda create -n nade-mnist python=3.10 -y
conda activate nade-mnist
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 数据集

项目选择了三种尺寸一致的 28×28 灰度图像数据集，可以自由选择：

| 数据集 | 图像内容 | 示例配置 |
| --- | --- | --- |
| [MNIST](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html) | 手写数字 | `categorical_mnist.yaml` |
| [Fashion-MNIST](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html) | 服装和鞋包 | `categorical_fashion_mnist.yaml` |
| [KMNIST](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.KMNIST.html) | 日文假名字形 | `categorical_kmnist.yaml` |

MNIST 适合先验证模型；Fashion-MNIST 的灰度层次和纹理更丰富，更适合观察非二值
模型的效果；KMNIST 可以用于检验模型对不同字形分布的建模能力。

下载默认的 MNIST 数据集：

```bash
python main.py --download-data
```

下载其他配置对应的数据集：

```bash
python main.py --config configs/categorical_fashion_mnist.yaml --download-data
python main.py --config configs/categorical_kmnist.yaml --download-data
```

直接开始训练时，如果本地没有对应数据，程序也会自动下载。MNIST 文件保存在
`data/`，Fashion-MNIST 和 KMNIST 分别保存在各自的子目录。

## 使用方法

默认配置是使用 MNIST 数据集训练二值图像生成模型 Bernoulli NADE：

```bash
python main.py
```

也可以采用多类别生成，使用不同训练集训练256类的灰度图像 Categorical NADE：

```bash
python main.py --config configs/categorical_mnist.yaml
python main.py --config configs/categorical_fashion_mnist.yaml
python main.py --config configs/categorical_kmnist.yaml
```

每条命令都会依次执行：

```text
加载数据 → 离散化 → NADE 训练 → 验证 → 测试 → 图像生成
```

## 参数配置

默认的 [`configs/config.yaml`](configs/config.yaml) 使用二值模式：

```yaml
data_dir: data
dataset: mnist
representation: binary
binarization: fixed
threshold: 0.5
validation_size: 10000
hidden_dim: 128
init_std: 0.01
epochs: 20
batch_size: 128
learning_rate: 0.001
grad_clip: 5.0
seed: 42
num_workers: 0
num_samples: 64
```

类别型配置的关键参数为：

```yaml
dataset: fashion_mnist
representation: categorical
num_categories: 256
hidden_dim: 32
batch_size: 32
```

- `dataset`：`mnist`、`fashion_mnist` 或 `kmnist`
- `representation`：`binary` 使用 Bernoulli，`categorical` 使用类别分布
- `num_categories`：类别型灰度固定为256类，对应原始像素值0～255
- `hidden_dim`：共享隐藏层维度
- `num_samples`：生成图片数量

256类 Softmax 比二值模型需要更多参数和内存，因此类别型示例使用较小的
`hidden_dim` 和 `batch_size`。

## 数据处理方式

### 二值模式

二值模式首先使用 `ToTensor` 将原始 `uint8` 像素从0～255归一化到0～1，然后
执行二值化：

```text
原始 uint8 像素 0～255
→ 除以255得到浮点数 0～1
→ 固定阈值或随机二值化
→ float32 像素 0或1
→ Bernoulli NADE
```

默认的固定二值化使用 `threshold: 0.5`。`stochastic` 模式则将归一化灰度作为
Bernoulli 概率进行随机采样。

### 类别型模式

类别型模式使用 `PILToTensor` 读取原始8位灰度值，不执行归一化、区间映射或
灰度量化：

```text
原始 uint8 像素 0～255
→ 保持数值不变并转换为 long
→ 256个类别标签 0～255
→ Categorical NADE + Softmax
```

只有保存对比图时，程序才临时除以255，将类别值转换为显示所需的0～1范围；
这一操作不参与模型训练。

## 模型原理

### 自回归分解

一张 28×28 图像被展平为 $D=784$ 个离散变量。NADE 将联合概率按照像素顺序分解为一系列条件概率：

```math
p(x) = \prod_{i=0}^{D-1} p\left(x_i \mid x_{0:i}\right)
```

其中 $x_{0:i}$ 表示第 $i$ 个像素之前的所有像素。

### 二值 NADE

二值模式令 $x_i\in\{0,1\}$，使用共享隐藏状态和 Bernoulli 条件概率：

```math
h_i = \sigma\left(c + \sum_{j=0}^{i-1} W_jx_j\right)
```

```math
p\left(x_i=1 \mid x_{0:i}\right)
= \sigma\left(b_i + V_i^{\mathsf T}h_i\right)
```

### 类别型 NADE

类别型模式使用 $K=256$，令 $x_i\in\{0,1,\ldots,255\}$。每个位置和类别具有
对应的隐藏状态贡献 $E_{j,x_j}$：

```math
h_i = \sigma\left(c + \sum_{j=0}^{i-1}E_{j,x_j}\right)
```

输出层为第 $i$ 个位置生成 $K$ 个 logits，并通过 Softmax 得到类别概率：

```math
p\left(x_i=k \mid x_{0:i}\right)
= \operatorname{softmax}\left(A_i h_i+b_i\right)_k
```

```math
\sum_{k=0}^{K-1}p\left(x_i=k \mid x_{0:i}\right)=1
```

训练时完整图像已知，因此可以使用累积和并行计算所有位置的条件概率；生成时后续
像素未知，仍需按照像素顺序逐个采样。

## 项目结构

```text
nade-mnist/
├── configs/
│   ├── config.yaml                       # 二值 MNIST
│   ├── categorical_mnist.yaml            # 256类灰度 MNIST
│   ├── categorical_fashion_mnist.yaml    # 256类灰度 Fashion-MNIST
│   └── categorical_kmnist.yaml           # 256类灰度 KMNIST
├── data/
│   └── README.md                         # 数据下载说明
├── src/
│   ├── dataset.py                        # 数据下载与 DataLoader
│   ├── preprocess.py                     # 二值化和原始类别转换
│   ├── model.py                          # 二值 NADE
│   ├── categorical_model.py              # 类别型 NADE
│   ├── train.py                          # 统一训练、评估和生成流程
│   ├── visualization.py                  # 真实/生成图片对比
│   └── utils.py                          # 配置和随机种子
├── main.py
├── requirements.txt
└── README.md
```

## 输出结果

运行过程中，终端会输出：

- 训练集与验证集 NLL
- 验证集与测试集 bits/dim

NLL 和 bits/dim 越低，表示模型对 MNIST 数据分布的拟合越好。

运行结束后会生成 `comparison.png`，左侧为真实离散化图像，右侧为模型生成样本。

## 参考资料

- Hugo Larochelle and Iain Murray, [The Neural Autoregressive Distribution
  Estimator](https://proceedings.mlr.press/v15/larochelle11a.html), AISTATS 2011
