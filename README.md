# NADE

## 项目介绍

NADE 是一个基于 PyTorch 的自回归生成模型项目，采用神经自回归分布估计器（Neural Autoregressive Distribution Estimator，NADE）学习数据的联合概率分布并生成新的样本。本项目支持三种 NADE 模型：

- Bernoulli NADE：面向二值离散数据，基于伯努利分布和 sigmoid 函数进行概率建模与样本生成。
- Categorical NADE：面向多类别离散数据，基于类别分布和 Softmax 函数进行概率建模与样本生成。
- RNADE：面向连续数据，使用等权高斯混合分布对每个条件概率密度进行建模与采样。


本项目中 Bernoulli NADE 和 Categorical NADE 用于图像生成，RNADE 用于连续表格数据的密度估计。相同的自回归分解方法还可以扩展到文本、序列及其他多维数据。


## 功能特性

- 支持各种数据类型：二值数据、多类别数据以及连续数据
- 分别使用 MNIST、Fashion-MNIST 和 UCI White Wine 数据集
- 使用共享隐藏层实现 NADE 自回归条件分布
- 基于 Bernoulli 分布和 sigmoid 实现 Bernoulli NADE
- 基于 Categorical 分布和 Softmax 实现 Categorical NADE
- 基于 Gaussian Mixture 和 Softplus 实现连续型 RNADE
- 使用负对数似然（NLL）和 bits per dimension 评价概率建模效果
- 图像模型会生成真实图像与模型样本的左右对比图
- RNADE 输出连续生成样本和逐特征统计结果
- 固定数据划分和随机种子，便于重复实验

## 安装

环境要求：Python 3.10+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活虚拟环境。


## 数据集

项目为每种 NADE 选择一种适合的数据集：

| 数据集 | 数据内容 | 示例配置 |
| --- | --- | --- |
| [MNIST](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html) | 手写数字图像 | `bernoulli.yaml` |
| [Fashion-MNIST](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html) | 服装和鞋包图像 | `categorical.yaml` |
| [UCI White Wine](https://archive.ics.uci.edu/dataset/186/wine+quality) | 11 个连续理化特征 | `continuous.yaml` |

MNIST 的笔画和背景适合 Bernoulli NADE 的二值建模；Fashion-MNIST 的灰度层次和纹理更丰富，适合观察 Categorical NADE 的非二值生成效果；UCI White Wine 是 RNADE 原论文使用的连续数据集，模型使用其中 11 个理化特征，并排除离散的 `quality` 标签。

根据需要指定并下载对应数据集：

```bash
python main.py --config configs/bernoulli.yaml --download-data
python main.py --config configs/categorical.yaml --download-data
python main.py --config configs/continuous.yaml --download-data
```

直接开始训练时，如果本地没有对应数据，程序也会自动下载。各数据集会分别保存在 `data/` 下对应的子目录中。

## 使用方法

使用 MNIST 训练二值图像生成模型 Bernoulli NADE：

```bash
python main.py --config configs/bernoulli.yaml
```

使用 Fashion-MNIST 训练 256 类灰度图像 Categorical NADE：

```bash
python main.py --config configs/categorical.yaml
```

使用 UCI White Wine 训练连续型 RNADE：

```bash
python main.py --config configs/continuous.yaml
```

图像模型依次执行：

```text
加载数据 → 二值化或类别转换 → NADE 训练 → 验证 → 测试 → 图像生成
```

连续型模型依次执行：

```text
加载数据 → 删除 quality → 标准化 → RNADE 训练 → 验证 → 测试 → CSV 生成
```

## 参数配置

二值模式的主要配置：

```yaml
dataset: mnist
representation: binary
binarization: fixed
threshold: 0.5
hidden_dim: 128
batch_size: 128
```

类别型模式保留原始 8 位灰度值：

```yaml
dataset: fashion_mnist
representation: categorical
num_categories: 256
hidden_dim: 32
batch_size: 32
```

连续型 RNADE 使用等权高斯混合条件分布：

```yaml
dataset: white_wine
representation: continuous
train_size: 3898
validation_size: 500
test_size: 500
num_components: 10
min_std: 0.01
hidden_dim: 64
batch_size: 128
learning_rate: 0.001
samples_path: outputs/rnade_white_wine_samples.csv
statistics_path: outputs/rnade_white_wine_statistics.csv
```

- `dataset`：数据集名称
- `representation`：`binary`、`categorical` 或 `continuous`
- `num_categories`：类别型灰度固定为 256 类，对应原始像素值 0～255
- `num_components`：连续型条件分布的高斯分量数
- `min_std`：高斯标准差的数值稳定下限
- `train_size`、`validation_size`、`test_size`：数据划分大小
- `hidden_dim`：共享隐藏层维度
- `num_samples`：训练结束后生成的样本数

完整参数见 [`configs/`](configs/)。

## 数据表示

### 二值模式

```text
原始 uint8 像素 0～255
→ 归一化到 0～1
→ 固定阈值或随机二值化
→ float32 像素 0 或 1
→ Bernoulli NADE
```

### 类别型模式

```text
原始 uint8 像素 0～255
→ 保持数值并转换为 long
→ 类别标签 0～255
→ Categorical NADE + Softmax
```

### 连续型模式

```text
UCI White Wine CSV
→ 删除离散 quality 标签
→ 确定性划分训练、验证和测试集
→ 使用训练集均值与标准差进行特征标准化
→ RNADE + 高斯混合分布
```

标准化参数只从训练集计算，避免验证集和测试集信息泄漏。生成结果会反标准化回原始
理化单位后写入 CSV。

## 模型原理

### 自回归分解

一个样本表示为 $D$ 个有固定顺序的变量。模型将联合分布分解为：

```math
p(\boldsymbol{x}) = \prod_{i=0}^{D-1} p(x_i \mid \boldsymbol{x}_{<i})
```

其中 $\boldsymbol{x}_{<i}=(x_0,\ldots,x_{i-1})$ 表示当前位置之前的所有变量。

### 二值 NADE

当 $x_i\in\{0,1\}$ 时，隐藏状态和 Bernoulli 条件概率为：

```math
h_i = \sigma\left(c + \sum_{j=0}^{i-1} W_jx_j\right)
```

```math
p(x_i=1 \mid \boldsymbol{x}_{<i})
= \sigma\left(b_i + V_i^{\mathsf T}h_i\right)
```

### 类别型 NADE

当 $x_i\in\{0,1,\ldots,255\}$ 时，类别嵌入形成隐藏状态：

```math
h_i = \sigma\left(c + \sum_{j=0}^{i-1}E_{j,x_j}\right)
```

输出层为位置 $i$ 生成 256 个 logits，并通过 Softmax 得到类别概率：

```math
p(x_i=k \mid \boldsymbol{x}_{<i})
= \mathrm{softmax}(A_i h_i+b_i)_k
```

### 连续型 RNADE

当 $x_i\in\mathbb{R}$ 时，每个条件分布由 $K$ 个等权一维高斯分布组成：

```math
p(x_i \mid \boldsymbol{x}_{<i})
= \sum_{k=1}^{K}\frac{1}{K}
\mathcal{N}\left(x_i;\mu_i^k,(\sigma_i^k)^2\right)
```

共享隐藏状态只使用当前位置之前的变量：

```math
h_i = \sigma\left(c + \sum_{j=0}^{i-1}W_jx_j\right)
```

输出层为每个位置预测 $K$ 个均值和 $K$ 个标准差。均值不受限制，标准差使用
Softplus 和最小值约束为正：

```math
(\mu_i^1,\ldots,\mu_i^K,s_i^1,\ldots,s_i^K)=f_i(h_i)
```

```math
\sigma_i^k = \mathrm{softplus}(s_i^k)+\sigma_{\min} > 0
```

训练目标是整个样本的负对数似然：

```math
\mathcal{L}(\boldsymbol{x})
= -\sum_{i=0}^{D-1}\log p(x_i \mid \boldsymbol{x}_{<i})
```

训练时可以利用累积和并行计算全部隐藏状态；生成时需要按照变量顺序，从对应条件
分布中逐个采样。RNADE 会先均匀选择一个高斯分量，再从该分量采样。

## 项目结构

```text
nade/
├── configs/
│   ├── bernoulli.yaml
│   ├── categorical.yaml
│   └── continuous.yaml
├── data/
│   └── README.md
├── nade/
│   ├── models/
│   │   ├── bernoulli.py
│   │   ├── categorical.py
│   │   └── continuous.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── reporting.py
│   ├── tabular.py
│   ├── training.py
│   └── visualization.py
├── main.py
├── requirements.txt
└── README.md
```

## 输出结果

### 训练与评估指标

训练期间，程序会逐轮输出训练集和验证集指标。训练结束后，将恢复验证集 NLL 最低
轮次的模型，并使用该模型完成测试和样本生成。

- `train NLL`：训练集平均负对数似然
- `validation NLL`：验证集平均负对数似然，用于选择最优模型
- `bits/dim`：平均每个维度所需的比特数，便于比较同一数据表示下的模型

NLL 和 bits/dim 均为越低越好。连续密度与离散概率的度量定义不同，因此不应直接
比较连续模型与离散模型的指标数值。

### 图像模型

Bernoulli NADE 和 Categorical NADE 会在项目根目录生成 `comparison.png`：

- 左侧 `Real`：测试集中的真实图像
- 右侧模型名称：NADE 生成的样本

两侧图像数量由配置项 `num_samples` 控制，可用于直观比较真实数据与生成结果。

### 连续模型

RNADE 会在 `outputs/` 目录生成以下文件：

- `rnade_white_wine_samples.csv`：生成的连续样本；每行表示一条样本，每列对应一个
  White Wine 理化特征，数值已还原到原始单位
- `rnade_white_wine_statistics.csv`：真实测试数据与生成数据的逐特征统计结果，包括
  均值和标准差

这些结果可用于检查生成样本的取值范围，以及模型是否学习到各个特征的边缘分布。

## 参考资料

- Hugo Larochelle and Iain Murray, [The Neural Autoregressive Distribution
  Estimator](https://proceedings.mlr.press/v15/larochelle11a.html), AISTATS 2011
- Benigno Uria, Iain Murray and Hugo Larochelle,
  [RNADE: The real-valued neural autoregressive density-estimator](https://proceedings.neurips.cc/paper/2013/hash/53adaf494dc89ef7196d73636eb2451b-Abstract.html),
  NeurIPS 2013
- Paulo Cortez et al., [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality),
  UCI Machine Learning Repository, 2009
