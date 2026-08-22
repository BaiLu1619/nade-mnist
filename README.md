# NADE-MNIST

## 项目介绍

NADE-MNIST 是一个基于 PyTorch 的二值图像生成项目，使用 Neural
Autoregressive Distribution Estimator（NADE）学习 MNIST 手写数字的概率分布，
并完成模型训练、测试集评估和图像生成。

## 功能特性

- 下载并预处理 MNIST 数据集
- 使用共享隐藏层实现 NADE 自回归条件分布
- 并行计算所有像素的训练概率
- 使用 NLL 和 bits per dimension 评价生成模型
- 按像素顺序生成新的手写数字
- 生成真实 MNIST 与模型样本的左右对比图
- 固定数据划分和随机种子，便于复现实验

## 安装

环境要求

- Python 3.10、3.11 或 3.12

可以先创建虚拟环境，再安装相关依赖

```bash
conda create -n nade-mnist python=3.10 -y
conda activate nade-mnist
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 数据准备

运行以下命令下载并校验 MNIST：

```bash
python main.py --download-data
```

数据将解压到 `data/`。下载器会依次尝试多个镜像，并使用 torchvision 提供的
MD5 校验值检查文件。直接运行主程序时，如果本地没有数据，也会自动下载。

## 使用方法

```bash
python main.py
```

该命令依次完成以下流程：

```text
加载数据 → 图像二值化 → NADE 训练 → 验证 → 测试 → 图像生成
```

使用其他配置文件：

```bash
python main.py --config path/to/config.yaml
```

## 参数配置

默认配置位于 [`configs/config.yaml`](configs/config.yaml)：

```yaml
data_dir: data
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

主要参数：

- `hidden_dim`：共享隐藏层维度
- `epochs`：训练轮数
- `batch_size`：批大小
- `learning_rate`：学习率
- `num_samples`：生成图片数量
- `binarization`：`fixed` 为固定阈值，`stochastic` 为随机二值化

## 模型原理

MNIST 图像经过二值化后被展平为 784 维的 0/1 向量。NADE 将联合概率按照
像素顺序分解为一系列条件概率：

```math
p(x) = \prod_{i=0}^{D-1} p\left(x_i \mid x_{0:i}\right)
```

预测第 $i$ 个像素时，NADE 使用之前的像素计算共享隐藏表示：

```math
h_i = \sigma\left(c + \sum_{j=0}^{i-1} W_j x_j\right)
```

然后通过该位置对应的输出权重计算 Bernoulli 条件概率：

```math
p\left(x_i=1 \mid x_{0:i}\right)
= \sigma\left(b_i + V_i^{\mathsf T}h_i\right)
```

其中 $x_{0:i}$ 表示第 $i$ 个像素之前的所有像素。隐藏层参数在不同位置之间共享，
训练时可以使用累积和并行计算所有条件概率；生成时则按照像素顺序逐个采样。

## 项目结构

```text
nade-mnist/
├── configs/
│   └── config.yaml          # 模型和训练参数
├── data/
│   └── README.md            # 数据下载说明
├── src/
│   ├── dataset.py           # MNIST 下载与 DataLoader
│   ├── preprocess.py        # 图像二值化
│   ├── model.py             # NADE 模型
│   ├── train.py             # 训练、评估和生成流程
│   ├── visualization.py     # 真实/生成图片对比
│   └── utils.py             # 配置和随机种子
├── main.py                  # 主程序入口
├── requirements.txt
└── README.md
```

## 输出结果

运行过程中，终端会输出：

- 训练集与验证集 NLL
- 验证集与测试集 bits/dim

NLL 和 bits/dim 越低，表示模型对 MNIST 数据分布的拟合越好。NADE 是生成模型，
因此不使用分类准确率。

运行结束后，项目根目录会生成 `comparison.png`。图片左侧为真实二值化 MNIST，
右侧为 NADE 生成的样本。

## 参考资料

- Hugo Larochelle and Iain Murray, [The Neural Autoregressive Distribution
  Estimator](https://proceedings.mlr.press/v15/larochelle11a.html), AISTATS 2011

## License

MIT
