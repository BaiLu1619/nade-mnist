# Data

根据需要指定并下载对应数据集：

```bash
# MNIST
python main.py --config configs/bernoulli.yaml --download-data

# Fashion-MNIST
python main.py --config configs/categorical.yaml --download-data

# UCI White Wine
python main.py --config configs/continuous.yaml --download-data
```

各数据集分别保存在 `data/mnist/`、`data/fashion_mnist/` 和 `data/white_wine/`，
避免同名文件相互覆盖。

这些文件由 `.gitignore` 排除，不提交到 GitHub。直接开始训练时，如果缺少对应
数据，程序也会自动下载并校验文件。

Fashion-MNIST 会依次尝试 Zalando 官方 GitHub 数据目录、TensorFlow 公开镜像和
torchvision 默认地址；单个下载源不可用时会自动切换。

White Wine 数据保存在 `data/white_wine/winequality-white.csv`。下载器会依次尝试
UCI 的 CSV 和 ZIP 官方入口，并验证 12 列表头、4,898 行数据以及所有数值；训练时
排除离散的 `quality` 列。
