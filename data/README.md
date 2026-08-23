# Data

数据集由程序按配置自动下载：

```bash
# MNIST
python main.py --download-data

# Fashion-MNIST
python main.py --config configs/categorical_fashion_mnist.yaml --download-data

# KMNIST
python main.py --config configs/categorical_kmnist.yaml --download-data
```

MNIST 的 IDX 文件直接保存在 `data/`，另外两种数据集分别保存在
`data/fashion_mnist/` 和 `data/kmnist/`，避免同名 IDX 文件相互覆盖。

这些文件由 `.gitignore` 排除，不提交到 GitHub。直接开始训练时，如果缺少对应
数据，程序也会自动下载并校验文件。
