# Data

在项目根目录运行以下命令即可准备 MNIST 数据：

```bash
python main.py --download-data
```

下载完成后，训练集与测试集的 IDX 文件直接保存在 `data/` 中。下载器会校验
压缩文件的 MD5，并依次尝试 CVDF/Google、Azure 和 torchvision 镜像。

直接运行 `python main.py` 时，如果本地没有数据，也会自动下载。
