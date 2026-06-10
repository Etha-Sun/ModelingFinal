# 复杂二维数据分布数据生成

生成默认数据：

```bash
python generate_data.py
```

默认每类分布生成 2000 个训练样本、2000 个测试样本和 2000 个隐藏测试样本。输出目录为 `data/`。

输出文件：

```text
data/train.npy
data/test.npy
data/train_label.npy
data/test_label.npy
data/hidden_test.npy
data/hidden_test_label.npy
data/metadata.json
```

标签含义：

```text
0: Gaussian Mixture
1: Ring
2: Two Moons
3: Spiral
```

常用参数：

```bash
python generate_data.py --seed 20260525
python generate_data.py --train-per-class 2000 --test-per-class 2000 --hidden-per-class 2000
python generate_data.py --output-dir data
```

如果环境安装了 `matplotlib`，可以额外生成预览图：

```bash
python generate_data.py --plot
```
