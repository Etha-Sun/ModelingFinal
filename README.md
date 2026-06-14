# 二维分布生成建模实验

本项目为数学建模期末大作业，研究 Gaussian Mixture、Ring、Two Moons、Spiral 四类二维分布上的生成建模问题。实验比较 KDE、GMM、VAE、DDPM 四类模型，并使用 MMD、Sliced Wasserstein Distance、Precision、Coverage、NLL 等指标评价生成质量。

## 环境说明

实验环境记录见 `results/run_manifest.json`。本次主要环境如下：

- Python 3.12.7
- NumPy 1.26.4
- SciPy 1.13.1
- scikit-learn 1.5.1
- PyTorch 2.5.1
- Matplotlib 3.9.2

安装依赖：

```bash
pip install -r requirements.txt
```

## 目录说明

- `report/main.pdf`：最终实验报告。
- `report/main.tex`：报告 LaTeX 源文件。
- `src/run_experiments.py`：完整实验代码，包括数据生成、模型训练、采样、评价和绘图。
- `distribution2d_gen/`：题目提供的数据生成程序。
- `data/`：由题目数据生成程序导出的默认训练集、测试集和隐藏测试集，主要用于查看数据格式和预览分布。
- `results/`：实验指标、生成样本、图像、表格和运行记录。

## 数据说明

`src/run_experiments.py` 不直接读取 `data/*.npy`。主实验会在脚本内部复现 `distribution2d_gen/generate_data.py` 的生成逻辑，并根据不同随机种子重新生成训练集、测试集和隐藏测试集，用于重复实验、隐藏测试集检查和统计均值、标准差。

因此，生成 `data/` 主要用于导出默认数据文件和预览图；完整实验只需要运行 `src/run_experiments.py`。

## 复现实验

生成默认数据文件和预览图：

```bash
python distribution2d_gen/generate_data.py --output-dir data --plot
```

运行完整实验：

```bash
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp \
python src/run_experiments.py \
  --out-dir results \
  --n-train 2000 --n-test 2000 --n-generate 2000 \
  --seeds 0 1 2 --vae-epochs 320 --ddpm-epochs 2500 --ddpm-steps 100
```
