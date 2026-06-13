# 二维分布生成建模实验

本项目为数学建模期末大作业，研究 Gaussian Mixture、Ring、Two Moons、Spiral 四类二维分布上的生成建模问题。实验比较 KDE、GMM、VAE、DDPM 四类模型，并使用 MMD、Sliced Wasserstein Distance、Precision、Coverage、NLL 等指标评价生成质量。

## 目录说明

- `report/main.pdf`：最终实验报告。
- `report/main.tex`：报告 LaTeX 源文件。
- `src/run_experiments.py`：完整实验代码，包括模型训练、采样、评价和绘图。
- `distribution2d_gen/`：题目提供的数据生成程序。
- `data/`：由数据生成程序生成的训练集、测试集和隐藏测试集。
- `results/`：实验指标、生成样本、图像、表格和运行记录。

## 复现实验

生成数据：

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