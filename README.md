# Modeling Final Project

This repository contains the final project for USTC Mathematical Modeling: a comparative study of generative models for two-dimensional complex distributions.

## Contents

- `src/run_experiments.py`: reproducible experiment pipeline.
- `results/`: metrics, generated samples, robustness/conditional-generation outputs, figures, LaTeX table fragments, and run manifest.
- `report/main.tex`: report source.
- `report/main.pdf`: compiled final report.
- `RESEARCH_PIPELINE_REPORT.md`: execution summary and reproduction commands.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp \
python src/run_experiments.py \
  --out-dir results \
  --n-train 1200 --n-test 900 --n-generate 900 \
  --seeds 0 1 2 --vae-epochs 320 --ddpm-epochs 2500 --ddpm-steps 100
```

Compile the report:

```bash
cd report
env LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```
