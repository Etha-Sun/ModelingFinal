# Research Pipeline Report

**Direction**: 二维分布生成建模  
**Plan source**: `/Users/sun/Desktop/Homework/Mathematical Modeling/PAPER_PLAN.md`  
**Reference format**: `hw_3/option2_insects/report.pdf`  
**Date**: 2026-06-10  
**Pipeline used**: fixed-topic implementation -> experiments -> report writing -> compile and QA

## Journey Summary

- The topic was already fixed by the user: the final project is the course option "二维分布生成建模".
- Integrated the provided `distribution2d_gen/generate_data.py` course generator for four 2D distributions: Gaussian Mixture, Ring, Two Moons, and Spiral.
- Implemented four unconditional model families: KDE, GMM, VAE, and DDPM.
- Implemented a unified conditional DDPM for the optional conditional-generation task.
- Ran the main comparison over 4 datasets x 4 models x 3 seeds, with 2000 training samples and 2000 test/generated samples per dataset.
- Computed MMD, sliced Wasserstein distance, Precision, Coverage, 2D grid support coverage, NLL for explicit-density models, and training time.
- Expanded robustness analysis to KDE, GMM, VAE, and DDPM over 3 seeds.
- Generated paper-quality PDF/PNG figures and LaTeX tables.
- Wrote and compiled a Chinese course-report-style PDF matching the previous homework report format.

## Final Status

- [x] Ready for course submission after user-side naming/packaging.
- [x] PDF compiles successfully.
- [x] No undefined references or citations.
- [x] No LaTeX overfull warnings.
- [x] PDF fonts are embedded; no Type 3 fonts remain.

## Key Result

The final conclusion is not a generic "neural models win" story. Under the provided official 2D generator and 2000 samples per class, KDE is the strongest overall sample-matching baseline: it achieves the lowest MMD, lowest sliced Wasserstein distance, and highest Coverage on all four distributions. GMM remains useful for explicit likelihood and high Precision on several datasets, DDPM can generate recognizable curved structures but does not beat KDE on the main metrics, and VAE visibly over-smooths low-density holes and curved supports. Conditional DDPM can generate all four classes from one model, but parameter sharing introduces a measurable quality cost.

## Files Created or Modified

- `final_project/src/run_experiments.py` — full data/model/evaluation/plot pipeline.
- `final_project/distribution2d_gen/` — provided course data generator.
- `final_project/data/` — default official train/test/hidden splits.
- `final_project/results/metrics_summary.csv` — aggregate metrics.
- `final_project/results/metrics_by_seed.csv` and `.json` — per-seed metrics.
- `final_project/results/conditional_metrics_summary.csv` — conditional DDPM metrics.
- `final_project/results/robustness.csv` — robustness metrics for all four model families.
- `final_project/results/figures/*.pdf` and `.png` — report figures.
- `final_project/results/tables/*.tex` — LaTeX table fragments.
- `final_project/results/run_manifest.json` — reproducibility record.
- `final_project/report/main.tex` — report source.
- `final_project/report/main.pdf` — compiled final report.

## Reproduction Command

```bash
python final_project/distribution2d_gen/generate_data.py \
  --output-dir final_project/data --plot

MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp \
python final_project/src/run_experiments.py \
  --out-dir final_project/results \
  --n-train 2000 --n-test 2000 --n-generate 2000 \
  --seeds 0 1 2 --vae-epochs 320 --ddpm-epochs 2500 --ddpm-steps 100
```

Compile:

```bash
cd final_project/report
env LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

## Remaining TODOs

- Package the submission zip using the required course naming convention.
