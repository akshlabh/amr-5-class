# MCLDNN — Automatic Modulation Recognition on RML2016.10a

Research project based on the [AMR-Benchmark](https://github.com/Richardzhangxx/AMR-Benchmark) by Richardzhangxx, with a cleaned-up `src/` implementation for training, evaluation, and experiment tracking.

## Project Focus

This repository implements and experiments with the **MCLDNN** model for Automatic Modulation Recognition (AMR) on the **RML2016.10a** dataset, restricted to a curated 5-class subset:

| Class | Type |
|---|---|
| BPSK  | Phase Shift Keying |
| QPSK  | Phase Shift Keying |
| 8PSK  | Phase Shift Keying |
| QAM16 | Quadrature Amplitude Modulation |
| QAM64 | Quadrature Amplitude Modulation |

An **ablation study** removing BPSK (4-class: QPSK, 8PSK, QAM16, QAM64) is maintained in parallel.

## About `src/`

The `src/` folder is the main code path for this repository. It contains the maintained implementation that drives the current experiments:

- `src/train.py` handles training from a YAML config, including checkpointing and resume support.
- `src/evaluate.py` runs single-experiment evaluation and side-by-side comparison of multiple runs.
- `src/dataset.py` loads the filtered dataset variants used by the 4-class and 5-class experiments.
- `src/models/` contains the active MCLDNN model definition used by the project.
- `src/utils/` provides shared helpers for metrics, plots, and reproducibility.

If you want to understand or modify the current workflow, start with `src/train.py` and `src/evaluate.py`, then follow the imports into `src/dataset.py`, `src/models/`, and `src/utils/`.

---

## Quick Start

### 1. Prepare datasets (run once, locally)

```bash
# Verify your pickle's class names first
python scripts/verify_classes.py data/RML2016.10a_dict.pkl

# Create 5-class filtered dataset
python scripts/prepare_dataset.py \
    --src  data/RML2016.10a_dict.pkl \
    --dst  data/RML2016.10a_5class.pkl \
    --classes BPSK QPSK 8PSK QAM16 QAM64

# Create 4-class filtered dataset (ablation)
python scripts/prepare_dataset.py \
    --src  data/RML2016.10a_dict.pkl \
    --dst  data/RML2016.10a_4class.pkl \
    --classes QPSK 8PSK QAM16 QAM64
```

Upload each `.pkl` to a separate **Kaggle Dataset**.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train (local or Kaggle)

```bash
# 5-class baseline (primary model)
python src/train.py --config configs/exp_5class_baseline.yaml

# 4-class ablation
python src/train.py --config configs/exp_4class_ablation.yaml

# Resume from checkpoint
python src/train.py --config configs/exp_5class_baseline.yaml \
    --resume experiments/5class_baseline/checkpoints/best_model.weights.h5
```

### 4. Evaluate and compare

```bash
# Evaluate single experiment
python src/evaluate.py \
    --config  configs/exp_5class_baseline.yaml \
    --weights experiments/5class_baseline/checkpoints/best_model.weights.h5

# Compare 5-class vs 4-class ablation
python src/evaluate.py \
    --compare \
    --configs  configs/exp_5class_baseline.yaml configs/exp_4class_ablation.yaml \
    --weights  <path_to_5class_weights> <path_to_4class_weights> \
    --labels   "5-class" "4-class ablation" \
    --save_dir experiments/comparison_5vs4
```

---

## Repository Structure

```
AMR/
├── configs/                         # Experiment YAML configs
│   ├── exp_5class_baseline.yaml     # PRIMARY — 5-class model
│   ├── exp_4class_ablation.yaml     # Ablation — BPSK removed
│   └── exp_5class_ablation.yaml     # Placeholder for future variants
│
├── src/                             # Main maintained implementation
│   ├── dataset.py                   # Class-filtered data loader
│   ├── train.py                     # Training entry point
│   ├── evaluate.py                  # Evaluation + comparison
│   ├── models/
│   │   └── mcldnn.py                # MCLDNN (TF2/Keras3)
│   └── utils/
│       ├── mltools.py               # Plots and metrics
│       └── seed.py                  # Reproducibility seeds
│
├── notebooks/
│   ├── train_5class_baseline.ipynb  # Kaggle training notebook
│   ├── train_4class_ablation.ipynb  # Kaggle ablation notebook
│   └── evaluate_and_compare.ipynb  # Comparison notebook
│
├── scripts/
│   ├── prepare_dataset.py          # Filter dataset to class subset
│   ├── verify_classes.py           # Check pickle class names
│   └── download_weights.py         # Download weights from GitHub Releases
│
├── RML201610a/MCLDNN/              # ORIGINAL upstream code (reference only)
│   ├── dataset2016.py
│   ├── main.py
│   ├── mltools.py
│   └── rmlmodels/MCLDNN.py
│
├── CHANGELOG.md                    # Training run log
├── requirements.txt
└── .gitignore
```

> **Note**: `data/`, `experiments/` are gitignored.  
> Datasets → Kaggle Datasets | Weights → GitHub Releases

---

## Experiment Tracks

| Track | Classes | Config | Weights |
|---|---|---|---|
| **5-class baseline** | BPSK, QPSK, 8PSK, QAM16, QAM64 | `exp_5class_baseline.yaml` | GitHub Release |
| **4-class ablation** | QPSK, 8PSK, QAM16, QAM64 | `exp_4class_ablation.yaml` | GitHub Release |
| **Future variants**  | Any | `exp_5class_ablation.yaml` template | TBD |

---

## Reproducibility

Every experiment is reproducible from:
1. This repository at a specific **git commit**
2. A specific **Kaggle dataset version** (filtered `.pkl`)
3. A specific **weight file** from GitHub Releases
4. The matching **YAML config file**

All runs are documented in [CHANGELOG.md](CHANGELOG.md).

---

## Credits

This project preserves and builds on the original AMR-Benchmark work by Richardzhangxx. Please credit the upstream repository if you reuse this code or ideas:

```bibtex
@misc{amr-benchmark,
  author = {Richardzhangxx},
  title  = {AMR-Benchmark},
  year   = {2022},
  url    = {https://github.com/Richardzhangxx/AMR-Benchmark}
}
```

Original MCLDNN paper:
> Meng, F., Chen, P., Wu, L., & Wang, X. (2019). Automatic modulation classification: A deep learning enabled approach. *IEEE Transactions on Vehicular Technology*, 68(11), 10760-10772.

The legacy upstream code is kept under `RML201610a/MCLDNN/` as a reference implementation.
