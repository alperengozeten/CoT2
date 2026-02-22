<div align="center">

# CoT2 - Continuous Chain of Thought Enables Parallel Exploration and Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2505.23648-b31b1b.svg)](https://arxiv.org/abs/2505.23648)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)

Official implementation of the paper  
**"Continuous Chain of Thought Enables Parallel Exploration and Reasoning."**

</div>

> Continuous CoT (CoT2) replaces one-token-at-a-time decoding with continuous simplex-weighted token mixtures, allowing the model to track and combine multiple reasoning traces in parallel through distributional CSFT supervision. It also introduces continuous sampling methods using Multi-Token Sampling and Dirichlet Sampling.

## What is in this repository?

| Component | Entry point | Launcher |
|---|---|---|
| Discrete CoT baseline | `train/discrete_train.py` | `bash scripts/discrete_evaluation.sh` |
| Continuous CoT (CoT2) | `train/continuous_train.py` | `bash scripts/continuous_evaluation.sh` |
| CoT2 + GRPO (MTS sampling) | `train/continuous_train_grpo_mts.py` | `bash scripts/grpo_evaluation.sh` |
| CoT2 + GRPO (Dirichlet sampling) | `train/continuous_train_grpo_dirichlet.py` | direct Python command |
| Evaluation helpers | `eval/evaluation.py` | imported by train scripts |
| Plotting utilities | `train/utils.py`, `visualization/generate_plots.py` | automatic + manual plotting |

The codebase includes experiments for MNNS, and includes ProntoQA/ProsQA assets under `prontoqa-prosqa/` (with local modifications).

## Repository layout

```text
.
├── data/                    # MNNS data generation utilities
├── eval/                    # evaluation helpers
├── scripts/                 # reproducible launch scripts
├── train/                   # training entry points (discrete, CoT2, GRPO variants)
├── visualization/           # plotting scripts
├── prontoqa-prosqa/         # adapted upstream ProntoQA/ProsQA code
├── logs/                    # run logs (created by scripts)
├── models/                  # saved checkpoints (created by scripts)
└── figures/                 # exported figures (created by scripts)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick start (MNNS)

All launchers create `logs/`, `models/`, and `figures/` directories and run jobs with `nohup` in the background.

```bash
# Discrete CoT baseline
bash scripts/discrete_evaluation.sh

# Continuous CoT (CoT2)
bash scripts/continuous_evaluation.sh

# CoT2 + GRPO (MTS)
bash scripts/grpo_evaluation.sh
```

Monitor training logs:

```bash
tail -f logs/<run_subdir>/<run_name>.out
```

Multi-seed launchers are also provided:

```bash
bash scripts/discrete_evaluation_multi_run.sh
bash scripts/continuous_evaluation_multi_run.sh
```

## Direct run example (Dirichlet GRPO)

```bash
python -u train/continuous_train_grpo_dirichlet.py \
  --max_seq_len 11 \
  --embedding_dim 32 \
  --digit_range 1 10 \
  --batch_size 16 \
  --seq_length 4 \
  --split_method random_permutation \
  --num_epochs 100 \
  --output_dir grpo_dirichlet_runs/ \
  --num_layers 1 \
  --num_heads 1 \
  --num_rollouts 8 \
  --K 3 \
  --kl_coeff 0.0 \
  --clip_epsilon 0.1 \
  --sampling_normalization true
```

## ProntoQA / ProsQA note

The `prontoqa-prosqa/` folder contains upstream-derived code plus modifications used in this project.  
Refer to `prontoqa-prosqa/README.md` for task-specific usage details.

## Citation

If you use this repository, please cite:

```bibtex
@misc{gozeten2025continuouschainthoughtenables,
  title={Continuous Chain of Thought Enables Parallel Exploration and Reasoning},
  author={Halil Alperen Gozeten and M. Emrullah Ildiz and Xuechen Zhang and Hrayr Harutyunyan and Ankit Singh Rawat and Samet Oymak},
  year={2025},
  eprint={2505.23648},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2505.23648}
}
```

## License

Top-level repository code is licensed under MIT (see `LICENSE`).  
`prontoqa-prosqa/` retains its own Apache-2.0 license (see `prontoqa-prosqa/LICENSE`).
