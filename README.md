# CoT2 - Continuous Chain of Thought Enables Parallel Exploration and Reasoning 

This repository contains the code for the paper: *“Continuous Chain of Thought Enables Parallel Exploration and Reasoning”*

## Overview

This codebase contains SFT and CSFT training methods for discrete CoT and CoT2 models, and also includes two GRPO variants with different sampling strategies, on the MNNS, ProntoQA, and ProsQA tasks. Below is the general directory structure:

1. **`discrete_train.py`**

   * Contains the SFT training and evaluation code for the *discrete* CoT model.
   * To run, use `discrete_evaluation.sh` or `discrete_evaluation_multi_run.sh` for multiple runs.

2. **`continuous_train.py`**

   * Implements CSFT training for *CoT2* models with a controllable supervision trajectory budget.
   * To run, use `continuous_evaluation.sh` or `continuous_evaluation_multi_run.sh` for multiple runs.

3. **`continuous_train_grpo_mts.py`**  && **`continuous_train_grpo_dirichlet.py`**

   * Integrates **GRPO** with CoT2 using **MTS** (multi-token sampling) and **Dirichlet** sampling.
   * Can be run using `grpo_evaluation.sh`.

For ProntoQA and ProsQA tasks, we use the same training procedure. For data processing,
we provide our modifications to the original repository of ProntoQA, along with run scripts, in the `prontoqa-prosqa/` folder.

## Citation

If you find our work helpful, please cite our paper:

```bibtex
@misc{gozeten2025continuouschainthoughtenables,
   title={Continuous Chain of Thought Enables Parallel Exploration and Reasoning}, 
   author={Halil Alperen Gozeten and M. Emrullah Ildiz and Xuechen Zhang and Hrayr Harutyunyan and Ankit Singh Rawat and Samet Oymak},
   year={2025},
   eprint={2505.23648},
   archivePrefix={arXiv},
   primaryClass={cs.LG},
   url={https://arxiv.org/abs/2505.23648}, 
}

```

