# SPaCE-MADAC

Reinforcement learning for Dynamic Algorithm Configuration (DAC) of multi-objective
evolutionary algorithms, evaluated on the MA-DAC / MaMo benchmark (MOEA/D-based).

This project implements a single-agent PPO baseline (a gap in prior work, which only
tested DQN and multi-agent methods), and integrates the SPACE curriculum-learning
algorithm on top of it to investigate instance selection on multiobjective DAC.

## Building on

- Xue et al., **MA-DAC** (NeurIPS 2022)
- Lu et al., **Seq-MADAC** (NeurIPS 2025)
- Eimer et al., **SPACE** 
- Zhang & Li, **MOEA/D**

## Setup

```bash
pip install -r rerun/requirements.txt
```

## Running

```bash
# DQN baseline
python -m rerun.dqn.dqn --key M_2_46_3 

# PPO baseline

Train:
```bash
python -m ppo.ppo --seed <seed>
```

Evaluate a trained model (where --train-seed is the seed that was used for training):
```bash
python -m ppo.test_ppo --train-seed <seed>
```

All other flags are locked at their defaults for these experiments — only `--seed` (train) / `--train-seed` (eval) vary across the 3 training seeds (42, 123, 2022).

# SPACE (0=no curriculum, 1=size growth only, 2=full SPACE)
python -m rerun.space.train_space_ppo --key M_2_46_3 --use-space 2
```

Each folder also has a matching `test_*.py` for evaluation (e.g. `test_ppo.py`),
using the same `--key` and flags as training.
