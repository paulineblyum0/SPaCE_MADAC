# SPaCE-MADAC

Reinforcement learning for Dynamic Algorithm Configuration (DAC) of multi-objective
evolutionary algorithms, evaluated on the MA-DAC / MaMo benchmark (MOEA/D-based).

This project implements a single-agent PPO baseline (a gap in prior work, which only
tested DQN and multi-agent methods), and integrates the SPACE curriculum-learning
algorithm on top of it.

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
python -m rerun.ppo.ppo --key M_2_46_3 

# SPACE (0=no curriculum, 1=size growth only, 2=full SPACE)
python -m rerun.space.train_space_ppo --key M_2_46_3 --use-space 2
```

Each folder also has a matching `test_*.py` for evaluation (e.g. `test_ppo.py`),
using the same `--key` and flags as training.
