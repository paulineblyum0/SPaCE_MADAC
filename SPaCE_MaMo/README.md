# Abstract

Multi-objective optimisation problems involve several, often conflicting, objectives that must be considered simultaneously. Their solution quality can strongly depend on how the optimisation algorithm is configured. Dynamic Algorithm Configuration (DAC) adapts these parameters during execution using reinforcement learning. However, when several heterogeneous parameters must be jointly configured, it remains unclear whether this requires a multi-agent implementation.
This dissertation investigates this question using MOEA/D within the MaMo benchmark. A single-agent Proximal Policy Optimisation (PPO) approach is compared with DQN and the established multi-agent MA-DAC method. Adaptive instance selection is then introduced through SPACE, a self-paced learning method, to examine whether controlling the instances encountered during training can further improve PPO for MaMo.



## Building on

- Xue et al., MA-DAC
- Eimer et al., SPACE
- Zhang & Li, MOEA/D
- Pickering, SPACE-ES

## Setup

```bash
pip install -r requirements.txt
```

Run all commands below from the project root.

## Running

### DQN baseline

The DQN implementation under `dqn/` (except `igd_eval_hook.py`) is adapted from the official MA-DAC repo, with modifications:
https://github.com/lamda-bbo/madac/tree/main/algos/dac

Train:
```bash
python -m dqn.dqn --key M_2_46_3 --seed <seed>
```

Evaluate (`--train-seed` is the seed used for training):
```bash
python -m dqn.test_dqn --train-seed <seed>
```

### PPO baseline

Train:
```bash
python -m ppo.ppo --seed <seed>
```

Evaluate:
```bash
python -m ppo.test_ppo --train-seed <seed>
```

### SPACE (instance selection for PPO)

The SPACE code was written using Vincent Pickering's SPACE-ES/BBOB implementation (Pickering, 2026) as a base.
It has since diverged substantially mostly to work under the parallelised set up and with the MaMo benchmark. 

`--use-space`: 0 = no curriculum (matches the PPO baseline), 1 = size growth only, 2 = SPACE.

Train:
```bash
python -m space_parallel.train_space --key M_2_46_3 --use-space 2 --seed <seed>
```

Evaluate:
```bash
python -m space_parallel.test_space --train-seed <seed>
```
--use-space must match between training and evaluation. Default is SPACE (use-space 2) for both. 

### MA-DAC

The MA-DAC training code under `madac/` (everything except `test_madac.py` and `madac_igd_eval_hook.py`)
is from the official implementation: https://github.com/lamda-bbo/madac/tree/main/algos/madac

Train:
```bash
python -m madac.main --config=vdn_ns --env-config=moea with env_args.key=M_2_46_3 seed=<seed>
```

Evaluate:
```bash
python -m madac.test_madac --train-seed <seed>
```

All flags are locked at their defaults for these experiments. Only `--seed` (train) / `--train-seed` (eval) 
vary, across the 3 training seeds used throughout (42, 123, 2022).

## Figures & tables

Each research question has its own script under `figures/`, run from the
project root:

```bash
python -m figures.rq1_adaptive_open
python -m figures.rq1_plateau
python -m figures.rq2_dqn_vs_madac
python -m figures.rq3_algorithm_comparison
python -m figures.rq4_learning_dynamics
python -m figures.rq5_space_curriculum
```

Each writes its figures and tables to `figures/output/`, prefixed by RQ name. 