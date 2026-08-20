# space/test_space.py
"""
Evaluation harness for SPACE-trained PPO models on MaMo.

Mirrors ppo/test_ppo.py exactly (same per-repeat reseeding, same
--early-stop flag-consistency requirement, same .npz output format) so
results drop into the existing comparison pipeline unchanged.

The one substantive difference: at eval time the curriculum is set once to
the full instance set and never touched again -- SPACE only shapes
*training*, evaluation should see every problem exactly as the PPO
baseline's evaluation does.
"""

import os
import argparse
import random
import numpy as np
import ray

from stable_baselines3 import PPO

from mamo.mamo_register import Task
from space.space_env import SPaceEnv
from space.space_enum import space_operation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, default='M_2_46_3')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--repeat', type=int, default=30)
    parser.add_argument('--population-size', type=int, default=210)
    parser.add_argument('--budget-ratio', type=int, default=100)
    parser.add_argument('--adaptive-open', action="store_true", default=True)
    parser.add_argument('--early-stop', action="store_true", default=False)
    parser.add_argument('--use-space', type=int, choices=[0, 1, 2], default=2,
                         help="Which trained condition to evaluate: "
                              "0=NO_SPACE, 1=JUST_SIZES, 2=INSTANCE_STATE")
    args = parser.parse_known_args()[0]
    return args


def set_global_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)


@ray.remote
def step_in_env(args, model_path, run_idx):
    """
    Runs one evaluation episode of the trained SPACE-PPO model on the
    problem specified by args.key. The model is loaded inside the remote
    function (not passed in as an already-loaded object) since SB3 models
    aren't guaranteed to pickle cleanly across Ray worker processes.
    """
    model = PPO.load(model_path)
    np.random.seed(args.seed + run_idx)
    random.seed(args.seed + run_idx)

    env = SPaceEnv(
        key=args.key,
        use_space=space_operation.NO_SPACE.value,
        budget_ratio=args.budget_ratio,
        population_size=args.population_size,
        save_history=True,
        adaptive_open=args.adaptive_open,
        early_stop=args.early_stop,
    )
    # Evaluation always sees the full instance set, in round-robin order --
    # SPACE curriculum shaping applies to training only. Forcing
    # NO_SPACE above means reset() falls through to MamoBase's own
    # round-robin, same as the PPO baseline's eval env.

    obs, _ = env.reset()

    terminated = False
    truncated = False
    info = {}
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
    env.close()
    return info


def space_run_baseline(args, model_path):
    save_path = f'./results/space/space{args.use_space}/'
    if not os.path.exists(save_path):
        os.umask(0)
        os.makedirs(save_path, mode=0o777)
    np.random.seed(args.seed)
    random.seed(args.seed)

    info = ray.get([step_in_env.remote(args, model_path, i)
                for i in range(args.repeat)])
    np.savez(
        f'{save_path}{args.key}_sd{args.seed}_rp{args.repeat}.npz',
        info_stack=info)


if __name__ == "__main__":
    args = get_args()

    training_key = args.key
    condition_dir = f"space{args.use_space}"
    model_path = os.path.join(
        "space", "results", training_key, condition_dir, "ppo_model"
    )

    set_global_seeds(args.seed)
    task = Task.get_task(name="all" + args.key.split("_")[-1])
    ray.init(num_cpus=args.repeat)
    for t in task:
        args.key = t
        space_run_baseline(args, model_path)
        print("===== Finish " + args.key + " =====")