import os
import argparse
import random
import numpy as np
import ray

from stable_baselines3 import PPO

from mamo.mamo_register import Task
from ppo.ppo_env import PPOMOEAEnv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, default='M_2_46_3')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--repeat', type=int, default=30)
    parser.add_argument('--budget-ratio', type=int, default=100)
    parser.add_argument('--n-ref-points', type=int, default=1000)
    parser.add_argument('--population-size', type=int, default=210)
    parser.add_argument('--save-history', action="store_true", default=False)
    args = parser.parse_known_args()[0]
    return args


def set_global_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)


@ray.remote
def step_in_env(args, model_path):
    """
    Runs one evaluation episode of the trained PPO model on the problem
    specified by args.key. The model is loaded inside the remote function
    (not passed in as an already-loaded object) since SB3 models aren't
    guaranteed to pickle cleanly across Ray worker processes.
    """
    model = PPO.load(model_path)
    env = PPOMOEAEnv(
        key=args.key,
        seed=args.seed,
        budget_ratio=args.budget_ratio,
        n_ref_points=args.n_ref_points,
        population_size=args.population_size,
        save_history=args.save_history,
    )
    obs, _ = env.reset(seed=args.seed)

    terminated = False
    truncated = False
    info = {}
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
    env.close()
    return info


def ppo_run_baseline(args, model_path):
    save_path = './results/ppo/'
    if not os.path.exists(save_path):
        os.umask(0)
        os.makedirs(save_path, mode=0o777)
    np.random.seed(args.seed)
    random.seed(args.seed)

    info = ray.get([step_in_env.remote(args, model_path)
                    for _ in range(args.repeat)])
    np.savez(
        f'{save_path}{args.key}_sd{args.seed}_rp{args.repeat}.npz',
        info_stack=info)


if __name__ == "__main__":
    args = get_args()
    policy_name = args.key
    model_path = os.path.join("ppo", "results", policy_name, "ppo_model")
    set_global_seeds(args.seed)
    task = Task.get_task(name="all" + args.key.split("_")[-1])
    ray.init(num_cpus=args.repeat)
    for t in task:
        args.key = t
        ppo_run_baseline(args, model_path)
        print("===== Finish " + args.key + " =====")