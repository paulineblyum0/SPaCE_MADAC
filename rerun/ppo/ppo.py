"""
PPO baseline for MaMo.

Run from rerun/ directory:
    python -m ppo.ppo --key M_2_46_3

Results are saved to ppo/results/<key>/
"""

import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from ppo.ppo_env import PPOMOEAEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, default="M_2_46_3",
                        help="MaMo instance key, e.g. M_2_46_3")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total environment steps for training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true",
                        help="Run SB3 env checker then exit")
    return parser.parse_args()


def main():
    args = parse_args()

    env = PPOMOEAEnv(key=args.key, seed=args.seed)

    if args.check:
        print("Running environment checker...")
        check_env(env, warn=True)
        print("check_env passed.")
        env.close()
        return

    results_dir = os.path.join("ppo", "results", args.key)
    os.makedirs(results_dir, exist_ok=True)
    env = Monitor(env, filename=os.path.join(results_dir, "monitor"))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=os.path.join(results_dir, "tb"),
    )

    print(f"Training PPO on {args.key} for {args.timesteps:,} steps...")
    model.learn(total_timesteps=args.timesteps)

    model_path = os.path.join(results_dir, "ppo_model")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()