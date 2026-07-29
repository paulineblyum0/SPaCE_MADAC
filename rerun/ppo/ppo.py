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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from ppo.ppo_env import PPOMOEAEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, default="M_2_46_3")
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--population-size', type=int, default=210)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--adaptive-open", action="store_true", default=True)
    parser.add_argument("--early-stop", action="store_true", default=False)
    parser.add_argument('--budget-ratio', type=int, default=100)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pi-arch", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--vf-arch", type=int, nargs="+", default=[64, 64])
    return parser.parse_args()


def main():
    args = parse_args()
 
    env = PPOMOEAEnv(key=args.key, seed=args.seed, adaptive_open=args.adaptive_open,
                      budget_ratio=args.budget_ratio)
 
    if args.check:
        print("Running environment checker...")
        check_env(env, warn=True)
        print("check_env passed.")
        env.close()
        return
 
    results_dir = os.path.join("ppo", "results", args.key)
    os.makedirs(results_dir, exist_ok=True)
    env = make_vec_env(
        lambda: PPOMOEAEnv(key=args.key, adaptive_open=args.adaptive_open,
                            budget_ratio=args.budget_ratio, early_stop=args.early_stop),
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=results_dir,
        monitor_kwargs={"info_keywords": ("best_igd", "last_igd")},
        vec_env_cls=SubprocVecEnv,
    )
 
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        policy_kwargs=dict(net_arch=dict(pi=args.pi_arch, vf=args.vf_arch)),
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