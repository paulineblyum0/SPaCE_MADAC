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
    parser.add_argument("--key", type=str, default="M_2_46_3",
                        help="MaMo instance key, e.g. M_2_46_3")
    parser.add_argument("--timesteps", type=int, default=400_000,
                        help="Total environment steps for training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true",
                        help="Run SB3 env checker then exit")
    parser.add_argument("--adaptive-open", action="store_true", default=False,
                     help="Enable adaptive weight adjustment (agent 3)")
    parser.add_argument('--budget-ratio', type=int, default=100)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (SB3 default 3e-4)")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient (SB3 default 0.0; tianshou default 0.01)") #changed from SB3 defailt (0.0) to tianshou default (0.01)
    parser.add_argument("--normalize", action="store_true", default=False,
                        help="Wrap env in VecNormalize (obs + reward normalization)")
    parser.add_argument("--n-steps", type=int, default=2048,
                    help="Rollout length per env before each PPO update (SB3 default 2048)")
    parser.add_argument("--batch-size", type=int, default=64,
                    help="Minibatch size for PPO update (SB3 default 64)")
    return parser.parse_args()


def main():
    args = parse_args()

    env = PPOMOEAEnv(key=args.key, seed=args.seed, adaptive_open=args.adaptive_open, budget_ratio=args.budget_ratio)

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
                            budget_ratio=args.budget_ratio),
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=results_dir,
        monitor_kwargs={"info_keywords": ("best_igd", "last_igd")},
        vec_env_cls=SubprocVecEnv,
    )

    if args.normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        tensorboard_log=os.path.join(results_dir, "tb"),
    )

    print(f"Training PPO on {args.key} for {args.timesteps:,} steps...")
    model.learn(total_timesteps=args.timesteps)

    model_path = os.path.join(results_dir, "ppo_model")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    if args.normalize:
        stats_path = os.path.join(results_dir, "vecnormalize.pkl")
        env.save(stats_path)
        print(f"VecNormalize stats saved to {stats_path}")
    
    env.close()


if __name__ == "__main__":
    main()