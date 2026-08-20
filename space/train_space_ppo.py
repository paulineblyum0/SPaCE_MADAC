# space/train_space_ppo.py
"""
SPACE-curriculum PPO training for MaMo.

Run from rerun/ directory:
    python -m space.train_space_ppo --key M_2_46_3 --use-space 2

Mirrors ppo/ppo.py as closely as possible so the two are a fair comparison --
same PPO hyperparameters, same key/adaptive-open/early-stop/budget-ratio
handling. The two deliberate differences are:
  1. n_envs is forced to 1 with DummyVecEnv (not configurable here) --
     UpdateEnvCallback reads/writes curriculum state via
     training_env.envs[0].unwrapped, which assumes a single environment.
  2. --use-space / --instance-ordering select the SPACE condition and are
     wired into SPaceEnv + UpdateEnvCallback.

Results are saved to space/results/<key>/
"""

import argparse
import logging
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from space.space_env import SPaceEnv
from space.space_callback import UpdateEnvCallback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, default="M_2_46_3")
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population-size", type=int, default=210)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--adaptive-open", action="store_true", default=True)
    parser.add_argument("--early-stop", action="store_true", default=False)
    parser.add_argument("--budget-ratio", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pi-arch", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--vf-arch", type=int, nargs="+", default=[64, 64])

    # SPACE-specific
    parser.add_argument(
        "--use-space", type=int, choices=[0, 1, 2], default=2,
        help="0=NO_SPACE (round-robin baseline), 1=JUST_SIZES (ablation), "
             "2=INSTANCE_STATE (full SPACE)",
    )
    parser.add_argument(
        "--instance-ordering", type=int, choices=[0, 1, 2], default=0,
        help="0=ABSOLUTE, 1=IMPROVEMENT, 2=RELATIVE_IMPROVEMENT",
    )
    parser.add_argument("--stability-threshold", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.1)

    return parser.parse_args()


def main():
    args = parse_args()

    condition_dir = f"space{args.use_space}"
    results_dir = os.path.join("space", "results", args.key, condition_dir)
    os.makedirs(results_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(results_dir, "space_debug.log")),
        ],
    )

    if args.check:
        print("Running environment checker...")
        env = SPaceEnv(
            key=args.key, use_space=args.use_space, seed=args.seed,
            adaptive_open=args.adaptive_open, budget_ratio=args.budget_ratio,
        )
        check_env(env, warn=True)
        print("check_env passed.")
        env.close()
        return

    # n_envs is forced to 1 (DummyVecEnv) -- see module docstring.
    env = make_vec_env(
        lambda: SPaceEnv(
            key=args.key,
            use_space=args.use_space,
            adaptive_open=args.adaptive_open,
            budget_ratio=args.budget_ratio,
            early_stop=args.early_stop,
        ),
        n_envs=1,
        seed=args.seed,
        monitor_dir=results_dir,
        monitor_kwargs={"info_keywords": ("best_igd", "last_igd")},
        vec_env_cls=DummyVecEnv,
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

    space_callback = UpdateEnvCallback(
        use_space_val=args.use_space,
        instance_ordering_val=args.instance_ordering,
        stability_threshold=args.stability_threshold,
        eta_const=args.eta,
    )

    print(
        f"Training SPACE-PPO on {args.key} for {args.timesteps:,} steps "
        f"(use_space={args.use_space}, instance_ordering={args.instance_ordering})..."
    )
    model.learn(total_timesteps=args.timesteps, callback=[space_callback])

    model_path = os.path.join(results_dir, "ppo_model")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()