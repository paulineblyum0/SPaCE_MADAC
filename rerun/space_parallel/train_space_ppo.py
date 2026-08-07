# space_parallel/train_space_ppo.py
"""
SPACE-curriculum PPO training for MaMo, n_envs=16 (SubprocVecEnv).

Run from rerun/ directory:
    python -m space_parallel.train_space_ppo --key M_2_46_3 --use-space 2

Exists side-by-side with space/train_space_ppo.py (the n_envs=1 version) --
neither package touches the other's files. See space_parallel/space_env.py
and space_parallel/space_callback.py module docstrings for what had to
change architecturally to make SPACE's curriculum logic safe under
subprocess parallelism.

Results are saved to space_parallel/results/<key>/space<use_space>/
(mirrors the space/results/<key>/space{0,1,2}/ layout already fixed in the
n_envs=1 package, so both trees can be diffed/compared directory-for-
directory).
"""

import argparse
import logging
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from space_parallel.space_env import SPaceEnvParallel
from space_parallel.space_callback import UpdateEnvCallbackParallel


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
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pi-arch", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--vf-arch", type=int, nargs="+", default=[64, 64])

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

ORDERING_NAMES = {0: "absolute", 1: "improvement", 2: "relative_improvement"}

def main():
    args = parse_args()

    if args.check:
        print("Running environment checker...")
        env = SPaceEnvParallel(
            key=args.key, use_space=args.use_space, seed=args.seed,
            adaptive_open=args.adaptive_open, budget_ratio=args.budget_ratio,
        )
        check_env(env, warn=True)
        print("check_env passed.")
        env.close()
        return

    if args.use_space == 2:  # INSTANCE_STATE — ordering choice actually matters
        ordering_tag = ORDERING_NAMES[args.instance_ordering]
        results_dir = os.path.join(
            "space_parallel", "results", args.key,
            f"space{args.use_space}_{ordering_tag}",
        )
    else:
        results_dir = os.path.join(
            "space_parallel", "results", args.key, f"space{args.use_space}"
        )
    os.makedirs(results_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(results_dir, "space_debug.log")),
        ],
    )

    env = make_vec_env(
        lambda: SPaceEnvParallel(
            key=args.key,
            use_space=args.use_space,
            adaptive_open=args.adaptive_open,
            budget_ratio=args.budget_ratio,
            early_stop=args.early_stop,
        ),
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

    space_callback = UpdateEnvCallbackParallel(
        key=args.key,
        use_space_val=args.use_space,
        instance_ordering_val=args.instance_ordering,
        stability_threshold=args.stability_threshold,
        eta_const=args.eta,
        probe_env_kwargs=dict(
            adaptive_open=args.adaptive_open,
            budget_ratio=args.budget_ratio,
            early_stop=args.early_stop,
        ),
    )

    print(
        f"Training SPACE-PPO (parallel, n_envs={args.n_envs}) on {args.key} "
        f"for {args.timesteps:,} steps (use_space={args.use_space}, "
        f"instance_ordering={args.instance_ordering})..."
    )
    model.learn(total_timesteps=args.timesteps, callback=[space_callback])

    model_path = os.path.join(results_dir, "ppo_model")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()