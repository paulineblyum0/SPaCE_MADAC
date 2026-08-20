"""
SPACE-curriculum PPO training for MaMo.
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
from space_parallel.space_igd_eval_hook import SpaceIGDEvalHook


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, default="M_2_46_3")
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population-size", type=int, default=210)
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
    # 0=NO_SPACE, 1=JUST_SIZES, 2=INSTANCE_STATE (full SPACE)
    parser.add_argument("--use-space", type=int, choices=[0, 1, 2], default=2)
    # 0=ABSOLUTE, 1=IMPROVEMENT, 2=RELATIVE_IMPROVEMENT
    parser.add_argument("--instance-ordering", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--igd-eval-freq-early", type=int, default=5000)
    parser.add_argument("--igd-eval-freq-late", type=int, default=10000)
    parser.add_argument("--igd-eval-switch-step", type=int, default=100_000)
    parser.add_argument("--igd-eval-repeats", type=int, default=10)

    return parser.parse_args()

ORDERING_NAMES = {0: "absolute", 1: "improvement", 2: "relative_improvement"}

def main():
    args = parse_args()

    # ordering only matters for full space
    if args.use_space == 2: 
        ordering_tag = ORDERING_NAMES[args.instance_ordering]
        results_dir = os.path.join(
            "results", "space_parallel", "trained", args.key,
            f"space{args.use_space}_{ordering_tag}",
            f"seed_{args.seed}"
        )
    else:
        results_dir = os.path.join(
            "results", "space_parallel", "trained", args.key, f"space{args.use_space}",
            f"seed_{args.seed}"
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
        # Log best/last IGD per episode
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

    callbacks = [
        UpdateEnvCallbackParallel(
            key=args.key,
            use_space_val=args.use_space,
            instance_ordering_val=args.instance_ordering,
            eta_const=args.eta,
            probe_env_kwargs=dict(
                adaptive_open=args.adaptive_open,
                budget_ratio=args.budget_ratio,
                early_stop=args.early_stop,
            ),
        )
    ]

    if args.igd_eval_freq_early > 0:
        callbacks.append(SpaceIGDEvalHook(
            train_args=args,
            results_dir=results_dir,
            eval_freq_early=args.igd_eval_freq_early,
            eval_freq_late=args.igd_eval_freq_late,
            switch_step=args.igd_eval_switch_step,
            n_repeats=args.igd_eval_repeats,
        ))

    print(
        f"Training SPACE-PPO (parallel, n_envs={args.n_envs}) on {args.key} "
        f"for {args.timesteps:,} steps (use_space={args.use_space}, "
        f"instance_ordering={args.instance_ordering})..."
    )
    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    model_path = os.path.join(results_dir, "ppo_model")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()