"""
PPO baseline for MaMo.
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import ray

from ppo.test_ppo import run_repeats
from mamo.mamo_register import get_maenv
from ppo.ppo_env import PPOMOEAEnv
from ppo.ppo_igd_eval_hook import PPOIGDEvalHook
from ppo.eval_args_util import build_eval_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, default="M_2_46_3")
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument('--population-size', type=int, default=210)
    parser.add_argument("--adaptive-open", action="store_true", default=True) #locked true so can't accidentally disable
    parser.add_argument("--early-stop", action="store_true", default=False)
    parser.add_argument('--budget-ratio', type=int, default=100)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pi-arch", type=int, nargs="+", default=[64, 64]) #change network width
    parser.add_argument("--vf-arch", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--igd-eval-freq-early", type=int, default=5000)
    parser.add_argument("--igd-eval-freq-late", type=int, default=10000)
    parser.add_argument("--igd-eval-switch-step", type=int, default=100_000)
    parser.add_argument("--igd-eval-repeats", type=int, default=10,)
    parser.add_argument("--checkpoint-every", type=int, default=100_000,)
    parser.add_argument("--eval-t0", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    results_dir = os.path.join("results", "ppo", "trained", args.key, f"seed_{args.seed}")
    os.makedirs(results_dir, exist_ok=True)

    env = make_vec_env(
        lambda: PPOMOEAEnv(key=args.key, adaptive_open=args.adaptive_open,
                           budget_ratio=args.budget_ratio, early_stop=args.early_stop,
                           population_size=args.population_size),
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
    print(model)
    print(f"Training PPO on {args.key} for {args.timesteps:,} steps...")

    # last minute addition in order to add eval at t0 to the training curve
    if args.eval_t0:
        t0_ckpt = os.path.join(results_dir, "_t0_checkpoint")
        model.save(t0_ckpt)

        eval_args = build_eval_args(args)

        ray.init(num_cpus=10)
        policy_ref = ray.put(model.policy.state_dict())
        flat = {}
        func_list, nobjs_list = get_maenv(args.key)
        for t in [f"{f}_{n}" for f in func_list for n in nobjs_list]:
            eval_args.key = t
            info = run_repeats(eval_args, policy_ref, 10)
            best = np.array([i["best_igd"] for i in info])
            last = np.array([i["last_igd"] for i in info])
            flat[f"{t}_best_mean"] = np.array([best.mean()])
            flat[f"{t}_best_std"]  = np.array([best.std()])
            flat[f"{t}_last_mean"] = np.array([last.mean()])
            flat[f"{t}_last_std"]  = np.array([last.std()])
        del policy_ref
        ray.shutdown()
        #shuts down training - only used for eval at t0 with no other training
        np.savez(os.path.join(results_dir, "igd_curve_t0.npz"), steps=np.array([0]), **flat)
        print(f"t=0 eval saved to {os.path.join(results_dir, 'igd_curve_t0.npz')}")
        env.close()
        return

    callbacks = []

    if args.checkpoint_every > 0:
        # save_freq counts per-env steps, so divide by n_envs to get the requested interval in total timesteps.
        callbacks.append(CheckpointCallback(
            save_freq=max(1, args.checkpoint_every // args.n_envs),
            save_path=os.path.join(results_dir, "checkpoints"),
            name_prefix="ppo",
        ))

    if args.igd_eval_freq_early > 0:
        callbacks.append(PPOIGDEvalHook(
            train_args=args,
            eval_freq_early=args.igd_eval_freq_early,
            eval_freq_late=args.igd_eval_freq_late,
            switch_step=args.igd_eval_switch_step,
            n_repeats=args.igd_eval_repeats,
        ))

    model.learn(total_timesteps=args.timesteps,
                callback=callbacks if callbacks else None)

    model_path = os.path.join(results_dir, "ppo_model")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()