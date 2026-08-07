import os
import argparse
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from ppo.ppo_env import PPOMOEAEnv


def parse_args():
    parser = argparse.ArgumentParser()
    # Identical env args to match ppo.py
    parser.add_argument("--key", type=str, default="M_2_46_3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population-size", type=int, default=210)
    parser.add_argument("--adaptive-open", action="store_true", default=True)
    parser.add_argument("--early-stop", action="store_true", default=False)
    parser.add_argument("--budget-ratio", type=int, default=100)
    parser.add_argument("--n-envs", type=int, default=16)

    # 4-hour budget settings
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--trial-timesteps", type=int, default=20_000)

    # IGD eval settings
    parser.add_argument("--eval-freq", type=int, default=2500,
                         help="Total env steps (across all envs) between IGD evals.")
    parser.add_argument("--n-eval-seeds", type=int, default=5,
                         help="Per paper: >=5 seeds recommended to avoid overtuning to a single seed.")
    return parser.parse_args()


class IGDEvalCallback(BaseCallback):
    """
    Evaluates the current policy by actually rolling out deterministic episodes
    and reading info['best_igd'] from PPOMOEAEnv — the same field test_ppo.py
    uses for final results. Reports the mean IGD (lower = better) to Optuna
    for pruning, instead of SB3's shaped reward.

    Eval seeds are fixed and disjoint from args.seed / args.seed+100 used for
    training/env construction, so trial-to-trial comparisons aren't just
    single-seed noise (per Eimer et al.'s tuning/test seed separation point).
    """

    def __init__(self, env_kwargs, trial, eval_freq=2500, n_eval_seeds=5,
                 eval_seed_base=100_000, verbose=0):
        super().__init__(verbose)
        self.env_kwargs = env_kwargs
        self.trial = trial
        self.eval_freq = eval_freq
        self.n_eval_seeds = n_eval_seeds
        self.eval_seed_base = eval_seed_base
        self.is_pruned = False
        self.last_mean_igd = np.inf

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.last_mean_igd = self._run_igd_eval()

            self.trial.report(self.last_mean_igd, self.n_calls)
            if self.trial.should_prune():
                self.is_pruned = True
                return False  # kills trial early
        return True

    def _run_igd_eval(self):
        igds = []
        for i in range(self.n_eval_seeds):
            env = PPOMOEAEnv(**self.env_kwargs)
            seed = self.eval_seed_base + i
            obs, _ = env.reset(seed=seed)
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
            env.close()
            igds.append(info["best_igd"])
        return float(np.mean(igds))


def sample_ppo_params(trial: optuna.Trial):
    """Focuses search space on high-impact stability and exploration parameters."""
    n_steps_exp = trial.suggest_int("n_steps_exp", 7, 11)      # 128 to 2048
    batch_size_exp = trial.suggest_int("batch_size_exp", 4, 8)  # 16 to 256

    n_steps = 2 ** n_steps_exp
    batch_size = 2 ** batch_size_exp

    if batch_size > n_steps:
        batch_size = n_steps

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 1e-6, 1e-2, log=True),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "n_epochs": trial.suggest_int("n_epochs", 3, 10),
    }


def objective(trial: optuna.Trial, args):
    params = sample_ppo_params(trial)

    env_kwargs = dict(
        key=args.key,
        adaptive_open=args.adaptive_open,
        budget_ratio=args.budget_ratio,
        early_stop=args.early_stop,
        population_size=args.population_size,
    )

    # n_envs-way parallel training, same architecture as ppo.py.
    # NOTE: if this config is meant to carry over to SPACE (n_envs=1,
    # DummyVecEnv), n_steps/batch_size found here are tuned against a
    # rollout buffer of size n_steps * args.n_envs and are not guaranteed
    # to transfer to n_envs=1 unscaled. Re-tune with --n-envs 1 for that case.
    train_env = make_vec_env(
        lambda: PPOMOEAEnv(**env_kwargs),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )

    model = PPO("MlpPolicy", train_env, verbose=0, seed=args.seed, **params)

    igd_callback = IGDEvalCallback(
        env_kwargs=env_kwargs,
        trial=trial,
        eval_freq=max(1, args.eval_freq // args.n_envs),
        n_eval_seeds=args.n_eval_seeds,
    )

    try:
        model.learn(total_timesteps=args.trial_timesteps, callback=igd_callback)
    except Exception:
        train_env.close()
        raise optuna.TrialPruned()

    train_env.close()

    if igd_callback.is_pruned:
        raise optuna.TrialPruned()

    # Fall back to a final IGD eval if we never hit eval_freq (short trials)
    if not np.isfinite(igd_callback.last_mean_igd) or igd_callback.last_mean_igd == np.inf:
        igd_callback.last_mean_igd = igd_callback._run_igd_eval()

    return igd_callback.last_mean_igd


if __name__ == "__main__":
    args = parse_args()

    # Lower IGD is better -> minimize, not maximize.
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=5000),
    )

    print(f"--- Starting Optuna Study ({args.n_trials} Trials, "
          f"{args.trial_timesteps:,} steps max/trial, "
          f"{args.n_eval_seeds} eval seeds, objective=mean best_igd) ---")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print("\n==========================================")
    print("      BEST HYPERPARAMETERS FOUND          ")
    print("==========================================")
    print("Best Trial Mean IGD:", study.best_value)
    print("\nCopy these directly into your ppo.py CLI arguments:")
    for k, v in study.best_params.items():
        print(f"  --{k}: {v}")

    # Reproducibility record (per Eimer et al.'s checklist — worth keeping
    # verbatim for your dissertation's methodology section)
    print("\n--- For methodology write-up ---")
    print(f"Tuning method: Optuna TPE + MedianPruner")
    print(f"Search space: n_steps in [128,2048] (log2), batch_size in [16,256] (log2),"
          f" lr in [1e-5,1e-3] (log), ent_coef in [1e-6,1e-2] (log),"
          f" clip_range in [0.1,0.3], n_epochs in [3,10]")
    print(f"Budget: {args.n_trials} trials x {args.trial_timesteps} timesteps")
    print(f"Objective: mean best_igd over {args.n_eval_seeds} eval seeds"
          f" (seed base {100_000}), evaluated every {args.eval_freq} steps")
    print(f"Training seed: {args.seed} (disjoint from eval seeds)")