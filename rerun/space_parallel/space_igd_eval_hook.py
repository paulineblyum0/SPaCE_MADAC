# space_parallel/space_igd_eval_hook.py
"""
Periodic IGD evaluation for SPACE-curriculum PPO training.

train_args must carry the same env-behavior flags training used
(adaptive_open, budget_ratio, early_stop, population_size, pi_arch,
vf_arch) -- these do not transfer via the policy state_dict, and a
mismatch here silently invalidates the curve, per the same train/test
flag consistency requirement as the PPO hook and the final evaluation.
use_space and instance_ordering are NOT part of eval_args: the eval env
is always forced to NO_SPACE regardless of what the model trained under
(see above), so passing the training run's use_space/instance_ordering
through would be a no-op at best and misleading at worst.

Run from rerun/ -- imported by space_parallel/train_space_ppo.py, not run
standalone.
"""
import os
import argparse

import numpy as np
import ray
from stable_baselines3.common.callbacks import BaseCallback

from mamo.mamo_register import get_maenv
from space.space_enum import space_operation
from space_parallel.test_space import run_repeats


class SpaceIGDEvalHook(BaseCallback):
    """
    Fires an IGD evaluation once self.num_timesteps has crossed each
    eval-freq-step threshold
    """

    def __init__(self, train_args, results_dir, eval_freq_early=5000, eval_freq_late=10000,
                 switch_step=100000, n_repeats=10, curve_path=None, verbose=0):
        super().__init__(verbose)

        self.eval_freq_early = eval_freq_early
        self.eval_freq_late = eval_freq_late
        self.switch_step = switch_step
        self.n_repeats = n_repeats

        self.curve_path = curve_path or os.path.join(results_dir, "igd_curve.npz")

        # save_history=False is load-bearing, not cosmetic -- see the
        # memory contract in ppo/ppo_igd_eval_hook.py's module docstring.
        # use_space is fixed to NO_SPACE regardless of train_args.use_space
        # -- see module docstring.
        self.eval_args = argparse.Namespace(
            key=train_args.key,
            seed=train_args.seed,
            budget_ratio=getattr(train_args, "budget_ratio", 100),
            population_size=getattr(train_args, "population_size", 210),
            adaptive_open=getattr(train_args, "adaptive_open", True),
            early_stop=getattr(train_args, "early_stop", False),
            pi_arch=getattr(train_args, "pi_arch", [64, 64]),
            vf_arch=getattr(train_args, "vf_arch", [64, 64]),
            use_space=space_operation.NO_SPACE.value,
            save_history=False,
        )

        self._last_eval_step = 0

        func_list, nobjs_list = get_maenv(train_args.key)
        self._task = [f"{f}_{n}" for f in func_list for n in nobjs_list]

        # Raw per-repeat scalars. Each append is a list of n_repeats floats,
        # so self._best[t] ends up shape (n_firings, n_repeats).
        self._steps = []
        self._best = {t: [] for t in self._task}
        self._last = {t: [] for t in self._task}

        self._ray_owns_init = False

    def _current_eval_freq(self):
        return (self.eval_freq_early if self.num_timesteps < self.switch_step
                else self.eval_freq_late)

    def _init_callback(self) -> None:
        os.makedirs(os.path.dirname(self.curve_path), exist_ok=True)
        if not ray.is_initialized():
            ray.init(num_cpus=self.n_repeats)
            self._ray_owns_init = True

    def _on_step(self) -> bool:
        # Same rollout-boundary safety argument as the PPO hook: PPO's
        # weights are frozen during rollout collection and only updated at
        # _on_rollout_end, so firing here evaluates a static policy.
        # UpdateEnvCallbackParallel's curriculum updates also happen at
        # _on_rollout_end and only affect the training envs' instance
        # pool, not the policy weights -- so this hook firing mid-rollout
        # relative to a curriculum change is likewise not a race: the
        # eval env is unrelated to the training envs' curriculum state.
        if self.num_timesteps - self._last_eval_step >= self._current_eval_freq():
            self._last_eval_step = self.num_timesteps
            self._run_eval()
        return True

    def _run_eval(self):
        policy_ref = ray.put(self.model.policy.state_dict())
        self._steps.append(int(self.num_timesteps))

        for t in self._task:
            self.eval_args.key = t
            infos = run_repeats(self.eval_args, policy_ref, self.n_repeats)
            self._best[t].append([float(i["best_igd"]) for i in infos])
            self._last[t].append([float(i["last_igd"]) for i in infos])
            del infos
            if self.verbose:
                print(f"[IGD eval @ {self.num_timesteps}] {t} done", flush=True)

        del policy_ref
        self._save_curve()

    def _save_curve(self):
        np.savez(
            self.curve_path,
            steps=np.asarray(self._steps, dtype=np.int64),          # (n_firings,)
            problems=np.asarray(self._task),                        # (n_problems,)
            best=np.asarray([self._best[t] for t in self._task]),   # (n_problems, n_firings, n_repeats)
            last=np.asarray([self._last[t] for t in self._task]),
        )

    def _on_training_end(self) -> None:
        if self._steps:
            self._save_curve()
        if self._ray_owns_init:
            ray.shutdown()


def load_curve(path):
    """
    Identical contract to ppo.ppo_igd_eval_hook.load_curve -- see that
    docstring for the mean/std-over-repeats-not-seeds caveat.
    """
    d = np.load(path, allow_pickle=False)
    steps, problems, best, last = d["steps"], d["problems"], d["best"], d["last"]
    out = {}
    for i, t in enumerate(problems):
        out[str(t)] = {
            "steps": steps,
            "best_raw": best[i],
            "last_raw": last[i],
            "best_mean": best[i].mean(axis=1),
            "best_std": best[i].std(axis=1, ddof=1),
            "last_mean": last[i].mean(axis=1),
            "last_std": last[i].std(axis=1, ddof=1),
        }
    return out