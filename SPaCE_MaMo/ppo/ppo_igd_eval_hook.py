"""
Produces the mid-training IGD curve evaluated over the training problems only.
"""
import os

import numpy as np
import ray
from stable_baselines3.common.callbacks import BaseCallback

from mamo.mamo_register import get_maenv
from ppo.test_ppo import run_repeats
from ppo.eval_args_util import build_eval_args


class PPOIGDEvalHook(BaseCallback):
    
    #fire eval after num_timesteps has crossed frequency threshold
    def __init__(self, train_args, eval_freq_early=5000, eval_freq_late=10000,
                 switch_step=100000, n_repeats=10, curve_path=None, verbose=0):
        super().__init__(verbose)

        self.eval_freq_early = eval_freq_early
        self.eval_freq_late = eval_freq_late
        self.switch_step = switch_step
        self.n_repeats = n_repeats

        seed_dir = f"seed_{train_args.seed}"
        self.curve_path = curve_path or os.path.join(
            "ppo", "results", train_args.key, seed_dir, "igd_curve.npz")

        # keep all args consisten with ppo training
        self.eval_args = build_eval_args(train_args)

        self._last_eval_step = 0

        func_list, nobjs_list = get_maenv(train_args.key)
        self._task = [f"{f}_{n}" for f in func_list for n in nobjs_list]

        # Raw per-repeat scalars (n_firings, n_repeats).
        self._steps = []
        self._best = {t: [] for t in self._task}
        self._last = {t: [] for t in self._task}

        self._ray_owns_init = False

    def _current_eval_freq(self):
        # finer resolution earlier on
        return (self.eval_freq_early if self.num_timesteps < self.switch_step
                else self.eval_freq_late)

    def _init_callback(self) -> None:
        os.makedirs(os.path.dirname(self.curve_path), exist_ok=True)
        if not ray.is_initialized():
            ray.init(num_cpus=self.n_repeats)
            self._ray_owns_init = True

    def _on_step(self) -> bool:
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
        # Save averages and sd for each task. Full save history removed since not plotting within episode curves
        np.savez(
            self.curve_path,
            steps=np.asarray(self._steps, dtype=np.int64), # (n_firings,)
            problems=np.asarray(self._task), # (n_problems,)
            best=np.asarray([self._best[t] for t in self._task]), # (n_problems, n_firings, n_repeats)
            last=np.asarray([self._last[t] for t in self._task]),
        )

    def _on_training_end(self) -> None:
        # Guard against an empty final write overwriting a good curve if training ends before any eval fired.
        if self._steps:
            self._save_curve()
        if self._ray_owns_init:
            ray.shutdown()

