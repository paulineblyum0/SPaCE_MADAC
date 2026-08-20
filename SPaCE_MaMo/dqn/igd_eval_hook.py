"""
Periodic IGD evaluation for Tianshou DQN training.
"""

import argparse

import numpy as np

from mamo.mamo_register import Task, get_maenv
from dqn.test_dqn import run_repeats


class DQNIGDEvalHook:
    def __init__(self, policy, args, eval_freq_early=5000, eval_freq_late=10000,
                 switch_step=100000, n_repeats=10, base_train_fn=None, save_path=None):
        self.policy = policy
        self.args = args
        self.eval_freq_early = eval_freq_early
        self.eval_freq_late = eval_freq_late
        self.switch_step = switch_step
        self.n_repeats = n_repeats
        self.base_train_fn = base_train_fn
        self.save_path = save_path

        self._last_eval_step = 0
        self._steps = []

        # loops through the three training functions
        func_list, nobjs_list = get_maenv(args.key)
        self.tasks = [f"{f}_{n}" for f in func_list for n in nobjs_list]
        self._history = {
            t: {"best_mean": [], "best_std": [], "last_mean": [], "last_std": []}
            for t in self.tasks
        }

    def _current_eval_freq(self, env_step):
        # finer resolution earlier on
        return self.eval_freq_early if env_step < self.switch_step else self.eval_freq_late

    def _eval_all_problems(self):
        # per-problem args copies
        self.policy.eval()
        for t in self.tasks:
            args_t = argparse.Namespace(**{**vars(self.args), "key": t})
            info = run_repeats(args_t, self.policy, self.n_repeats)
            best_igd = np.array([i["best_igd"] for i in info])
            last_igd = np.array([i["last_igd"] for i in info])
            self._history[t]["best_mean"].append(best_igd.mean())
            self._history[t]["best_std"].append(best_igd.std())
            self._history[t]["last_mean"].append(last_igd.mean())
            self._history[t]["last_std"].append(last_igd.std())
        self.policy.train()

    def __call__(self, epoch, env_step):
        # preserve existing epsilon-annealing behaviour
        if self.base_train_fn is not None:
            self.base_train_fn(epoch, env_step)

        if env_step - self._last_eval_step >= self._current_eval_freq(env_step):
            self._last_eval_step = env_step
            self._steps.append(env_step)
            self._eval_all_problems()
            if self.save_path is not None:
                flat = {
                    f"{t}_{metric}": np.array(values)
                    for t, hist in self._history.items()
                    for metric, values in hist.items()
                }
                np.savez(self.save_path, steps=np.array(self._steps), **flat)