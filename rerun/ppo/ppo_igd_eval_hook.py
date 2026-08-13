"""


Produces the mid-training IGD curve. Scope is the THREE training problems
for the key (get_maenv -> e.g. M_2_46_3 -> DTLZ2_3, WFG4_3, WFG6_3), not
the full 8-problem M=3 suite: the curve is a training diagnostic, and the
held-out problems are covered by the final 30-repeat evaluation in
test_ppo.py. (dqn/igd_eval_hook.py still evaluates all 8 via
Task.get_task; the three needed here are a subset of those, so the two
curves remain directly comparable per-problem.)

Mechanism differs from the DQN version in one way that is NOT
interchangeable:

  Hook point. Tianshou's train_fn(epoch, env_step) has no SB3 equivalent.
  SB3's BaseCallback._on_step() is called once per n_envs steps collected,
  and self.num_timesteps increments by n_envs each call (not by 1). The
  threshold-crossing check below accounts for that stride the same way
  train_fn's did for step_per_collect=32.

  Firing mid-rollout is safe: PPO's weights are frozen during rollout
  collection and only updated in train() at _on_rollout_end, so an eval
  fired at any point between rollout boundaries is evaluating a static
  policy, not one that is changing underneath it.

Run from rerun/ -- imported by ppo/ppo.py, not run standalone.
"""
import os
import argparse

import numpy as np
import ray
from stable_baselines3.common.callbacks import BaseCallback

from mamo.mamo_register import get_maenv
from ppo.test_ppo import run_repeats


class PPOIGDEvalHook(BaseCallback):
    """
    Fires an IGD evaluation once self.num_timesteps has crossed each
    eval-freq-step threshold (after the threshold, not necessarily on it,
    same as the DQN hook). Cadence is two-phase, matching
    DQNIGDEvalHook._current_eval_freq: eval_freq_early while
    num_timesteps < switch_step, eval_freq_late afterward -- learning is
    typically front-loaded, so the tail doesn't need as fine a grid.

    Problems are evaluated consecutively; only the n_repeats repeats
    within a single problem are Ray-parallel.
    """

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

        # save_history=False is load-bearing, not cosmetic -- see the
        # memory contract in the module docstring.
        self.eval_args = argparse.Namespace(
            key=train_args.key,
            seed=train_args.seed,
            budget_ratio=getattr(train_args, "budget_ratio", 100),
            population_size=getattr(train_args, "population_size", 210),
            adaptive_open=getattr(train_args, "adaptive_open", True),
            early_stop=getattr(train_args, "early_stop", False),
            pi_arch=getattr(train_args, "pi_arch", [64, 64]),
            vf_arch=getattr(train_args, "vf_arch", [64, 64]),
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
        # same early/late split as DQNIGDEvalHook._current_eval_freq,
        # keyed on self.num_timesteps instead of env_step
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
            # Explicit: without this the final problem's infos stay bound
            # across the savez below.
            del infos
            if self.verbose:
                print(f"[IGD eval @ {self.num_timesteps}] {t} done", flush=True)

        del policy_ref
        self._save_curve()

    def _save_curve(self):
        # Rewritten in full on every firing (same convention as the DQN
        # hook's igd_curve.npz) -- affordable because the payload is a few
        # thousand floats, not episode histories. Real float arrays, not
        # an object-pickled ragged list.
        np.savez(
            self.curve_path,
            steps=np.asarray(self._steps, dtype=np.int64),          # (n_firings,)
            problems=np.asarray(self._task),                        # (n_problems,)
            best=np.asarray([self._best[t] for t in self._task]),   # (n_problems, n_firings, n_repeats)
            last=np.asarray([self._last[t] for t in self._task]),
        )

    def _on_training_end(self) -> None:
        # Guard against an empty final write clobbering a good curve if
        # training ends before any eval fired.
        if self._steps:
            self._save_curve()
        if self._ray_owns_init:
            ray.shutdown()


def load_curve(path):
    """
    Reads igd_curve.npz and returns {problem: {"steps", "best_mean",
    "best_std", "last_mean", "last_std", "best_raw", "last_raw"}}.

    mean/std are over the n_repeats evaluation repeats at a fixed training
    seed -- i.e. evaluation stochasticity of MOEA/D under a frozen policy,
    NOT training variance across seeds. To get the latter, take the
    per-seed mean first and then std over seeds.
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