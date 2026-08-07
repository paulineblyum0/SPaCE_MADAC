"""
Periodic IGD evaluation for SB3 PPO training.

Mirrors dqn/igd_eval_hook.py's design (early/late cadence split, all 8
problems in the training key's task suite, 10 Ray-parallel repeats per
problem, consecutive-per-problem, seed + run_idx seeding) but the
mechanism differs in two ways that are NOT interchangeable with the DQN
version:

1. Hook point: Tianshou's train_fn(epoch, env_step) has no SB3 equivalent.
   SB3's BaseCallback._on_step() is called once per n_envs=16 steps
   collected, and self.num_timesteps increments by n_envs each call (not
   by 1). The threshold-crossing check below accounts for that stride the
   same way train_fn's did for step_per_collect=32.

   Firing mid-rollout is safe: PPO's weights are frozen during rollout
   collection and only updated in train() at _on_rollout_end, so an eval
   fired at any point between rollout boundaries is evaluating a static
   policy, not one that's changing underneath it.

2. Serialization: unlike Tianshou's torch.nn.Module policies (picklable,
   passed directly into step_in_env.remote), SB3 models aren't guaranteed
   to pickle cleanly across Ray worker processes (see test_ppo.py). Every
   firing therefore saves a scratch checkpoint to disk and Ray workers
   PPO.load() it fresh -- an extra save + n_repeats loads per firing that
   the DQN version didn't need. In practice this is cheap relative to
   episode runtime: test_ppo.py's own final 30-repeat evaluation already
   does PPO.load() inside every one of its 30 remote workers, so the same
   pattern here at 10 workers per firing isn't a new cost class, just a
   fixed small tax on top of the eval itself.

We do NOT use SB3's built-in EvalCallback: it drives on evaluate_policy's
mean episode reward, not IGD, has no Ray parallelism, and evaluates a
single eval_env rather than looping the 8-problem suite. Same reasoning
that led to dropping Tianshou's built-in test_collector for DQN.

Run from rerun/ -- imported by ppo/ppo.py, not run standalone.
"""
import os
import argparse

import numpy as np
import ray
from stable_baselines3.common.callbacks import BaseCallback

from mamo.mamo_register import Task
from ppo.test_ppo import run_repeats


class PPOIGDEvalHook(BaseCallback):
    """
    Fires an IGD evaluation once self.num_timesteps has crossed each
    eval-freq-step threshold (after the threshold, not necessarily on it,
    same as the DQN hook). Cadence is two-phase, matching
    DQNIGDEvalHook._current_eval_freq: eval_freq_early while
    num_timesteps < switch_step, eval_freq_late afterward -- learning is
    typically front-loaded, so the tail doesn't need as fine a grid.

    Evaluates on all 8 problems in the training key's task suite,
    n_repeats Ray-parallel seeds per problem, problems evaluated
    consecutively.

    train_args must carry the same env-behavior flags PPO was trained
    with (adaptive_open, budget_ratio, early_stop, population_size) --
    these do not transfer automatically via PPO.load() and a mismatch
    here silently invalidates the curve, per the same train/test flag
    consistency requirement as the final 30-repeat evaluation. ppo.py's
    current parser carries all of these directly, so no getattr fallback
    is strictly needed, but they're kept as a defensive default matching
    test_ppo.py's own get_args() in case this hook is ever driven from a
    training script with a leaner arg set.
    """

    def __init__(self, train_args, eval_freq_early=5000, eval_freq_late=10000,
                 switch_step=100000, n_repeats=10,
                 checkpoint_path=None, curve_path=None, verbose=0):
        super().__init__(verbose)

        self.eval_freq_early = eval_freq_early
        self.eval_freq_late = eval_freq_late
        self.switch_step = switch_step
        self.n_repeats = n_repeats
        self.checkpoint_path = checkpoint_path or os.path.join(
            "ppo", "results", train_args.key, "_eval_checkpoint")
        self.curve_path = curve_path or os.path.join(
            "ppo", "results", train_args.key, "igd_curve.npz")

        # test_ppo.py's step_in_env only needs these five fields now --
        # n_ref_points is gone (PPOMOEAEnv/step_in_env no longer take it)
        # and save_history is hardcoded True inside step_in_env itself,
        # so it's not part of the args namespace at all anymore.
        self.eval_args = argparse.Namespace(
            key=train_args.key,
            seed=train_args.seed,
            budget_ratio=getattr(train_args, "budget_ratio", 100),
            population_size=getattr(train_args, "population_size", 210),
            adaptive_open=getattr(train_args, "adaptive_open", True),
            early_stop=getattr(train_args, "early_stop", False),
        )

        self._last_eval_step = 0
        self._task = Task.get_task(name="all" + train_args.key.split("_")[-1])
        self._history = []  # list of {"step": int, "results": {problem: info_list}}
        self._ray_owns_init = False

    def _current_eval_freq(self):
        # same early/late split as DQNIGDEvalHook._current_eval_freq,
        # keyed on self.num_timesteps instead of env_step
        return self.eval_freq_early if self.num_timesteps < self.switch_step else self.eval_freq_late

    def _init_callback(self) -> None:
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        if not ray.is_initialized():
            ray.init(num_cpus=self.n_repeats)
            self._ray_owns_init = True

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self._current_eval_freq():
            self._last_eval_step = self.num_timesteps
            self._run_eval()
        return True

    def _run_eval(self):
        self.model.save(self.checkpoint_path)

        per_problem = {}
        for t in self._task:
            self.eval_args.key = t
            per_problem[t] = run_repeats(
                self.eval_args, self.checkpoint_path, self.n_repeats)
            if self.verbose:
                print(f"[IGD eval @ {self.num_timesteps}] {t} done")

        self._history.append({"step": self.num_timesteps, "results": per_problem})
        # Overwritten on every firing -- full running history saved each
        # time (not just latest), same as DQN's igd_curve.npz convention.
        np.savez(self.curve_path, history=self._history)

    def _on_training_end(self) -> None:
        if self._ray_owns_init:
            ray.shutdown()