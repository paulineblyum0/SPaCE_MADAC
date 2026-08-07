"""
Periodic IGD evaluation for Tianshou DQN training.

Tianshou's OffPolicyTrainer has no EvalCallback/BaseCallback mechanism like
SB3. The only step-level hook is train_fn(epoch, env_step), already used in
dqn.py for epsilon annealing. This hook wraps that same call point.

Reuses test_dqn.py's run_repeats/step_in_env directly rather than
duplicating eval-episode logic, so seeding and episode-running can't drift
between the mid-training diagnostic curve and the final reported
30-repeat evaluation.

Scope: evaluates on ALL 8 problems in the training key's task suite (same
"all" + M convention as test_dqn.py's __main__), not just the training key
itself -- e.g. training on M_2_46_3 evaluates on all 8 M=3 problems each
time it fires. Problems are evaluated consecutively, one after another;
only the n_repeats repeats *within* a single problem are Ray-parallel.
That means each trigger costs 8 sequential rounds of n_repeats-parallel
episodes, not one.

Usage in dqn.py, replacing the existing train_fn:

    import ray
    from dqn.igd_eval_hook import DQNIGDEvalHook

    igd_hook = DQNIGDEvalHook(
        policy=policy,
        args=args,
        eval_freq_early=5000,     # cadence while env_step < switch_step
        eval_freq_late=10000,     # coarser cadence after switch_step --
                                   # learning is typically front-loaded, so
                                   # the tail doesn't need the finer grid
        switch_step=100000,
        n_repeats=10,             # hardcoded 10 for the training curve;
                                   # final reported policy still uses 30
                                   # via test_dqn.py, unchanged
        base_train_fn=train_fn,   # existing eps-annealing function
        save_path=os.path.join(log_path, "igd_curve.npz"),
    )

    ray.init(num_cpus=10)  # >= n_repeats; call once, before trainer.run()

    trainer = OffPolicyTrainer(
        ...
        params=OffPolicyTrainerParams(
            ...
            training_fn=igd_hook,   # replaces training_fn=train_fn
            ...
        ),
    )

ray.init() is deliberately NOT called inside this module -- dqn.py owns
that, same convention as test_dqn.py's own ray.init(num_cpus=dqn_args.repeat)
in its __main__.
"""

import argparse

import numpy as np

from mamo.mamo_register import Task
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

        # same task-suite convention as test_dqn.py's __main__: all 8
        # problems sharing the training key's M value, e.g. M_2_46_3 -> all3
        self.tasks = Task.get_task(name="all" + args.key.split("_")[-1])
        self._history = {
            t: {"best_mean": [], "best_std": [], "last_mean": [], "last_std": []}
            for t in self.tasks
        }

    def _current_eval_freq(self, env_step):
        # finer resolution while learning is likely still moving; coarser
        # once past switch_step, on the assumption (worth checking against
        # the actual curve) that most improvement happens early and the
        # tail is comparatively flat
        return self.eval_freq_early if env_step < self.switch_step else self.eval_freq_late

    def _eval_all_problems(self):
        # per-problem args copies -- deliberately NOT mutating self.args.key
        # in place. self.args is the same object dqn.py uses elsewhere (e.g.
        # save_best_fn's checkpoint filename), so overwriting args.key here
        # would silently corrupt those after the first eval fires.
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