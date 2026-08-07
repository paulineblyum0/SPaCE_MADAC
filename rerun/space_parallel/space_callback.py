# space_parallel/space_callback.py
"""
SPACE curriculum callback for PPO+MaMo, n_envs=16 (SubprocVecEnv) variant.

Differs from space.space_callback.UpdateEnvCallback in three ways, all
forced by subprocess parallelism:

  1. Value-function probing (get_instance_evals / get_mean_q) is done
     against a dedicated, single-process probe env -- never against any
     of the 16 live training envs. Resetting a live training env
     out-of-band while SB3's collect_rollouts loop has it mid-rollout
     would corrupt terminal-observation bookkeeping and rollout-buffer
     alignment; the probe env sidesteps this entirely by never
     participating in rollout collection. The probe env is a plain
     space.space_env.SPaceEnv (the n_envs=1 class) -- its
     set_curriculum([i])-then-reset() pattern is exactly what's needed
     here and is already validated.

  2. Curriculum growth/reordering decisions are made once per rollout
     (_on_rollout_end), not per-step/per-episode -- "curriculum
     exhaustion" isn't a coherent global concept across 16 independent
     episode streams, but "a rollout's worth of new experience has been
     collected and the policy is about to be updated" is, and it's the
     natural point to re-probe with a fresh model snapshot.

  3. The resulting active pool is pushed to all 16 subprocess envs via
     VecEnv.env_method("set_active_curriculum", ...) rather than direct
     attribute mutation (which isn't possible across process boundaries).

The size-growth math (_update_curriculum_size) and ordering math
(_order_instances_*) are unchanged from the n_envs=1 version.
"""

import logging

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

from space_parallel.space_enum import space_operation, instance_ordering
from space.space_env import SPaceEnv  # reused unmodified as the probe env


class UpdateEnvCallbackParallel(BaseCallback):
    def __init__(
        self,
        key: str,
        use_space_val: int = 2,
        instance_ordering_val: int = 0,
        stability_threshold: int = 3,
        eta_const: float = 0.1,
        step_size_const: int = 1,
        probe_env_kwargs: dict = None,
        logger_override: logging.Logger = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.key = key
        self.use_space = space_operation(use_space_val)
        self.instance_ordering = instance_ordering(instance_ordering_val)

        self.stability_threshold = stability_threshold
        self.eta_const = eta_const
        self.step_size_const = step_size_const
        self.probe_env_kwargs = probe_env_kwargs or {}

        self.space_logger = logger_override or logging.getLogger("space_callback_parallel")

        # Single-process probe env, never touched by rollout collection.
        # use_space=INSTANCE_STATE here only enables SPaceEnv's
        # set_curriculum()-driven instance forcing on reset(); the actual
        # size-growth/ordering decisions live in this callback, not in
        # the probe env itself.
        self.probe_env = SPaceEnv(
            key=self.key,
            use_space=space_operation.INSTANCE_STATE.value,
            **self.probe_env_kwargs,
        )

        self.num_training_functions = self.probe_env.num_training_functions
        self.last_evals = {i: 0.0 for i in range(self.num_training_functions)}

        self.curriculum_size = 1
        self.curriculum = list(range(self.num_training_functions))
        self.last_q = 0.0
        self.stable_streak = 0
        self.unstable_streak = 0
        self.before_first_rollout = True

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        if self.use_space == space_operation.NO_SPACE:
            return
        self._update_curriculum()

    def _on_step(self) -> bool:
        # All curriculum logic now happens at rollout boundaries -- see
        # module docstring point 2.
        return True

    def _on_rollout_end(self) -> None:
        self.space_logger.info(
            "Rollout collected at %d timesteps", self.num_timesteps
        )
        if self.use_space != space_operation.NO_SPACE:
            self._update_curriculum_size(self.active_curriculum)
            self._update_curriculum()
        self.before_first_rollout = False

    def _on_training_end(self) -> None:
        self.probe_env.close()

    # ------------------------------------------------------------------
    # Curriculum size growth (unchanged math from the n_envs=1 version)
    # ------------------------------------------------------------------

    def _update_curriculum_size(self, curriculum) -> None:
        mean_q = self._get_mean_q(curriculum)
        delta_q = np.abs(np.abs(mean_q) - np.abs(self.last_q))
        is_stable = delta_q <= self.eta_const * np.abs(self.last_q)

        if is_stable:
            self.stable_streak += 1
            self.unstable_streak = 0
        else:
            self.stable_streak = 0
            self.unstable_streak += 1

        self.last_q = mean_q

        if self.stable_streak >= self.stability_threshold + 1:
            old_size = self.curriculum_size
            self.curriculum_size = min(
                old_size + self.step_size_const, self.num_training_functions
            )
            self.space_logger.info("Growing curriculum: %d -> %d", old_size, self.curriculum_size)
        elif self.unstable_streak >= self.stability_threshold - 1:
            old_size = self.curriculum_size
            self.curriculum_size = max(old_size - self.step_size_const, 1)
            self.space_logger.info("Shrinking curriculum: %d -> %d", old_size, self.curriculum_size)

    # ------------------------------------------------------------------
    # Curriculum ordering (unchanged math from the n_envs=1 version)
    # ------------------------------------------------------------------

    def _update_curriculum(self) -> None:
        if self.use_space == space_operation.JUST_SIZES or \
                self.instance_ordering == instance_ordering.NONE:
            ordered = list(range(self.num_training_functions))
        elif self.before_first_rollout:
            self.space_logger.info("Before first policy update: random curriculum")
            ordered = np.random.permutation(self.num_training_functions).tolist()
        elif self.instance_ordering == instance_ordering.ABSOLUTE:
            ordered = self._order_instances_qvals()
        elif self.instance_ordering == instance_ordering.IMPROVEMENT:
            ordered = self._order_instances_improvement()
        elif self.instance_ordering == instance_ordering.RELATIVE_IMPROVEMENT:
            ordered = self._order_instances_relative_improvement()
        else:
            ordered = list(range(self.num_training_functions))

        self.curriculum = ordered
        new_curriculum = ordered[: self.curriculum_size]
        self.active_curriculum = new_curriculum 

        # Broadcast to all subprocess training envs -- the cross-process
        # equivalent of env.set_curriculum() + env.reset_curriculum_index()
        # in the n_envs=1 version.
        self.training_env.env_method("set_active_curriculum", new_curriculum)

        self.space_logger.info("New curriculum: %s", new_curriculum)

    def _order_instances_qvals(self):
        evals = self._get_instance_evals()
        self.last_evals = {i: v for i, v in enumerate(evals)}
        return list(np.argsort(evals))

    def _order_instances_improvement(self):
        evals = self._get_instance_evals()
        improvement = {i: evals[i] - self.last_evals[i] for i in range(len(evals))}
        if all(v == 0 for v in improvement.values()):
            self.space_logger.info("No improvement, keeping curriculum unchanged")
            return self.curriculum
        self.last_evals = {i: v for i, v in enumerate(evals)}
        return sorted(improvement, key=improvement.get, reverse=True)

    def _order_instances_relative_improvement(self):
        evals = self._get_instance_evals()
        relative = {}
        for i, new_eval in enumerate(evals):
            old_eval = self.last_evals[i] or 1e-5
            relative[i] = (new_eval - old_eval) / old_eval
        if all(v == 0 for v in relative.values()):
            self.space_logger.info("No relative improvement, keeping curriculum unchanged")
            return self.curriculum
        self.last_evals = {i: v for i, v in enumerate(evals)}
        return sorted(relative, key=relative.get, reverse=True)

    # ------------------------------------------------------------------
    # Value-function queries -- always against the probe env, never a
    # live training env.
    # ------------------------------------------------------------------

    def _get_instance_evals(self):
        evals = []
        for i in range(self.num_training_functions):
            self.probe_env.set_curriculum([i])
            self.probe_env.reset_curriculum_index()
            obs, _ = self.probe_env.reset()
            obs_t = obs_as_tensor(obs, self.model.device).unsqueeze(0)
            with th.no_grad():
                val = self.model.policy.predict_values(obs_t)
            evals.append(float(val.detach().cpu().numpy().squeeze()))
        self.space_logger.info("Instance evals: %s", evals)
        return np.array(evals)

    def _get_mean_q(self, curriculum):
        # Probe the instances actually in the (possibly ordered)
        # curriculum -- curriculum[i], not the canonical index i itself.
        # Matches the original SPACE/Vincent semantics of "mean Q over
        # whatever's currently active", not "mean Q over instances 0..size-1".
        qs = []
        for i in range(len(curriculum)):
            self.probe_env.set_curriculum([curriculum[i]])
            self.probe_env.reset_curriculum_index()
            obs, _ = self.probe_env.reset()
            obs_t = obs_as_tensor(obs, self.model.device).unsqueeze(0)
            with th.no_grad():
                val = self.model.policy.predict_values(obs_t)
            qs.append(float(val.detach().cpu().numpy().squeeze()))
        return float(np.mean(qs))