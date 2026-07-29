# space/space_callback.py
"""
SPACE curriculum callback for PPO+MaMo.

Owns all SPACE decision logic: when to grow the curriculum, and how to
order instances within it. Reads/writes curriculum state on the training
env via self.training_env.envs[0].unwrapped -- this assumes n_envs=1
(DummyVecEnv), matching the PPO-baseline comparison setup. Do not point
this at a SubprocVecEnv with n_envs > 1; the single-environment assumption
is baked into every method below.

Note on timing: SPaceEnv.reset() advances curriculum_index and is called
automatically by the vec env immediately after a terminal step() -- before
_on_step() runs. So this callback always detects curriculum exhaustion one
episode after the fact; the episode that ran in between uses a fallback
instance (curriculum[0] of the *old* curriculum). This mirrors the original
SPACE implementation's timing and is treated as an accepted minor
inefficiency, not a bug.
"""

import logging

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

from space.space_enum import space_operation, instance_ordering


class UpdateEnvCallback(BaseCallback):
    def __init__(
        self,
        use_space_val: int = 2,
        instance_ordering_val: int = 0,
        stability_threshold: int = 3,
        eta_const: float = 0.1,
        step_size_const: int = 1,
        logger_override: logging.Logger = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.use_space = space_operation(use_space_val)
        self.instance_ordering = instance_ordering(instance_ordering_val)

        self.stability_threshold = stability_threshold
        self.eta_const = eta_const
        self.step_size_const = step_size_const

        self.space_logger = logger_override or logging.getLogger("space_callback")

        # Populated in _on_training_start once the env is known
        self.num_training_functions = None
        self.last_evals = {}

        # Curriculum-growth bookkeeping
        self.curriculum_size = 1
        self.curriculum = []
        self.last_q = 0.0
        self.stable_streak = 0
        self.unstable_streak = 0

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        env = self.training_env.envs[0].unwrapped
        self.num_training_functions = env.num_training_functions
        self.last_evals = {i: 0.0 for i in range(self.num_training_functions)}

        if self.use_space == space_operation.NO_SPACE:
            return
        self._update_curriculum()

    def _on_step(self) -> bool:
        if self.use_space == space_operation.NO_SPACE:
            return True

        env = self.training_env.envs[0].unwrapped
        current_index = env.curriculum_index
        curriculum = env.curriculum

        if current_index > len(curriculum):
            self.space_logger.info("Curriculum exhausted, transitioning...")
            self._update_curriculum_size(curriculum)
            self._update_curriculum()

        return True

    def _on_rollout_end(self) -> None:
        self.space_logger.info(
            "Rollout collected at %d timesteps", self.num_timesteps
        )
        self.training_env.envs[0].unwrapped.before_first_rollout = False

    # ------------------------------------------------------------------
    # Curriculum size growth
    # ------------------------------------------------------------------

    def _update_curriculum_size(self, curriculum) -> None:
        env = self.training_env.envs[0].unwrapped
        mean_q = self._get_mean_q(env, curriculum)
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
            self.space_logger.info(
                "Growing curriculum: %d -> %d", old_size, self.curriculum_size
            )
        elif self.unstable_streak >= self.stability_threshold - 1:
            old_size = self.curriculum_size
            self.curriculum_size = max(old_size - self.step_size_const, 1)
            self.space_logger.info(
                "Shrinking curriculum: %d -> %d", old_size, self.curriculum_size
            )

    # ------------------------------------------------------------------
    # Curriculum ordering
    # ------------------------------------------------------------------

    def _update_curriculum(self) -> None:
        env = self.training_env.envs[0].unwrapped

        if self.use_space == space_operation.JUST_SIZES or \
                self.instance_ordering == instance_ordering.NONE:
            ordered = list(range(self.num_training_functions))

        elif env.before_first_rollout:
            self.space_logger.info("Before first policy update: random curriculum")
            ordered = np.random.permutation(self.num_training_functions).tolist()

        elif self.instance_ordering == instance_ordering.ABSOLUTE:
            ordered = self._order_instances_qvals(env)

        elif self.instance_ordering == instance_ordering.IMPROVEMENT:
            ordered = self._order_instances_improvement(env)

        elif self.instance_ordering == instance_ordering.RELATIVE_IMPROVEMENT:
            ordered = self._order_instances_relative_improvement(env)

        else:
            ordered = list(range(self.num_training_functions))

        self.curriculum = ordered
        new_curriculum = ordered[: self.curriculum_size]

        env.set_curriculum_size(self.curriculum_size)
        env.set_curriculum(new_curriculum)
        env.reset_curriculum_index()
        # Skip index 0: the episode that just ran already consumed a
        # fallback instance while this curriculum was being built (see
        # module docstring on reset()/step() timing).
        env.set_curriculum_index(1)

        self.space_logger.info("New curriculum: %s", new_curriculum)

    def _order_instances_qvals(self, env):
        evals = self._get_instance_evals(env)
        self.last_evals = {i: v for i, v in enumerate(evals)}
        return list(np.argsort(evals))

    def _order_instances_improvement(self, env):
        evals = self._get_instance_evals(env)
        improvement = {i: evals[i] - self.last_evals[i] for i in range(len(evals))}

        if all(v == 0 for v in improvement.values()):
            self.space_logger.info("No improvement, keeping curriculum unchanged")
            return self.curriculum

        self.last_evals = {i: v for i, v in enumerate(evals)}
        return sorted(improvement, key=improvement.get, reverse=True)

    def _order_instances_relative_improvement(self, env):
        evals = self._get_instance_evals(env)
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
    # Value-function queries
    # ------------------------------------------------------------------

    def _get_instance_evals(self, env):
        evals = []
        prev_curriculum = env.get_curriculum()

        for i in range(self.num_training_functions):
            env.set_curriculum([i])
            env.reset_curriculum_index()
            obs, _ = env.reset()
            obs_t = obs_as_tensor(obs, self.model.device).unsqueeze(0)
            with th.no_grad():
                val = self.model.policy.predict_values(obs_t)
            evals.append(float(val.detach().cpu().numpy().squeeze()))

        env.set_curriculum(prev_curriculum)
        self.space_logger.info("Instance evals: %s", evals)
        return np.array(evals)

    def _get_mean_q(self, env, curriculum):
        prev_curriculum = env.get_curriculum()
        qs = []

        for i in range(len(curriculum)):
            env.set_curriculum([i])
            env.reset_curriculum_index()
            obs, _ = env.reset()
            obs_t = obs_as_tensor(obs, self.model.device).unsqueeze(0)
            with th.no_grad():
                val = self.model.policy.predict_values(obs_t)
            qs.append(float(val.detach().cpu().numpy().squeeze()))

        env.set_curriculum(prev_curriculum)
        return float(np.mean(qs))