"""
SPACE curriculum callback for PPO+MaMo. Owns all decision logic.
"""

import logging

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

from space_parallel.space_enum import space_operation, instance_ordering
from space_parallel.space_env import SPaceEnvParallel


class UpdateEnvCallbackParallel(BaseCallback):
    def __init__(
        self,
        key: str,
        use_space_val: int = 2,
        instance_ordering_val: int = 0,
        eta_const: float = 0.1,
        step_size_const: int = 1,
        probe_env_kwargs: dict = None,
        logger_override: logging.Logger = None,
        verbose: int = 0,
        probe_seed_base: int = 0,
    ):
        super().__init__(verbose)

        self.key = key
        self.use_space = space_operation(use_space_val)
        self.instance_ordering = instance_ordering(instance_ordering_val)

        self.eta_const = eta_const
        self.step_size_const = step_size_const
        self.probe_env_kwargs = probe_env_kwargs or {}
        self.probe_seed_base = probe_seed_base

        self.space_logger = logger_override or logging.getLogger("space_callback_parallel")

        # probe env, never touched by rollout collection
        self.probe_env = SPaceEnvParallel(
            key=self.key,
            use_space=space_operation.INSTANCE_STATE.value,
            **self.probe_env_kwargs,
        )

        self.num_training_functions = self.probe_env.num_training_functions
        self.last_evals = {i: 0.0 for i in range(self.num_training_functions)}

        self.curriculum_size = 1
        self.curriculum = list(range(self.num_training_functions))
        self.active_curriculum = self.curriculum[: self.curriculum_size]

        # None (not 0.0) so the first boundary records a baseline instead of scoring against a zero threshold
        self.last_q = None

        # True until the first policy update has run.
        self.before_first_update = True

    
    # Callback hooks

    def _on_training_start(self) -> None:
        if self.use_space == space_operation.NO_SPACE:
            return
        # Initial curriculum: size 1, random ordering.
        self._update_curriculum(evals=None)

    def _on_rollout_start(self) -> None:
        if self.use_space == space_operation.NO_SPACE:
            return
        if self.before_first_update:
            return

        # Single probe pass per boundary, shared by the size and the ordering decision.
        evals = self._get_instance_evals()
        self._update_curriculum_size(evals)
        self._update_curriculum(evals)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.space_logger.info("Rollout collected at %d timesteps (policy update follows)",self.num_timesteps)
        self.before_first_update = False

    def _on_training_end(self) -> None:
        self.probe_env.close()


    def _update_curriculum_size(self, evals) -> None:
        # mean_q over the curriculum that was active for the rollout just collected
        mean_q = float(np.mean([evals[i] for i in self.active_curriculum]))

        if self.last_q is None:
            self.last_q = mean_q
            self.space_logger.info("First size check skipped; baseline mean_q=%.6f", mean_q)
            return

        # test whether value estimate has stabilised
        delta_q = np.abs(np.abs(mean_q) - np.abs(self.last_q))
        self.last_q = mean_q
        threshold = self.eta_const * np.abs(self.last_q)
       

        self.space_logger.info(
            "mean_q=%.6f delta_q=%.6f threshold=%.6f",
            mean_q, delta_q, threshold,
        )

        if delta_q <= threshold and self.curriculum_size < self.num_training_functions:
            old_size = self.curriculum_size
            self.curriculum_size = min(old_size + self.step_size_const, self.num_training_functions)
            self.space_logger.info("Growing curriculum: %d -> %d", old_size, self.curriculum_size)
        else:
            self.space_logger.info("Curriculum size held at %d", self.curriculum_size)
        

    # ------------------------------------------------------------------
    # Curriculum ordering
    # ------------------------------------------------------------------

    def _update_curriculum(self, evals) -> None:
        if self.use_space == space_operation.JUST_SIZES or self.instance_ordering == instance_ordering.NONE:
            ordered = list(range(self.num_training_functions))
        elif evals is None:
            self.space_logger.info("Before first policy update: random curriculum")
            ordered = np.random.permutation(self.num_training_functions).tolist()
        elif self.instance_ordering == instance_ordering.ABSOLUTE:
            ordered = self._order_instances_qvals(evals)
        elif self.instance_ordering == instance_ordering.IMPROVEMENT:
            ordered = self._order_instances_improvement(evals)
        elif self.instance_ordering == instance_ordering.RELATIVE_IMPROVEMENT:
            ordered = self._order_instances_relative_improvement(evals)
        else:
            ordered = list(range(self.num_training_functions))

        # only train on the active curriculum
        self.curriculum = [int(i) for i in ordered]
        self.active_curriculum = self.curriculum[: self.curriculum_size]

        # broadcast to every subprocess env
        self.training_env.env_method("set_active_curriculum", self.active_curriculum)

        self.space_logger.info(
            "Curriculum size %d, active %s (full order %s)",
            self.curriculum_size, self.active_curriculum, self.curriculum,
        )

    def _order_instances_qvals(self, evals):
        # Ascending order
        self.last_evals = {i: float(v) for i, v in enumerate(evals)}
        return np.argsort(evals).tolist()

    def _order_instances_improvement(self, evals):
        # order by whicher instance's value estimate rose the most since last check
        improvement = {i: float(evals[i]) - self.last_evals[i]
                       for i in range(len(evals))}
        if all(v == 0 for v in improvement.values()):
            self.space_logger.info("No improvement, keeping curriculum unchanged")
            return self.curriculum
        self.last_evals = {i: float(v) for i, v in enumerate(evals)}
        return sorted(improvement, key=improvement.get, reverse=True)

    def _order_instances_relative_improvement(self, evals):
        # improvement normalised by the instance's previous value
        relative = {}
        for i, new_eval in enumerate(evals):
            old_eval = self.last_evals[i] or 1e-5
            denom = max(abs(old_eval), 1e-5)
            relative[i] = (float(new_eval) - old_eval) / denom
        if all(v == 0 for v in relative.values()):
            self.space_logger.info("No relative improvement, keeping curriculum unchanged")
            return self.curriculum
        self.last_evals = {i: float(v) for i, v in enumerate(evals)}
        return sorted(relative, key=relative.get, reverse=True)

    # Value-function queries 

    def _get_instance_evals(self):
        evals = []
        for i in range(self.num_training_functions):
            # Force the probe env onto instance i alone, then read critic's V(s0) for that instance's initial state.
            self.probe_env.set_active_curriculum([i])
            obs, _ = self.probe_env.reset(seed=self.probe_seed_base + i)
            obs_t = obs_as_tensor(
                np.asarray(obs, dtype=np.float32), self.model.device
            ).unsqueeze(0)  # add batch dim of 1, as predict_values expects a batch
            with th.no_grad():
                val = self.model.policy.predict_values(obs_t)
            evals.append(float(val.detach().cpu().numpy().squeeze()))
        self.space_logger.info("Instance evals: %s", evals)
        return np.array(evals)