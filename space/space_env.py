# space/space_env.py
"""
SPACE-aware wrapper around PPOMOEAEnv.

Exposes the curriculum API contract expected by space_callback.UpdateEnvCallback:
    .curriculum, .curriculum_index, .before_first_rollout
    .set_curriculum(), .get_curriculum(), .set_curriculum_size(),
    .reset_curriculum_index(), .set_curriculum_index()

When use_space == NO_SPACE, falls through to PPOMOEAEnv's normal behavior
(MamoBase's built-in round-robin over func_select) unchanged -- this keeps
--use-space 0 runs identical to the existing PPO baseline.

When SPACE is active, this class takes over instance selection directly:
it freezes a canonical, never-reshuffled ordering of instances captured at
construction time, and on each reset() forces the inner MamoBase env onto
exactly the instance named by the current curriculum position. This avoids
MamoBase's func_select reshuffle-on-wraparound, which would otherwise
silently break the index->instance mapping SPACE's value-ordering relies on.
"""

from ppo.ppo_env import PPOMOEAEnv
from space.space_enum import space_operation


class SPaceEnv(PPOMOEAEnv):
    def __init__(self, key="WFG6_3", use_space=0, **kwargs):
        super().__init__(key=key, **kwargs)

        self.use_space = space_operation(use_space)

        # Canonical instance ordering, captured before MamoBase can shuffle it.
        # Every curriculum index (0..num_training_functions-1) refers to a
        # fixed position in this list for the lifetime of the env.
        mamo_env = self._inner.env
        self._canonical_instances = list(mamo_env.func_select)
        self.num_training_functions = len(self._canonical_instances)

        # SPACE curriculum state (mutated by UpdateEnvCallback)
        self.curriculum = list(range(self.num_training_functions))
        self.curriculum_size = 1
        self.curriculum_index = 0
        self.before_first_rollout = True

    # ------------------------------------------------------------------
    # Curriculum API expected by UpdateEnvCallback
    # ------------------------------------------------------------------

    def set_curriculum(self, instance_indices):
        self.curriculum = list(instance_indices)

    def get_curriculum(self):
        return self.curriculum.copy()

    def set_curriculum_size(self, size: int):
        self.curriculum_size = size

    def reset_curriculum_index(self):
        self.curriculum_index = 0

    def set_curriculum_index(self, index: int):
        self.curriculum_index = index

    # ------------------------------------------------------------------
    # Instance selection
    # ------------------------------------------------------------------

    def _force_instance(self, canonical_idx: int):
        """
        Directly overrides the inner MamoBase env's instance selection,
        bypassing its own fun_index auto-advance and reshuffle.
        """
        mamo_env = self._inner.env
        mamo_env.func_select = [self._canonical_instances[canonical_idx]]
        mamo_env.fun_index = 0

    def reset(self, seed=None, options=None):
        if self.use_space != space_operation.NO_SPACE:
            if 0 <= self.curriculum_index < len(self.curriculum):
                canonical_idx = self.curriculum[self.curriculum_index]
            else:
                # Guard for the transition window between "curriculum
                # exhausted" and the callback installing the next one.
                canonical_idx = self.curriculum[0]
            self._force_instance(canonical_idx)
            self.curriculum_index += 1

        # NO_SPACE: fall through untouched -- MamoBase's own round-robin
        # over its (possibly reshuffled) func_select still applies.
        return super().reset(seed=seed, options=options)