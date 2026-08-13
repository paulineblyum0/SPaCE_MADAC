# space_parallel/space_env.py
"""
Parallel-safe SPACE-aware wrapper around PPOMOEAEnv, for use under
SubprocVecEnv with n_envs > 1.

Differs from space.space_env.SPaceEnv in one essential way: there is no
single shared curriculum_index that a callback can read/write directly,
because each instance of this class lives in its own subprocess, with no
shared Python object state. Instead:

  - The active instance pool ("active curriculum") is pushed into every
    subprocess via VecEnv.env_method("set_active_curriculum", ...),
    called from space_parallel.space_callback.UpdateEnvCallbackParallel
    whenever the curriculum changes.
  - Each env independently round-robins over its own *local* copy of that
    pool on every reset() -- this replaces the single global
    curriculum_index from the n_envs=1 version. Perfect synchronization of
    "which instance is served on which global step" is neither achievable
    nor necessary across 16 independent episode streams; what matters for
    SPACE's curriculum-growth logic is only that every env samples from
    the same active pool at any given time.

Do not use this class with n_envs=1 -- use space.space_env.SPaceEnv for
that; its global-index semantics are simpler and already validated.
"""

from ppo.ppo_env import PPOMOEAEnv
from space_parallel.space_enum import space_operation
from mamo.mamo_register import get_maenv


class SPaceEnvParallel(PPOMOEAEnv):
    def __init__(self, key="WFG6_3", use_space=0, **kwargs):
        super().__init__(key=key, **kwargs)

        self.use_space = space_operation(use_space)

        # Canonical instance ordering, derived from the register rather than
        # from MamoBase.func_select. MamoBase.__init__ applies an unseeded
        # random.shuffle to func_select, so reading it here would give a
        # different index->instance mapping in every process and every run.
        # This is MamoBase's pre-shuffle construction order.
        funcs, nobjs = get_maenv(key)
        self._canonical_instances = [(f, n) for f in funcs for n in nobjs]
        self.num_training_functions = len(self._canonical_instances)

        # Local curriculum state -- lives entirely inside this subprocess.
        # Starts as the full instance set so any rollouts collected before
        # the first callback broadcast behave like round-robin over
        # everything (matching the n_envs=1 version's
        # before_first_rollout -> random-permutation behavior in spirit).
        self._active_curriculum = list(range(self.num_training_functions))
        self._local_idx = 0

    # ------------------------------------------------------------------
    # Called remotely via env_method from the main process
    # ------------------------------------------------------------------

    def set_active_curriculum(self, instance_indices):
        self._active_curriculum = list(instance_indices)
        self._local_idx = 0

    def get_num_training_functions(self):
        return self.num_training_functions

    # ------------------------------------------------------------------
    # Instance selection
    # ------------------------------------------------------------------

    def _force_instance(self, canonical_idx: int):
        mamo_env = self._inner.env
        mamo_env.func_select = [self._canonical_instances[canonical_idx]]
        mamo_env.fun_index = 0

    def reset(self, seed=None, options=None):
        if self.use_space != space_operation.NO_SPACE:
            pool = self._active_curriculum or list(range(self.num_training_functions))
            canonical_idx = pool[self._local_idx % len(pool)]
            self._force_instance(canonical_idx)
            self._local_idx += 1

        # NO_SPACE: fall through untouched -- MamoBase's own round-robin
        # over its (possibly reshuffled) func_select still applies, same
        # as the n_envs=1 version and the plain PPO baseline.
        return super().reset(seed=seed, options=options)