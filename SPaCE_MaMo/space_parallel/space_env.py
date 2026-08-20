"""
SPACE wrapper around PPOMOEAEnv, for use under SubprocVecEnv with n_envs > 1.
"""

from ppo.ppo_env import PPOMOEAEnv
from space_parallel.space_enum import space_operation
from mamo.mamo_register import get_maenv


class SPaceEnvParallel(PPOMOEAEnv):
    def __init__(self, key="WFG6_3", use_space=0, **kwargs):
        super().__init__(key=key, **kwargs)

        self.use_space = space_operation(use_space)

        # Fixed (function, n_objectives) ordering from the register, not MamoBase (unseeded shuffle).
        funcs, nobjs = get_maenv(key)
        self._canonical_instances = [(f, n) for f in funcs for n in nobjs]
        self.num_training_functions = len(self._canonical_instances)

        # Local curriculum state
        self._active_curriculum = list(range(self.num_training_functions))
        self._local_idx = 0

    
    # Called remotely via env_method from the main process
    def set_active_curriculum(self, instance_indices):
        self._active_curriculum = list(instance_indices)
        self._local_idx = 0

    def get_num_training_functions(self):
        return self.num_training_functions


    # Instance selection

    def _force_instance(self, canonical_idx: int):
        # Directly overwrite the MamoBase env's instance list so it serves exactly this one instance on the next reset
        mamo_env = self._inner.env
        mamo_env.func_select = [self._canonical_instances[canonical_idx]]
        mamo_env.fun_index = 0

    def reset(self, seed=None, options=None):
         # Fall back to the full set for rollouts collected before the first callback update
        if self.use_space != space_operation.NO_SPACE:
            pool = self._active_curriculum or list(range(self.num_training_functions))
            canonical_idx = pool[self._local_idx % len(pool)]
            self._force_instance(canonical_idx)
            self._local_idx += 1

        return super().reset(seed=seed, options=options)