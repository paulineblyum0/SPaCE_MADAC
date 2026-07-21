import numpy as np
import gymnasium as gym
from gymnasium import spaces

from mamo.saenv import MOEAEnv


class PPOMOEAEnv(gym.Env):
    """
    Single-agent PPO wrapper around MOEAEnv.
    """

    def __init__(self, key="WFG6_3", **kwargs):
        super().__init__()
        # Instantiate the underlying env but do not use its step/action_space
        self._inner = MOEAEnv(key=key, **kwargs)

        # MultiDiscrete matches the 4 agent action spaces: [4, 4, 4, 2]
        self.action_space = spaces.MultiDiscrete([4, 4, 4, 2])

        # Flat Box observation: the 22-dim obs vector only, no mask
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(22,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs_dict, info = self._inner.reset(seed=seed, options=options)
        return obs_dict['obs'].astype(np.float32), info

    def step(self, action):
        """
        action: np.ndarray of shape (4,), dtype int, from MultiDiscrete sampler.
        Passed directly to moead_step().
        """
        action = np.asarray(action, dtype=np.int32)
        reward, done, info = self._inner.env.moead_step(action)
        obs = self._inner.env.obs.astype(np.float32)
        return obs, reward, done, False, info

    def close(self):
        self._inner.close()