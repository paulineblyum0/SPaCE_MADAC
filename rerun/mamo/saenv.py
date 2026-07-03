import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from mamo.moead_env import MamoBase


class MOEAEnv(gym.Env):
    def __init__(self,
                 key="WFG6_3",
                 n_ref_points=1000,
                 population_size=210,
                 seed=2022,
                 budget_ratio=50,
                 test=False,
                 wo_obs=False,
                 save_history=False,
                 early_stop=True,
                 baseline=False,
                 adaptive_open=False,
                 replay=False,
                 replay_dir=None,
                 ban_agent=4,
                 reward_type=0):
        super().__init__()
        self.env = MamoBase(key,
                            n_ref_points,
                            population_size,
                            seed,
                            budget_ratio,
                            test,
                            wo_obs,
                            save_history,
                            early_stop,
                            baseline,
                            adaptive_open,
                            replay,
                            replay_dir,
                            ban_agent,
                            reward_type)
        self.total_n_actions = self.env.total_n_actions
        self.n_obs = self.env.n_obs
        self.n_agents = self.env.n_agents
        self.action_space = spaces.Discrete(self.total_n_actions)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(0.0, 1.0, shape=(self.n_obs,), dtype=np.float32),
            'mask': spaces.Box(0, 1, shape=(self.total_n_actions,), dtype=bool)
        })

    def step(self, _action):
        """
        Returns reward, terminated, info
        :type action: int or np.int
        :param action: for dimension: Neighbor Size, Operator Type, Operator Parameter, Adaptive Weights
                [x, x, x, x]: shape=n_agents
        :return: obs, reward, terminated, truncated, info
        """
        action = np.zeros(self.n_agents, dtype=np.int32)
        action[0] = _action // (4 * 4 * 2)
        action[1] = _action % (4 * 4 * 2) // (4 * 2)
        action[2] = _action % (4 * 2) // 2
        action[3] = _action % 2
        reward, done, info = self.env.moead_step(action)
        
        return self.get_obs(), reward, done, False, info   # obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):   
        """Returns initial observations and info"""
        super().reset(seed=seed) # propagates seed to self.np_random
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.env.moead_reset()
        return self.get_obs(), {} 

    def close(self):
        self.env.moead_reset()

    def get_obs(self):
        """ Returns all agent observations in a list """
        obs = {'obs': self.env.obs, 'mask': self.get_avail_actions()}
        return obs

    def get_avail_actions(self):
        ava = np.full(self.env.total_n_actions, True)
        if self.env.adaptive_cooling_time > 0 or (
                np.sum(self.env.action_count[0]) >= self.env.adaptive_end // self.env.population_size):
            ava[1::2] = False
        return ava

    def render(self):
        raise NotImplementedError


if __name__ == "__main__":
    env = MOEAEnv()
    obs, info = env.reset() # unpack (obs, info)
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action) # unpack 5-tuple
        print(reward, terminated, truncated, info)
        if terminated or truncated:
            obs, info = env.reset()