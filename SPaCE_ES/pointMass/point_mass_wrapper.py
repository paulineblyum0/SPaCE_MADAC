import numpy as np
import gymnasium
from gymnasium import spaces
from contextual_point_mass import ContextualPointMass


class PointMassWrapper(gymnasium.Env):
    def __init__(self, instance_feats, test, action_space_size=10):
        self.instances = instance_feats
        self.inst_id = -1
        if test:
            self.instance_set_size = 1
            self.curr_set = instance_feats
            self.indices = np.arange(len(instance_feats))
        else:
            self.instance_set_size = 1 / len(self.instances)
            self.curr_set = [self.instances[0]]
            self.indices = [0]
        self.action_space = spaces.Box(
            np.array([-10.0, -10.0]), np.array([10.0, 10.0])
        )
        self.action_mapping = {}
        maps_to = np.linspace(-1, 1, num=action_space_size, endpoint=True)
        for i in range(action_space_size):
            for j in range(action_space_size):
                self.action_mapping[i * action_space_size + j] = [
                    maps_to[i],
                    maps_to[j],
                ]
        # self.observation_space = self.env.observation_space["observation"]
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(4), high=np.ones(4) * np.inf
        )
        self.test = None
        self.c_step = 0
        

    def reset(self):
        self.c_step = 0
        self.inst_id = (self.inst_id + 1) % len(self.curr_set)
        goal_pos = self.curr_set[self.inst_id][:2]
        start_pos = self.curr_set[self.inst_id][2:]
        obs = self.env.reset()
        return obs

    def step(self, action):
        self.c_step += 1
        # a = self.action_mapping[action]
        out = self.env.step(np.array(action))
        return out

    def get_id(self):
        return self.inst_id

    def set_test(self, test):
        self.test = test

    def get_feats(self):
        return len(self.instances)

    def get_instance_set(self):
        return self.indices, self.curr_set

    def get_instances(self):
        return self.instances

    def get_instance_size(self):
        return int(np.ceil(len(self.instances) * self.instance_set_size))

    def increase_set_size(self, kappa, multi):
        if multi:
            self.instance_set_size = self.instance_set_size * kappa
        else:
            self.instance_set_size += kappa / len(self.instances)

    def set_instance_set(self, indices):
        size = int(np.ceil(len(self.instances) * self.instance_set_size))
        if size <= 0:
            size = 1
        self.curr_set = np.array(self.instances)[indices[:size]]
        self.indices = indices


class CPMWrapper(PointMassWrapper):
    def __init__(self, instance_feats, test, external_eval=False):
        super().__init__(instance_feats, test)
        self.env = ContextualPointMass()

    def step(self, action):
        self.c_step += 1
        obs, r, terminated, truncated, info = self.env.step(action)
        if self.c_step >= 100:
            truncated = True
        return obs, r, terminated, truncated, info

    def reset(self, **kwargs):
        self.c_step = 0
        self.inst_id = (self.inst_id + 1) % len(self.curr_set)
        self.env = ContextualPointMass(self.curr_set[self.inst_id])
        obs, info = self.env.reset()
        return obs, info
