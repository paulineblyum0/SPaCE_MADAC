from functools import partial

from mamo.multiagentenv import MultiAgentEnv
from mamo.maenv import MOEAEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["moea"] = partial(env_fn, env=MOEAEnv)
