"""
Evaluation for a trained MA-DAC agent.
"""
import argparse
import glob
import os
import random
from types import SimpleNamespace as SN

import numpy as np
import ray
import torch as th
import yaml

from mamo.mamo_register import Task
from madac.controllers import REGISTRY as mac_REGISTRY
from madac.envs import REGISTRY as env_REGISTRY
from madac.components.episode_buffer import EpisodeBatch
from madac.components.transforms import OneHot


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-key', type=str, default='M_2_46_3')
    parser.add_argument('--alg', type=str, default='vdn_ns')
    parser.add_argument('--seed', type=int, default=2022)  # Evaluation reseeding base, not training seed
    parser.add_argument('--train-seed', type=int, default=2022)  # Which trained policy to load
    parser.add_argument('--repeat', type=int, default=30)
    args = parser.parse_known_args()[0]
    return args


def set_global_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)


def load_config(alg_name, train_key):

    # rebuilds the training config dict from the same yaml files run.py loaded
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    with open(os.path.join(config_dir, "default.yaml")) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    with open(os.path.join(config_dir, "envs", "moea.yaml")) as f:
        env_config = yaml.load(f, Loader=yaml.SafeLoader)
    with open(os.path.join(config_dir, "algs", f"{alg_name}.yaml")) as f:
        alg_config = yaml.load(f, Loader=yaml.SafeLoader)

    def recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    config = recursive_update(config, env_config)
    config = recursive_update(config, alg_config)
    config['env_args']['key'] = train_key
    config['use_cuda'] = False
    config['device'] = 'cpu'
    return config


def find_checkpoint_dir(alg_name, train_seed, map_name):
    pattern = os.path.join(
        "results", "madac", "models", f"{alg_name}_seed{train_seed}_rnn*_{map_name}_*")
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one checkpoint dir matching {pattern}, "
            f"found {len(matches)}: {matches}")
    return matches[0]


def find_final_timestep(checkpoint_dir):
    timesteps = [int(name) for name in os.listdir(checkpoint_dir)
                 if os.path.isdir(os.path.join(checkpoint_dir, name)) and name.isdigit()]
    return max(timesteps)


@ray.remote
def step_in_env(mac_type, agent_state_dict, scheme, groups, preprocess,
                 args_dict, key, seed, run_idx):
    """
    Runs one evaluation episode of a reconstructed MAC on `key`.
    """
    args = SN(**args_dict)
    np.random.seed(seed + run_idx)
    random.seed(seed + run_idx)

    env_args = {**args.env_args, "key": key, "seed": seed + run_idx,
                "replay": False, "save_history": True}
    env = env_REGISTRY[args.env](**env_args)
    episode_limit = env.episode_limit

    mac = mac_REGISTRY[mac_type](scheme, groups, args)
    mac.agent.load_state_dict(agent_state_dict)

    batch = EpisodeBatch(scheme, groups, 1, episode_limit + 1,
                          preprocess=preprocess, device="cpu")
    env.reset()
    mac.init_hidden(batch_size=1)

    terminated = False
    t = 0
    env_info = {}
    while not terminated:
        batch.update({
            "state": [env.get_state()],
            "avail_actions": [env.get_avail_actions()],
            "obs": [env.get_obs()],
        }, ts=t)

        actions = mac.select_actions(batch, t_ep=t, t_env=0, test_mode=True)
        reward, terminated, env_info = env.step(actions[0])

        batch.update({
            "actions": actions,
            "reward": [(reward,)],
            "terminated": [(terminated != env_info.get("episode_limit", False),)],
        }, ts=t)
        t += 1

    return {"best_igd": env_info.get("best_igd"), "last_igd": env_info.get("last_igd"),
            "igd_his": env_info.get("igd_his")}


def run_repeats(mac_type, agent_state_ref, scheme, groups, preprocess,
                 args_dict, key, seed, n_repeats):
    # n_repeats parallel eval episodes
    return ray.get([
        step_in_env.remote(mac_type, agent_state_ref, scheme, groups,
                            preprocess, args_dict, key, seed, i)
        for i in range(n_repeats)
    ])


def madac_run_baseline(args, config, agent_state_dict):
    save_path = f'./results/madac/eval/{args.train_key}/seed_{args.train_seed}/'
    if not os.path.exists(save_path):
        os.umask(0)
        os.makedirs(save_path, mode=0o777)

    args_ns = SN(**config)

    # Throwaway env just to read env_info 
    probe_env = env_REGISTRY[args_ns.env](**args_ns.env_args)
    env_info = probe_env.get_env_info()
    probe_env.close()

    config['n_agents'] = env_info['n_agents']
    config['n_actions'] = env_info['n_actions']
    config['state_shape'] = env_info['state_shape']

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": config['n_agents']}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=config['n_actions'])])}

    agent_state_ref = ray.put(agent_state_dict)

    task = Task.get_task(name="all" + args.train_key.split("_")[-1])
    for t in task:
        info = run_repeats(config['mac'], agent_state_ref, scheme, groups,
                            preprocess, config, t, args.seed, args.repeat)
        np.savez(
            f'{save_path}{t}_sd{args.seed}_rp{args.repeat}.npz',
            info_stack=info)
        print(f"===== Finish {t} =====")

    del agent_state_ref


if __name__ == "__main__":
    args = get_args()
    set_global_seeds(args.seed)

    config = load_config(args.alg, args.train_key)
    checkpoint_name = config['name']  # from the yaml, not args.alg directly
    checkpoint_dir = find_checkpoint_dir(checkpoint_name, args.train_seed, args.train_key)
    timestep = find_final_timestep(checkpoint_dir)
    print(f"Loading checkpoint: {checkpoint_dir}/{timestep}")

    agent_state_dict = th.load(
        os.path.join(checkpoint_dir, str(timestep), "agent.th"),
        map_location='cpu')

    ray.init(num_cpus=args.repeat, object_store_memory=2 * 1024**3)
    try:
        madac_run_baseline(args, config, agent_state_dict)
    finally:
        ray.shutdown()