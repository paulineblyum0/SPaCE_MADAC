import os
import argparse
import random
import numpy as np
import ray

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from mamo.mamo_register import Task
from ppo.ppo_env import PPOMOEAEnv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, default='M_2_46_3')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--repeat', type=int, default=30)
    parser.add_argument('--budget-ratio', type=int, default=100)
    parser.add_argument('--n-ref-points', type=int, default=1000)
    parser.add_argument('--population-size', type=int, default=210)
    parser.add_argument('--save-history', action="store_true", default=False)
    parser.add_argument('--adaptive-open', action="store_true", default=False)
    parser.add_argument('--early-stop', action="store_true", default=False)
    parser.add_argument('--normalize', action="store_true", default=False)
    args = parser.parse_known_args()[0]
    return args


def set_global_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)

def load_obs_rms(stats_path, key, **env_kwargs):
    """Extract frozen obs_rms from a saved VecNormalize checkpoint."""
    dummy = DummyVecEnv([lambda: PPOMOEAEnv(key=key, **env_kwargs)])
    vecnorm = VecNormalize.load(stats_path, dummy)
    vecnorm.training = False  # freeze: never update stats at eval time
    return vecnorm.obs_rms, vecnorm.clip_obs, vecnorm.epsilon


def normalize_obs(obs, obs_rms, clip_obs, epsilon):
    obs_norm = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon)
    return np.clip(obs_norm, -clip_obs, clip_obs).astype(np.float32)


@ray.remote
def step_in_env(args, model_path, run_idx, stats_path=None):
    """
    Runs one evaluation episode of the trained PPO model on the problem
    specified by args.key. The model is loaded inside the remote function
    (not passed in as an already-loaded object) since SB3 models aren't
    guaranteed to pickle cleanly across Ray worker processes.
    """
    model = PPO.load(model_path)
    np.random.seed(args.seed + run_idx)
    random.seed(args.seed + run_idx)
    env = PPOMOEAEnv(
        key=args.key,
        budget_ratio=args.budget_ratio,
        n_ref_points=args.n_ref_points,
        population_size=args.population_size,
        save_history=args.save_history,
        adaptive_open=args.adaptive_open,
        early_stop=args.early_stop,
    )

    obs_rms = clip_obs = epsilon = None
    if stats_path is not None:
        obs_rms, clip_obs, epsilon = load_obs_rms(
            stats_path, args.key,
            budget_ratio=args.budget_ratio,
            adaptive_open=args.adaptive_open,
        )
    
    obs, _ = env.reset()
    if obs_rms is not None:
        obs = normalize_obs(obs, obs_rms, clip_obs, epsilon)

    terminated = False
    truncated = False
    info = {}
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if obs_rms is not None:
            obs = normalize_obs(obs, obs_rms, clip_obs, epsilon)
    env.close()
    return info


def ppo_run_baseline(args, model_path, stats_path=None):
    args.save_history = True
    save_path = './results/ppo/'
    if not os.path.exists(save_path):
        os.umask(0)
        os.makedirs(save_path, mode=0o777)
    np.random.seed(args.seed)
    random.seed(args.seed)

    info = ray.get([step_in_env.remote(args, model_path, i, stats_path)
                for i in range(args.repeat)])
    np.savez(
        f'{save_path}{args.key}_sd{args.seed}_rp{args.repeat}.npz',
        info_stack=info)


if __name__ == "__main__":
    args = get_args()
    policy_name = args.key
    model_path = os.path.join("ppo", "results", policy_name, "ppo_model")

    stats_path = None
    if args.normalize:
        stats_path = os.path.join("ppo", "results", policy_name, "vecnormalize.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"--normalize was passed but no VecNormalize stats found at "
                f"{stats_path}. This model cannot be evaluated correctly without it."
            )
    
    set_global_seeds(args.seed)
    task = Task.get_task(name="all" + args.key.split("_")[-1])
    ray.init(num_cpus=args.repeat)
    for t in task:
        args.key = t
        ppo_run_baseline(args, model_path, stats_path)
        print("===== Finish " + args.key + " =====")