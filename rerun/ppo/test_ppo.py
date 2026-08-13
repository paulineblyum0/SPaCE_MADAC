import os
import argparse
import random
import numpy as np
import ray

from stable_baselines3 import PPO

from mamo.mamo_register import Task
from ppo.ppo_env import PPOMOEAEnv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, default='M_2_46_3')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--repeat', type=int, default=30)
    parser.add_argument('--population-size', type=int, default=210)
    parser.add_argument('--budget-ratio', type=int, default=100)
    parser.add_argument('--adaptive-open', action="store_true", default=True)
    parser.add_argument('--early-stop', action="store_true", default=False)
    parser.add_argument('--pi-arch', type=int, nargs="+", default=[64, 64])
    parser.add_argument('--vf-arch', type=int, nargs="+", default=[64, 64])
    # Final reported evaluation keeps histories (needed for the appendix
    # within-episode IGD traces). The mid-training curve hook overrides
    # this to False via its own eval_args namespace.
    parser.add_argument('--save-history', action="store_true", default=True)
    args = parser.parse_known_args()[0]
    return args


def set_global_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)


@ray.remote
def step_in_env(args, policy_state, run_idx, stats_path=None):
    """
    Runs one evaluation episode of the trained PPO policy on the problem
    specified by args.key. The policy is reconstructed inside the remote
    function from a state_dict (not passed in as an already-loaded SB3
    model) since SB3 models aren't guaranteed to pickle cleanly across Ray
    worker processes.

    args.save_history controls whether the returned info carries the full
    within-episode population trace. It must be False for the
    mid-training curve: retaining ~50 firings' worth of traces in the
    training process is several GB. It is True for the final 30-repeat
    evaluation, where the traces are the point.
    """
    np.random.seed(args.seed + run_idx)
    random.seed(args.seed + run_idx)

    save_history = getattr(args, "save_history", True)

    env = PPOMOEAEnv(
        key=args.key, budget_ratio=args.budget_ratio,
        population_size=args.population_size, save_history=save_history,
        adaptive_open=args.adaptive_open, early_stop=args.early_stop,
    )
    model = PPO("MlpPolicy", env,
                policy_kwargs=dict(net_arch=dict(pi=args.pi_arch, vf=args.vf_arch)))
    model.policy.load_state_dict(policy_state)
    model.policy.eval()

    obs, _ = env.reset()

    terminated = False
    truncated = False
    info = {}
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
    env.close()
    return info


def run_repeats(args, policy_state, n_repeats, stats_path=None):
    """
    n_repeats Ray-parallel evaluation episodes on args.key, returning the
    raw list of info dicts. Single source of truth for "run an eval
    episode": both this module's 30-repeat post-training evaluation and
    ppo/ppo_igd_eval_hook.py's mid-training curve call it, so seeding and
    episode logic can't drift between the two.

    Caller is responsible for ray.init() beforehand.
    """
    return ray.get([step_in_env.remote(args, policy_state, i, stats_path)
                    for i in range(n_repeats)])


def ppo_run_baseline(args, policy_state, save_path='./results/ppo/'):
    if not os.path.exists(save_path):
        os.umask(0)
        os.makedirs(save_path, mode=0o777)
    np.random.seed(args.seed)
    random.seed(args.seed)

    info = run_repeats(args, policy_state, args.repeat)
    np.savez(
        f'{save_path}{args.key}_sd{args.seed}_rp{args.repeat}.npz',
        info_stack=info)


if __name__ == "__main__":
    args = get_args()

    train_key = args.key
    
    model_path = os.path.join("ppo", "results", train_key, f"seed_{args.seed}", "ppo_model")

    set_global_seeds(args.seed)
    task = Task.get_task(name="all" + train_key.split("_")[-1])
    ray.init(num_cpus=args.repeat)

    model = PPO.load(model_path)
    policy_ref = ray.put(model.policy.state_dict())

    save_path = f'./results/ppo/seed_{args.seed}/'
    for t in task:
        args.key = t
        ppo_run_baseline(args, policy_ref, save_path=save_path)
        print("===== Finish " + args.key + " =====")