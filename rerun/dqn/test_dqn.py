import os
from mamo.saenv import MOEAEnv
from mamo.mamo_register import Task
import argparse
import torch
import numpy as np
import random
import ray

from tianshou.utils.net.common import Net
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.data import Batch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, default='M_2_46_5')
    parser.add_argument('--seed', type=int, default=2022) #Evaluation reseeding base, not training seed
    parser.add_argument('--train-seed', type=int, default=2022) #Which trained policy to load, Selects results/dqn/<key>/seed_<train-seed>/
    parser.add_argument('--population-size', type=int, default=210)
    parser.add_argument('--budget-ratio', type=int, default=100)
    parser.add_argument('--n-ref-points', type=int, default=1000)
    parser.add_argument('--save-history', action="store_true", default=True)
    parser.add_argument('--baseline', action="store_true",
                        default=False)  # use default operator Type
    parser.add_argument('--adaptive-open', action="store_true",
                        default=True)  # use adaptive Weights
    parser.add_argument('--early-stop', action="store_true", default=False)
    parser.add_argument('--test', action="store_true", default=False)
    args = parser.parse_known_args()[0]
    return args


class DQNargs:
    train_num = 16
    test_num = 4
    hidden_sizes = [128, 128, 128]
    lr = 1e-3
    gamma = 0.9
    n_step = 3
    target_update_freq = 160
    buffer_size = 20000
    eps_train = 0.1
    epoch = 20
    step_per_epoch = 5000
    step_per_collect = 32
    update_per_step = 0.05
    batch_size = 64
    episode_per_test = 15
    repeat = 30


def set_global_seeds(seed: int):
    """
    Set the random seed of pytorch, numpy and random.
    params:
        seed: an integer refers to the random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def _env_kwargs(args):
    """vars(args) minus script-only args that MOEAEnv's constructor doesn't accept."""
    return {k: v for k, v in vars(args).items() if k != 'train_seed'}

@ray.remote
def step_in_env(args, policy, run_idx):
    """
    Runs one evaluation episode of `policy` on the problem specified by
    args.key. Seeded with args.seed + run_idx (matches test_ppo.py's
    per-repeat seeding pattern) so repeats don't share RNG state -- each
    Ray worker is its own process, so this reseeds only that worker's
    local random/np.random, not the caller's.
    """
    np.random.seed(args.seed + run_idx)
    random.seed(args.seed + run_idx)
    env = MOEAEnv(**_env_kwargs(args))
    obs, _ = env.reset()

    info = {'best_igd': 1e6, 'last_igd': 1e6}
    terminated = False
    truncated = False
    while not (terminated or truncated):
        batch = Batch(obs=Batch(obs=obs['obs'][np.newaxis, :], mask=obs['mask'][np.newaxis, :]), info={})
        act = policy(batch).act[0]
        obs, r, terminated, truncated, info = env.step(act)
    return info


def run_repeats(args, policy, n_repeats):
    """
    Runs n_repeats evaluation episodes of `policy` on args.key in parallel
    via Ray, each with its own seed (args.seed + run_idx). Caller is
    responsible for ray.init() with num_cpus >= n_repeats beforehand.

    This is the single source of truth for "run an eval episode" -- both
    test_dqn.py's own 30-repeat post-training evaluation and dqn.py's
    mid-training IGD curve hook call this same function, so seeding and
    episode logic can't drift between the two.

    Returns a list of info dicts, one per repeat.
    """
    return ray.get([step_in_env.remote(args, policy, i) for i in range(n_repeats)])


def dqn_run_baseline(args=get_args(), dqn_args=None, policy_name=None):
    args.save_history = True
    ao_tag = '' if args.adaptive_open else '_ao_false'
    save_path = f'./results/dqn/eval_trainsd{args.train_seed}{ao_tag}/'
    if not os.path.exists(save_path):
        os.umask(0)
        os.makedirs(save_path, mode=0o777)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = MOEAEnv(**_env_kwargs(args))
    state_shape = env.observation_space['obs'].shape
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=dqn_args.hidden_sizes)
    policy = DiscreteQLearningPolicy(
        model=net,
        action_space=env.action_space,
        observation_space=env.observation_space,
        eps_inference=0.0,
    )
    policy_weights = torch.load(
        f'./results/dqn/{policy_name}/seed_{args.train_seed}{ao_tag}/{policy_name}policy.pth',
        map_location='cpu',
    )
    policy.load_state_dict(policy_weights)
    policy.eval()

    info = run_repeats(args, policy, dqn_args.repeat)
    np.savez(
        f'{save_path}{args.key}_trainsd{args.train_seed}_evalsd{args.seed}_rp{dqn_args.repeat}.npz',
        info_stack=info)


if __name__ == "__main__":
    args = get_args()
    policy_name = args.key
    dqn_args = DQNargs()
    set_global_seeds(args.seed)
    task = Task.get_task(name="all" + args.key.split("_")[-1])
    ray.init(num_cpus=dqn_args.repeat)
    for t in task:
        args.key = t
        dqn_run_baseline(args, dqn_args, policy_name)
        print("===== Finish" + args.key + " =====")