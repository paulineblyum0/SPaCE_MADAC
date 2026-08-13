import os
import torch
import argparse

from mamo.saenv import MOEAEnv

import ray
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter

from dqn.igd_eval_hook import DQNIGDEvalHook


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, default='WFG6_3')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--population-size', type=int, default=210)
    parser.add_argument('--budget-ratio', type=int, default=100)
    parser.add_argument('--n-ref-points', type=int, default=1000)
    parser.add_argument('--save-history', action="store_true", default=False)
    parser.add_argument('--baseline', action="store_true", default=False)
    parser.add_argument('--adaptive-open', action="store_true", default=True)
    parser.add_argument('--early-stop', action="store_true", default=False)
    parser.add_argument('--test', action="store_true", default=False)
    args = parser.parse_known_args()[0]
    return args


class DQNargs:
    train_num = 16
    hidden_sizes = [128, 128, 128]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 3e-4
    gamma = 0.99
    n_step = 3
    target_update_freq = 160
    buffer_size = 50000
    eps_train = 0.1
    epoch = 20
    step_per_epoch = 20000
    step_per_collect = 32
    update_per_step = 0.05  # maps to update_step_num_gradient_steps_per_sample
    batch_size = 32

    # mid-training IGD diagnostic curve (separate from the final 30-repeat
    # evaluation done later by test_dqn.py) -- finer resolution early,
    # coarser once past switch_step, since improvement is typically
    # front-loaded and the tail doesn't need as fine a grid
    igd_eval_freq_early = 5000
    igd_eval_freq_late = 10000
    igd_eval_switch_step = 100000
    igd_eval_repeats = 10


if __name__ == "__main__":
    args = get_args()
    args.early_stop = True
    dqn_args = DQNargs()

    env = MOEAEnv(**vars(args))

    train_envs = SubprocVectorEnv(
        [lambda: MOEAEnv(**vars(args)) for _ in range(dqn_args.train_num)])

    train_envs.seed(args.seed)

    # observation_space is a Dict space with 'obs' and 'mask' keys;
    # Net only sees the 'obs' subspace
    state_shape = env.observation_space['obs'].shape
    action_shape = env.action_space.n

    net = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=dqn_args.hidden_sizes,
    ).to(dqn_args.device)

    # DiscreteQLearningPolicy handles action selection and epsilon-greedy exploration.
    # DQN handles the Q-learning update logic.
    policy = DiscreteQLearningPolicy(
        model=net,
        action_space=env.action_space,
        observation_space=env.observation_space,
        eps_training=dqn_args.eps_train,
        eps_inference=0.0,
    )

    algorithm = DQN(
        policy=policy,
        optim=AdamOptimizerFactory(lr=dqn_args.lr),
        gamma=dqn_args.gamma,
        n_step_return_horizon=dqn_args.n_step,
        target_update_freq=dqn_args.target_update_freq,
    )

    buf = VectorReplayBuffer(dqn_args.buffer_size, buffer_num=len(train_envs))
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)

    ao_tag = '' if args.adaptive_open else '_ao_false'
    log_path = os.path.join('./results/dqn', args.key, f'seed_{args.seed}{ao_tag}')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def train_fn(epoch, env_step):
        # Epsilon annealing schedule
        if env_step <= 100000:
            policy.set_eps_training(dqn_args.eps_train)
        elif env_step <= 500000:
            eps = dqn_args.eps_train - (env_step - 100000) / \
                400000 * (0.9 * dqn_args.eps_train)
            policy.set_eps_training(eps)
        else:
            policy.set_eps_training(0.1 * dqn_args.eps_train)

    # Ray workers for the mid-training eval hook -- separate pool from the
    # SubprocVectorEnv workers above, sized to igd_eval_repeats since the
    # only thing parallelised within a single problem's eval is the repeats.
    ray.init(num_cpus=dqn_args.igd_eval_repeats)

    igd_hook = DQNIGDEvalHook(
        policy=policy,
        args=args,
        eval_freq_early=dqn_args.igd_eval_freq_early,
        eval_freq_late=dqn_args.igd_eval_freq_late,
        switch_step=dqn_args.igd_eval_switch_step,
        n_repeats=dqn_args.igd_eval_repeats,
        base_train_fn=train_fn,
        save_path=os.path.join(log_path, "igd_curve.npz"),
    )

    trainer = OffPolicyTrainer(
        algorithm=algorithm,
        params=OffPolicyTrainerParams(
            max_epochs=dqn_args.epoch,
            epoch_num_steps=dqn_args.step_per_epoch,
            collection_step_num_env_steps=dqn_args.step_per_collect,
            collection_step_num_episodes=None,
            update_step_num_gradient_steps_per_sample=dqn_args.update_per_step,
            batch_size=dqn_args.batch_size,
            training_collector=train_collector,
            training_fn=igd_hook,
            logger=logger,
        ),
    )

    result = trainer.run()
    print(f'Finished training! Duration: {result.timing.total_time:.2f}s')

    # Save final policy unconditionally, same convention as ppo.py
    # (model.save() after model.learn() with no intermediate "best" tracking).
    model_path = os.path.join(log_path, args.key + 'policy.pth')
    torch.save(policy.state_dict(), model_path)
    print(f'Model saved to {model_path}')