"""
cartpole_space_corrected.py

Corrected implementation of SPACE (Self-Paced Context Evaluation) on CartPole,
based on the logic from baselines_spl.py (the real SPACE implementation).

Three bugs fixed vs the original Pytorch CartPole.ipynb:

  Bug 1 (Round Robin): run_cartpole_rr used local model/memory/epsilon that
    were never passed into train_the_model() or model.get_action(), so the
    local model never actually trained. Fixed by making RR self-contained with
    explicit parameter passing.

  Bug 2 (SPACE - critical): The curriculum ordering was computed but never
    actually used to gate which instances the agent trained on. env.length
    cycled unconditionally through all instances regardless of curr_set.
    Fixed by cycling through curr_set instead of instances[curr].

  Bug 3 (SPACE - ordering metric): The notebook sorted by absolute Q-value
    ascending. The real SPACE (Algorithm 1) ranks by *improvement* in Q-value
    (V_t - V_{t-1}), sorted descending (most-improved first). Fixed.

Gymnasium migration notes (gym → gymnasium):
  - gymnasium.make('CartPole-v1')  (v0 is removed in gymnasium)
  - env.reset() returns (obs, info) -- unpack accordingly
  - env.step() returns (obs, reward, terminated, truncated, info)
    done = terminated or truncated
  - env.seed() removed -- pass seed via env.reset(seed=seed) on first call
    and via np/random/torch for reproducibility

Usage:
    python cartpole_space.py

Requires: gymnasium, torch, numpy, matplotlib.
"""

import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Hyperparameters (match original notebook)
# ---------------------------------------------------------------------------
EPISODES      = 300
DISCOUNT      = 0.99
BATCH_SIZE    = 32
MIN_MEMORY    = 100          # minimum replay buffer size before training
EPSILON_START = 0.99
EPSILON_DECAY = 0.99         # per-step decay (matches original notebook)
MIN_EPSILON   = 0.01
ETA           = 0.025         # plateau threshold: |delta_Q| <= eta * |Q|
KAPPA         = 1             # number of instances to add when plateau detected

INSTANCES     = [0.3, 0.4, 0.5]   # pole lengths (easy → hard)
N_SEEDS       = 3                  # seeds to run per condition

# ---------------------------------------------------------------------------
# DQN Model
# ---------------------------------------------------------------------------
# Input: CartPole obs (4 dims) + pole length appended = 5 dims
# Output: Q-values for each action (2)

def mish(x):
    """Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))"""
    return x * torch.tanh(nn.functional.softplus(x))


class DQN(nn.Module):
    def __init__(self, input_dim=5, n_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Mish(),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, 64),
            nn.Mish(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer + training step (shared helper, takes explicit objects)
# ---------------------------------------------------------------------------

def train_step(model, target_model, memory, optimizer, batch_size=BATCH_SIZE,
               discount=DISCOUNT):
    """Single gradient step from replay buffer. Returns loss or None."""
    if len(memory) < MIN_MEMORY:
        return None

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states      = torch.FloatTensor(np.array(states))
    actions     = torch.LongTensor(actions).unsqueeze(1)
    rewards_t   = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(np.array(next_states))
    dones_t     = torch.FloatTensor(dones)

    # Current Q
    q_values = model(states).gather(1, actions).squeeze(1)

    # Target Q (from frozen target network)
    with torch.no_grad():
        max_next_q = target_model(next_states).max(1)[0]
        target_q   = rewards_t + discount * max_next_q * (1 - dones_t)

    loss = nn.functional.mse_loss(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def get_action(model, state, epsilon, n_actions=2):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    with torch.no_grad():
        q = model(torch.FloatTensor(state).unsqueeze(0))
        return q.argmax().item()


def make_state(obs, length):
    """Append pole length as context to the observation."""
    return np.append(obs, length)


def reset_env(env, length, seed=None):
    """
    Set pole length and reset env.
    Gymnasium reset() returns (obs, info); seed is passed on first call only.
    Returns obs array.
    """
    env.unwrapped.length = length
    env.unwrapped.polemass_length = env.unwrapped.masspole * length
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    return obs


# ---------------------------------------------------------------------------
# SPACE helpers (translated from baselines_spl.py)
# ---------------------------------------------------------------------------

def get_instance_evals(model, env, instances):
    """
    Query V(s_0) for every instance.
    s_0 = initial obs with instance feature (pole length) appended.
    We use max Q(s_0) as the value estimate (DQN equivalent of V(s_0)).

    Mirrors baselines_spl.get_instance_evals: resets to each instance,
    queries the network at s_0, restores state.
    """
    evals = []
    for length in instances:
        obs = reset_env(env, length)
        s0 = make_state(obs, length)
        with torch.no_grad():
            q = model(torch.FloatTensor(s0).unsqueeze(0))
            val = q.max().item()
        evals.append(val)
    return np.array(evals)


def order_instances_improvement(model, env, instances, last_evals):
    """
    Rank instances by improvement in V(s_0) since last curriculum update.
    Returns (indices sorted by improvement desc, current evals).

    Mirrors baselines_spl.order_instances_improvement:
      improvement = evals - last_evals
      return np.argsort(improvement)[::-1], evals
    """
    evals = get_instance_evals(model, env, instances)
    improvement = evals - last_evals
    indices = np.argsort(improvement)[::-1]   # descending: most-improved first
    return indices, evals


def get_mean_q(model, env, curr_set):
    """
    Mean V(s_0) over the current curriculum set.
    Used for the plateau detection condition.

    Mirrors baselines_spl.get_mean_q.
    """
    vals = []
    for length in curr_set:
        obs = reset_env(env, length)
        s0 = make_state(obs, length)
        with torch.no_grad():
            q = model(torch.FloatTensor(s0).unsqueeze(0))
            vals.append(q.max().item())
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Round Robin baseline (Bug 1 fixed: fully self-contained)
# ---------------------------------------------------------------------------

def run_cartpole_rr(seed):
    """
    Round Robin baseline: cycle through all instances in fixed order,
    training a DQN.

    Bug 1 fix: the original notebook defined local model/memory/epsilon
    inside run_cartpole_rr but then called the global train_the_model() and
    model.get_action(), so the local objects were never actually trained.
    Here everything is self-contained with explicit passing.

    Gymnasium: reset() returns (obs, info); step() returns 5-tuple with
    terminated and truncated; seed passed via reset(seed=...).
    """
    # Gymnasium: CartPole-v1 (v0 removed); seed via reset(), not env.seed()
    env = gym.make('CartPole-v1')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model        = DQN()
    target_model = DQN()
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer    = optim.Adam(model.parameters())
    memory       = deque(maxlen=10000)
    epsilon      = EPSILON_START

    curr_idx        = 0
    episode_rewards = []
    rewards_by_inst = {l: [] for l in INSTANCES}
    first_episode   = True

    for episode in range(EPISODES):
        length = INSTANCES[curr_idx]
        # Pass seed only on the very first reset for reproducibility
        obs = reset_env(env, length, seed=seed if first_episode else None)
        first_episode = False
        state = make_state(obs, length)

        done = False
        ep_reward = 0

        while not done:
            action = get_action(model, state, epsilon)
            # Gymnasium step() returns (obs, reward, terminated, truncated, info)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = make_state(next_obs, length)
            ep_reward += reward

            r = reward if not done else -100
            memory.append((state, action, r, next_state, float(done)))
            train_step(model, target_model, memory, optimizer)

            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY

            state = next_state

        episode_rewards.append(ep_reward)
        rewards_by_inst[length].append(ep_reward)

        # Update target network each episode (matches original notebook)
        target_model.load_state_dict(model.state_dict())

        # Advance to next instance (round-robin over all instances)
        curr_idx = (curr_idx + 1) % len(INSTANCES)

    env.close()
    return episode_rewards, rewards_by_inst


# ---------------------------------------------------------------------------
# SPACE (all three bugs fixed)
# ---------------------------------------------------------------------------

def run_cartpole_space(seed):
    """
    SPACE curriculum: starts with the easiest instance, grows the active set
    when V(s_0) plateaus, and always prioritises training on the instances
    with the most recent improvement.

    Bug 2 fix: training cycles through curr_set (the active curriculum),
      not unconditionally through all instances.
    Bug 3 fix: ordering uses improvement in V(s_0) since last update,
      not absolute V(s_0) ascending.

    Gymnasium: reset() returns (obs, info); step() returns 5-tuple with
    terminated and truncated; seed passed via reset(seed=...).
    """
    env = gym.make('CartPole-v1')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model        = DQN()
    target_model = DQN()
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer    = optim.Adam(model.parameters())
    memory       = deque(maxlen=10000)
    epsilon      = EPSILON_START

    # Curriculum state (mirrors baselines_spl main loop)
    n_instances   = 1                                  # active set size
    curr_set      = [INSTANCES[0]]                     # start: easiest only
    ordered_idxs  = list(range(len(INSTANCES)))        # index ordering
    last_evals    = np.zeros(len(INSTANCES))           # V(s_0) at last update
    old_mean_q    = -np.inf
    curr_pos      = 0                                  # position in curr_set

    episode_rewards  = []
    rewards_by_inst  = {l: [] for l in INSTANCES}
    curriculum_sizes = []
    curriculum_log   = []
    first_episode    = True

    for episode in range(EPISODES):
        # --- Bug 2 fix: pick from curr_set, not from all instances ---
        length = curr_set[curr_pos % len(curr_set)]
        obs = reset_env(env, length, seed=seed if first_episode else None)
        first_episode = False
        state = make_state(obs, length)

        done = False
        ep_reward = 0

        while not done:
            action = get_action(model, state, epsilon)
            # Gymnasium step() returns (obs, reward, terminated, truncated, info)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = make_state(next_obs, length)
            ep_reward += reward

            r = reward if not done else -100
            memory.append((state, action, r, next_state, float(done)))
            train_step(model, target_model, memory, optimizer)

            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY

            state = next_state

        episode_rewards.append(ep_reward)
        rewards_by_inst[length].append(ep_reward)
        target_model.load_state_dict(model.state_dict())

        curr_pos += 1
        

        # After one full pass through the current curriculum set, update
        if episode % 50 == 0 and episode > 0 and len(memory) >= MIN_MEMORY:
            # --- Bug 3 fix: rank by improvement in V(s_0) ---
            ordered_idxs, last_evals = order_instances_improvement(
                model, env, INSTANCES, last_evals
            )

            # Plateau detection (mirrors baselines_spl main loop):
            #   delta_q <= eta * |old_mean_q|  =>  add kappa instances
            mean_q  = get_mean_q(model, env, curr_set)
            delta_q = abs(abs(mean_q) - abs(old_mean_q))
            
            print(f"Episode {episode}: curriculum size={n_instances}, curr_set={curr_set}, mean_q={mean_q:.3f}, delta_q={delta_q:.3f}")

            if delta_q <= ETA * abs(old_mean_q) and n_instances < len(INSTANCES):
                n_instances = min(n_instances + KAPPA, len(INSTANCES))

            old_mean_q = mean_q

            # Rebuild active set: top-n_instances by improvement order
            curr_set = [INSTANCES[ordered_idxs[i]] for i in range(n_instances)]
            curr_pos = 0

        curriculum_sizes.append(n_instances)
        curriculum_log.append(list(curr_set))

    env.close()
    return episode_rewards, rewards_by_inst, curriculum_sizes, curriculum_log


# ---------------------------------------------------------------------------
# Run experiments over multiple seeds
# ---------------------------------------------------------------------------

def run_experiments(n_seeds=N_SEEDS):
    rr_all    = []
    space_all = []
    space_curriculum_sizes = []

    for seed in range(n_seeds):
        print(f"  Seed {seed}...")

        rr_rewards, _ = run_cartpole_rr(seed)
        rr_all.append(rr_rewards)

        space_rewards, _, cs, _ = run_cartpole_space(seed)
        space_all.append(space_rewards)
        space_curriculum_sizes.append(cs)

    return (np.array(rr_all),
            np.array(space_all),
            np.array(space_curriculum_sizes))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(rr_all, space_all, space_curriculum_sizes):
    episodes = np.arange(EPISODES)

    mean_rr    = rr_all.mean(axis=0)
    std_rr     = rr_all.std(axis=0)
    mean_space = space_all.mean(axis=0)
    std_space  = space_all.std(axis=0)

    fig = plt.figure(figsize=(18, 7.5), dpi=50)
    plt.plot(np.arange(len(mean_rr)), mean_rr, label='RR')
    plt.fill_between(np.arange(len(mean_rr)), mean_rr + std_rr, mean_rr - std_rr, alpha=0.2)
    plt.plot(np.arange(len(mean_space)), mean_space, label='SPACE')
    plt.fill_between(np.arange(len(mean_space)), mean_space + std_space, mean_space - std_space, alpha=0.2)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.savefig('cartpole_space_results.png', dpi=50, bbox_inches='tight')
    plt.show()
    print("Plot saved to cartpole_space_results.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"Running {N_SEEDS} seeds × 2 conditions × {EPISODES} episodes...")
    rr_all, space_all, space_cs = run_experiments(N_SEEDS)

    print("\nMean final reward (last 10 episodes):")
    print(f"  Round Robin: {rr_all[:, -10:].mean():.1f}")
    print(f"  SPACE:       {space_all[:, -10:].mean():.1f}")

    plot_results(rr_all, space_all, space_cs)