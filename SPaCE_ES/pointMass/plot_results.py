"""
plot_results.py

Produces two plots from the output of baselines_spl.py:

  1. Learning curve — mean eval reward over training steps for SPL vs RR,
     averaged across seeds with ±1 std shading.

  2. Curriculum membership plot — which instances are active in the curriculum
     at each iteration (SPL only), styled after the SPaCE paper's Figure 3.
     One row per seed, stacked vertically, plus a RR row at the bottom.

Usage:
    python plot_results.py --spl_dirs results/cpm_ppo_spl0 results/cpm_ppo_spl1 ...
                           --rr_dirs  results/cpm_ppo_rr0  results/cpm_ppo_rr1  ...
                           --outdir   plots/

All arguments have defaults matching the run script's output structure.
"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

matplotlib.rc('font', size=14)
PALETTE = sns.color_palette('colorblind')


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def read_reward_file(path):
    """
    Read an eval_reward.txt or test_reward.txt file.
    Each line: mean_reward <tab> step <tab> total_steps <tab> n_instances
    Returns array of (total_steps, mean_reward).
    """
    entries = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                reward = float(parts[0])
                total_steps = float(parts[2])
                entries.append((total_steps, reward))
    return np.array(entries)


def read_curriculum_file(path):
    """
    Read an instance_curriculum.txt file.
    Each line is a list of instance indices active at that iteration.
    Returns list of lists of ints.
    """
    curriculum = []
    with open(path) as f:
        for line in f:
            line = line.strip()[1:-1]  # strip [ ]
            indices = []
            for token in line.split():
                try:
                    indices.append(int(token))
                except ValueError:
                    pass
            curriculum.append(indices)
    return curriculum


def read_curriculum_size_file(path):
    """Read curriculum_size.txt — one int per line."""
    sizes = []
    with open(path) as f:
        for line in f:
            try:
                sizes.append(int(line.strip()))
            except ValueError:
                pass
    return sizes


# ---------------------------------------------------------------------------
# Plot 1: Learning curve
# ---------------------------------------------------------------------------

def plot_learning_curve(spl_dirs, rr_dirs, outdir):
    fig, ax = plt.subplots(figsize=(10, 5))

    def load_rewards(dirs):
        all_rewards = []
        for d in dirs:
            path = os.path.join(d, 'eval_reward.txt')
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping.")
                continue
            data = read_reward_file(path)
            if len(data):
                all_rewards.append(data[:, 1])  # just the reward column
        return all_rewards

    spl_rewards = load_rewards(spl_dirs)
    rr_rewards = load_rewards(rr_dirs)

    def plot_condition(rewards_list, label, color):
        if not rewards_list:
            return
        min_len = min(len(r) for r in rewards_list)
        arr = np.array([r[:min_len] for r in rewards_list])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.arange(min_len)
        ax.plot(x, mean, label=label, color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

    plot_condition(spl_rewards, 'SPL', PALETTE[0])
    plot_condition(rr_rewards, 'RR', PALETTE[1])

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Eval Reward')
    ax.set_title('Learning Curve — CPM Point Mass Gate')
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, 'learning_curve.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Curriculum membership
# ---------------------------------------------------------------------------

def plot_curriculum(spl_dirs, n_instances, outdir):
    """
    For each seed: read instance_curriculum.txt and show which instances
    are in the active set at each iteration as a coloured step-fill plot.
    Seeds are stacked vertically. A final row shows RR (all instances always).
    """
    n_seeds = len(spl_dirs)
    # Read all curricula
    curricula = []
    for d in spl_dirs:
        path = os.path.join(d, 'instance_curriculum.txt')
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            curricula.append([])
        else:
            curricula.append(read_curriculum_file(path))

    if not any(curricula):
        print("No curriculum files found, skipping curriculum plot.")
        return

    max_iters = max(len(c) for c in curricula if c)

    # Build per-instance binary presence arrays: shape (n_seeds, n_instances, max_iters)
    presence = np.zeros((n_seeds, n_instances, max_iters))
    for s, curriculum in enumerate(curricula):
        for t, active in enumerate(curriculum):
            for idx in active:
                if idx < n_instances:
                    presence[s, idx, t] = 1

    fig, axes = plt.subplots(
        n_seeds + 1, 1,
        figsize=(14, 2 * (n_seeds + 1)),
        sharex=True
    )

    x = np.arange(max_iters)

    for s in range(n_seeds):
        ax = axes[s]
        for inst_idx in range(n_instances):
            y = presence[s, inst_idx]
            # offset each instance slightly so overlaps are visible
            color = PALETTE[inst_idx % len(PALETTE)]
            ax.fill_between(x, y * (inst_idx + 1), y * inst_idx,
                            step='post', color=color, alpha=0.8)
        ax.set_yticks([])
        ax.set_ylabel(f'Seed {s}', fontsize=10, rotation=0, labelpad=40)
        ax.set_ylim([0, n_instances])

    # RR row: all instances always active
    ax_rr = axes[-1]
    ax_rr.fill_between(x, n_instances, 0, step='post',
                       color=PALETTE[6 % len(PALETTE)], alpha=0.6)
    ax_rr.set_yticks([])
    ax_rr.set_ylabel('RR', fontsize=10, rotation=0, labelpad=40)
    ax_rr.set_ylim([0, n_instances])
    ax_rr.set_xlabel('Iteration')

    fig.suptitle('Curriculum Membership — CPM Point Mass Gate', y=1.01)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, 'curriculum_membership.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot CPM SPACE results')
    parser.add_argument(
        '--spl_dirs', nargs='+',
        default=[f'results/cpm_ppo_spl{s}' for s in range(10)],
        help='Output directories for SPL seeds'
    )
    parser.add_argument(
        '--rr_dirs', nargs='+',
        default=[f'results/cpm_ppo_rr{s}' for s in range(10)],
        help='Output directories for RR seeds'
    )
    parser.add_argument(
        '--n_instances', type=int, default=100,
        help='Total number of training instances'
    )
    parser.add_argument(
        '--outdir', default='plots',
        help='Directory to save plots'
    )
    args = parser.parse_args()

    # Filter to dirs that actually exist
    spl_dirs = [d for d in args.spl_dirs if os.path.isdir(d)]
    rr_dirs = [d for d in args.rr_dirs if os.path.isdir(d)]

    if not spl_dirs and not rr_dirs:
        print("No result directories found. Run baselines_spl.py first.")
        exit(1)

    plot_learning_curve(spl_dirs, rr_dirs, args.outdir)
    if spl_dirs:
        plot_curriculum(spl_dirs, args.n_instances, args.outdir)