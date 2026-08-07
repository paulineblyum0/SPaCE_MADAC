"""
Reward-sparsity / variance-decomposition diagnostic for the MaMo PPO env.

Purpose: turn the "ep_rew_mean is flat, MOEA/D's internal dynamics dominate
episode return" attribution into a measured finding.

Two things measured, per step, across rollouts:
  1. What fraction of steps carry zero reward at all (reward_type=0 is
     best-improvement-gated, so this is expected to be high).
  2. A one-way-ANOVA-style variance decomposition: how much of total reward
     variance is explained by *which instance* you're on, vs. how much of
     the leftover (instance-adjusted) variance is explained by *which
     action* was taken (broken out per action head, since the action space
     is MultiDiscrete([4,4,4,2])).

Actions are sampled uniformly at random rather than taken from a trained
policy. This isolates the environment's structural sensitivity to action
choice from whatever a specific policy has converged to -- a trained
policy that's collapsed onto similar actions everywhere would show low
action variance for the wrong reason (no exploration left to measure),
not because the environment doesn't respond to action.

Caveat: steps within an episode aren't independent (reward at t depends on
the trajectory so far), so this is an approximate, exploratory variance
decomposition -- same spirit as the fANOVA importance measure in Eimer et
al., not a rigorous nested ANOVA with proper error terms. Treat eta^2
values as descriptive effect sizes, not p-values.

Run from rerun/:
    python -m ppo.reward_diagnostic --key M_2_46_3
"""

import argparse
import numpy as np
import pandas as pd

from ppo.ppo_env import PPOMOEAEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, default="M_2_46_3")
    parser.add_argument("--episodes-per-instance", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adaptive-open", action="store_true", default=True)
    parser.add_argument("--budget-ratio", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=210)
    return parser.parse_args()


def eta_squared(values, group_labels):
    """
    One-way ANOVA effect size: fraction of total variance in `values`
    explained by group membership (`group_labels`).
        eta^2 = SS_between / SS_total
    0 = groups carry no information about the outcome.
    1 = outcome is fully determined by group membership.
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(group_labels)
    grand_mean = values.mean()
    ss_total = np.sum((values - grand_mean) ** 2)
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(labels):
        mask = labels == g
        group_mean = values[mask].mean()
        ss_between += mask.sum() * (group_mean - grand_mean) ** 2
    return ss_between / ss_total


def collect_random_action_rollouts(key, episodes_per_instance, seed, **env_kwargs):
    """
    Rolls out `episodes_per_instance` episodes per canonical instance under
    PURELY RANDOM actions (env.action_space.sample()), logging per-step
    reward, instance identity, and the 4 action components.

    Relies on MamoBase's own per-episode instance cycling (fun_index
    advances by one each reset()), so running N * num_instances episodes
    naturally covers each instance N times.
    """
    probe_env = PPOMOEAEnv(key=key, **env_kwargs)
    num_instances = len(probe_env._inner.env.func_select)
    probe_env.close()

    rows = []
    total_episodes = episodes_per_instance * num_instances

    for ep in range(total_episodes):
        env = PPOMOEAEnv(key=key, **env_kwargs)
        obs, _ = env.reset(seed=seed + ep)
        instance_name = str(env._inner.env.env_name)

        terminated = truncated = False
        t = 0
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rows.append({
                "episode": ep,
                "instance": instance_name,
                "step": t,
                "reward": reward,
                "a0": int(action[0]),
                "a1": int(action[1]),
                "a2": int(action[2]),
                "a3": int(action[3]),
            })
            t += 1
        env.close()

        if (ep + 1) % 5 == 0:
            print(f"  collected {ep + 1}/{total_episodes} episodes...")

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    env_kwargs = dict(
        adaptive_open=args.adaptive_open,
        budget_ratio=args.budget_ratio,
        population_size=args.population_size,
    )

    print(f"Collecting random-action rollouts for key={args.key} "
          f"({args.episodes_per_instance} episodes/instance)...")
    df = collect_random_action_rollouts(
        args.key, args.episodes_per_instance, args.seed, **env_kwargs
    )

    n_steps = len(df)
    n_zero = int((df["reward"] == 0).sum())

    print("\n" + "=" * 60)
    print("REWARD SPARSITY")
    print("=" * 60)
    print(f"Total steps collected:        {n_steps}")
    print(f"Episodes:                     {df['episode'].nunique()}")
    print(f"Instances covered:            {sorted(df['instance'].unique())}")
    print(f"Fraction of zero-reward steps: {n_zero / n_steps:.1%}")
    print(f"Reward mean / std (all steps): {df['reward'].mean():.4f} / {df['reward'].std():.4f}")
    print(f"Reward mean / std (nonzero only): "
          f"{df.loc[df['reward'] != 0, 'reward'].mean():.4f} / "
          f"{df.loc[df['reward'] != 0, 'reward'].std():.4f}")

    print("\n" + "=" * 60)
    print("VARIANCE DECOMPOSITION")
    print("=" * 60)

    eta2_instance = eta_squared(df["reward"], df["instance"])
    print(f"eta^2(instance)          = {eta2_instance:.3f}   "
          f"<- fraction of TOTAL reward variance explained by which instance")

    df["reward_resid"] = df["reward"] - df.groupby("instance")["reward"].transform("mean")
    resid_var = df["reward_resid"].var()
    print(f"\nResidual variance after removing instance effect: {resid_var:.4f} "
          f"(vs. total variance {df['reward'].var():.4f})")

    print("\nOf what's left after removing instance effect, per action head:")
    for a in ["a0", "a1", "a2", "a3"]:
        eta2_a = eta_squared(df["reward_resid"], df[a])
        print(f"  eta^2({a} | instance)  = {eta2_a:.3f}   "
              f"<- fraction of instance-adjusted variance explained by agent {a}'s action")

    print("\nInterpretation guide:")
    print("  Low eta^2 for all four action heads + high eta^2(instance) supports")
    print("  the 'MOEA/D instance dynamics dominate return, action barely moves it'")
    print("  hypothesis quantitatively. A surprisingly high eta^2 on one specific")
    print("  action head would instead point to which of the 4 sub-agents actually")
    print("  carries signal -- worth knowing either way.")

    out_path = f"reward_diagnostic_{args.key}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nRaw per-step data saved to {out_path}")


if __name__ == "__main__":
    main()