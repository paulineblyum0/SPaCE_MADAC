"""
RQ3: DQN vs PPO vs MA-DAC, 3 seeds each. Run from rerun/: python -m figures.rq3_algorithm_comparison
"""

from figures import data, report, results_tables

# DQN = adaptive_open=True (RQ1's corrected baseline). MA-DAC uses all 3 seeds here (RQ2 uses 1).
SERIES = {label: data.ALL_ALGORITHMS[label] for label in ("DQN", "PPO", "MA-DAC")}
PAIRS = [("DQN", "PPO"), ("PPO", "MA-DAC")]

if __name__ == "__main__":
    # Plots + descriptives: pooled across all 90 (seed x repeat) values, no test.
    report.run("rq3", SERIES, eval_problems=data.EVAL_PROBLEMS,
                pairwise_stats=False, pool_repeats=True)

    # Significance: paired per-seed Wilcoxon (repeat j = same env draw across
    # algorithms, confirmed via seeding order in test_dqn/ppo/madac.py), not
    # the pooled rank-sum report.run's pairwise_stats would otherwise call.
    results_tables.rq3_statistics(SERIES, data.EVAL_PROBLEMS, PAIRS, prefix="rq3_")