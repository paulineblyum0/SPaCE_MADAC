"""
RQ4: training/learning dynamics for DQN vs PPO vs MA-DAC.
Produces IGD training curves + AUC.

Run:  python -m figures.rq4_learning_dynamics
"""

from figures import data, report

SERIES = {label: data.ALL_ALGORITHMS[label] for label in ("DQN", "PPO", "MA-DAC")}

if __name__ == "__main__":
    # produce training curves per problem and averaged
    report.run("rq4", SERIES,
                eval_problems=None,
                train_problems=data.TRAIN_PROBLEMS,
                within_episode=False,
                training_curve=True,
                training_curve_avg=True,
                tables=False,
                auc=False,
                metric="last",
                pairwise_stats=False,
                pool_repeats=False,
                training_curve_band=None)

    # make auc table
    report.run("rq4", SERIES,
                eval_problems=None,
                train_problems=data.TRAIN_PROBLEMS,
                within_episode=False,
                training_curve=False,
                training_curve_avg=False,
                tables=False,
                auc=True,
                pairwise_stats=False,
                pool_repeats=False)