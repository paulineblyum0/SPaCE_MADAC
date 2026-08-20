"""
Plots and tables for RQ2: does the corrected DQN baseline still lose to MA-DAC once it's directly comparable.

Single seed (2022) for both algorithms deliberately, not the 3-seed average used elsewhere. 
To mirror the MA-DAC paper's own one-seed protocol.
"""

from figures import data, report

PAPER_SEED = 2022


def _single_seed(label):
    curve_template, eval_template, _ = data.ALL_ALGORITHMS[label]
    return (curve_template, eval_template, [PAPER_SEED])


SERIES = {
    "DQN": _single_seed("DQN"),
    "MA-DAC": _single_seed("MA-DAC"),
}

if __name__ == "__main__":
    report.run("rq2", SERIES, eval_problems=data.EVAL_PROBLEMS,
                pairwise_stats=True, pool_repeats=True)