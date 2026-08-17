"""
RQ2: does the corrected DQN baseline (adaptive_open=True, from RQ1)
still lose to MA-DAC once it's compared like-for-like -- true single-
agent vs true multi-agent control of the same MOEA/D, at the paper's
own scale?

Single training seed (2022) for both algorithms, deliberately not the
3-seed average used everywhere else in this chapter: the MA-DAC paper
itself reports one seed, so this comparison mirrors that protocol
rather than improving on it, and RQ3 is where the 3-seed, PPO-inclusive
extension happens. Say this explicitly in the caption/text so a reader
doesn't mistake it for an oversight.

Statistics: pooled Wilcoxon rank-sum per problem (rq2_statistics). With
one seed per side this reduces to a plain 30-vs-30 repeat-level test --
there is no seed-pairing question to raise here the way there was in
RQ1.

Within-episode band: pool_repeats=True, so the band is spread across
the 30 evaluation repeats rather than across seeds (there is only one
seed, so the seed-only band would otherwise be zero-width). Caption
should say the band is repeat-level noise, not policy variability.

Run from rerun/:  python -m figures.rq2_dqn_vs_madac
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