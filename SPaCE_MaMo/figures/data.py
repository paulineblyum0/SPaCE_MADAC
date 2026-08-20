""" 
All plots and tables import from here. Maps files to the algorithms"
"""

from pathlib import Path

import numpy as np

# CONFIG 
KEY = "M_2_46_3"
SEEDS = [42, 123, 2022]
EVAL_SEED = 2022
REPEATS = 30
METRIC = "last" # "last" or "best" IGD

# Problems the training curve covers
TRAIN_PROBLEMS = ["DTLZ2_3", "WFG4_3", "WFG6_3"]

# Problems the final evaluation covers 
EVAL_PROBLEMS = ([f"DTLZ{i}_3" for i in (2, 4)]
                 + [f"WFG{i}_3" for i in range(4, 10)])


ALL_ALGORITHMS = {
    "DQN": (
        "results/dqn/trained/{key}/seed_{seed}/igd_curve.npz",
        "results/dqn/eval/{key}/seed_{seed}/{problem}_sd{eval_seed}_rp30.npz",
        SEEDS,
    ),
    "PPO": (
        "results/ppo/trained/{key}/seed_{seed}/igd_curve.npz",
        "results/ppo/eval/{key}/seed_{seed}/{problem}_sd{eval_seed}_rp30.npz",
        SEEDS,
    ),
    "SPACE": (
        "results/space_parallel/trained/{key}/space2_improvement/seed_{seed}/igd_curve.npz",
        "results/space_parallel/eval/{key}/space2_improvement/seed_{seed}/{problem}_sd{eval_seed}_rp30.npz",
        SEEDS,
    ),
    "MA-DAC": (
        "results/madac/igd_curves/vdn_ns_{key}_seed_{seed}.npz",
        "results/madac/eval/{key}/seed_{seed}/{problem}_sd{eval_seed}_rp30.npz",
        SEEDS,
    ),
    "Just Sizes": (
        "results/space_parallel/trained/{key}/space1/seed_{seed}/igd_curve.npz",
        "results/space_parallel/eval/{key}/space1/seed_{seed}/{problem}_sd{eval_seed}_rp30.npz",
        SEEDS,
    )
}

# Training curves

def _load_curve(path, problem, metric):
    
    d = np.load(path, allow_pickle=False)
    steps = d["steps"].astype(float)

    if "problems" in d.files:
        problems = [str(p) for p in d["problems"]]
        if problem not in problems:
            raise KeyError(f"{problem} not in {path} (has {problems})")
        y = d[metric][problems.index(problem)].mean(axis=1)
    else:
        key = f"{problem}_{metric}_mean"
        if key not in d.files:
            raise KeyError(f"{key} not in {path}")
        y = d[key]

    # Last minute addition to prepend policy eval at t0 for the training curves
    t0_path = path.with_name(path.stem + "_t0" + path.suffix)
    if t0_path.exists() and steps[0] > 0:
        d0 = np.load(t0_path, allow_pickle=False)
        key0 = f"{problem}_{metric}_mean"
        if key0 in d0.files:
            steps = np.concatenate([[0.0], steps])
            y = np.concatenate([d0[key0], y])

    return steps, y


def training_curve_matrix(series, label, problem, metric=METRIC):

    template, _, seeds = series[label]
    if template is None:
        return None, None
    curves = {}
    for seed in seeds:
        p = Path(template.format(key=KEY, seed=seed))
        if not p.exists():
            print(f"[skip] {p}")
            continue
        curves[seed] = _load_curve(p, problem, metric)
    if not curves:
        return None, None

    # Interpolate each seeds curve to the shared timestep 
    lo = max(s[0] for s, _ in curves.values())
    hi = min(s[-1] for s, _ in curves.values())
    grid = np.array(sorted({v for s, _ in curves.values() for v in s if lo <= v <= hi}))
    Y = np.array([np.interp(grid, s, y) for s, y in curves.values()])
    return grid, Y


def training_curve_matrix_avg(series, label, problems, metric=METRIC):
    # training curve values averaged across all training problems
    grids, curves = [], []
    for problem in problems:
        grid, Y = training_curve_matrix(series, label, problem, metric)
        if grid is None:
            continue
        grids.append(grid)
        curves.append(Y)
    if not curves:
        return None, None

    lo = max(g[0] for g in grids)
    hi = min(g[-1] for g in grids)
    common = np.array(sorted({v for g in grids for v in g if lo <= v <= hi}))
    per_problem_mean = np.array([
        np.mean([np.interp(common, g, y) for y in Y], axis=0)
        for g, Y in zip(grids, curves)
    ])  # (n_problems, n_points), seed axis already averaged out
    return common, per_problem_mean


# Final evaluation

def _eval_path(series, label, seed, problem):
    _, template, _ = series[label]
    return Path(template.format(key=KEY, seed=seed, problem=problem,
                                eval_seed=EVAL_SEED))


def final_igd_matrix(series, label, problem, metric=METRIC):
    """ Returns (n_seeds, n_repeats) scalar IGD. Column j is the same eval seed across every algorithm. """
    _, _, seeds = series[label]
    rows, kept = [], []
    for seed in seeds:
        p = _eval_path(series, label, seed, problem)
        if not p.exists():
            print(f"[skip] {p}")
            continue
        d = np.load(p, allow_pickle=True)
        rows.append([float(i[f"{metric}_igd"]) for i in d["info_stack"]])
        kept.append(seed)
    if not rows:
        return None, None
    return np.array(rows), kept


def within_episode_matrix(series, label, problem, pool_repeats=False):
    """
    pool_repeats=False: (n_seeds, n_generations),seed's final policy trace averaged over its own repeats (across-seed
    (policy) variability)
    pool_repeats=True: (n_seeds * n_repeats, n_generations) all repeats pooled
    """
    _, _, seeds = series[label]
    per_seed = []
    for seed in seeds:
        p = _eval_path(series, label, seed, problem)
        if not p.exists():
            print(f"[skip] {p}")
            continue
        d = np.load(p, allow_pickle=True)
        traces = [np.asarray(i["igd_his"], dtype=float) for i in d["info_stack"]]
        width = min(len(t) for t in traces)
        traces = [t[:width] for t in traces]
        if pool_repeats:
            per_seed.append(np.array(traces))
        else:
            per_seed.append(np.mean(traces, axis=0)[None, :])
    if not per_seed:
        return None, None
    width = min(r.shape[1] for r in per_seed)
    rows = np.concatenate([r[:, :width] for r in per_seed], axis=0)
    return np.arange(width), rows