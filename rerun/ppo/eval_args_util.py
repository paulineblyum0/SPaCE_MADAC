import argparse


def build_eval_args(train_args, save_history=False):
    """Builds an eval-time argparse to mirror the training config, so eval always matches training exactly."""
    return argparse.Namespace(
        key=train_args.key,
        seed=train_args.seed,
        budget_ratio=getattr(train_args, "budget_ratio", 100),
        population_size=getattr(train_args, "population_size", 210),
        adaptive_open=getattr(train_args, "adaptive_open", True),
        early_stop=getattr(train_args, "early_stop", False),
        pi_arch=getattr(train_args, "pi_arch", [64, 64]),
        vf_arch=getattr(train_args, "vf_arch", [64, 64]),
        save_history=save_history,
    )