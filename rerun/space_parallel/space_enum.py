


from enum import Enum


class space_operation(Enum):
    NO_SPACE = 0        # round-robin baseline, matches existing PPO behavior
    JUST_SIZES = 1       # curriculum size growth only, uniform ordering (ablation)
    INSTANCE_STATE = 2   # full SPACE: value-informed ordering + size growth


class instance_ordering(Enum):
    ABSOLUTE = 0             # order by raw value estimate V(s0) per instance
    IMPROVEMENT = 1          # order by absolute change in V(s0) since last update
    RELATIVE_IMPROVEMENT = 2 # order by (change in V(s0)) / (previous V(s0))
    NONE = 3