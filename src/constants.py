from enum import Enum


class TerminationReasons(Enum):
    NotEnded = 0
    NoBetterStrategy = 1
    Converged = 2
    IterationsLimit = 3
    ProbablyConverged = 4
