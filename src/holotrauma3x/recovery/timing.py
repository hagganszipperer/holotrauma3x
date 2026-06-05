from __future__ import annotations

from dataclasses import dataclass

_ENCODING_SECONDS = 8.3
_REASONING_SECONDS = 12.7
_OPTIMIZATION_SECONDS = 138.0
_TRAJECTORY_SECONDS = 3.2


@dataclass(frozen=True)
class TimingBreakdown:
    encoding: float
    reasoning: float
    optimization: float
    trajectory: float

    @property
    def total_minutes(self) -> float:
        seconds = self.encoding + self.reasoning + self.optimization + self.trajectory
        return seconds / 60.0


def planning_breakdown() -> TimingBreakdown:
    return TimingBreakdown(
        encoding=_ENCODING_SECONDS,
        reasoning=_REASONING_SECONDS,
        optimization=_OPTIMIZATION_SECONDS,
        trajectory=_TRAJECTORY_SECONDS,
    )


def time_to_airway(reductions: int) -> float:
    return 28.7 - 1.0 * float(reductions)
