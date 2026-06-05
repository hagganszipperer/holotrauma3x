from __future__ import annotations

import torch


def airway_rescue_rate(area: torch.Tensor, critical: float) -> float:
    secured = (area >= critical).float().mean()
    return 100.0 * float(secured)


def trajectory_success_rate(flags: torch.Tensor) -> float:
    return 100.0 * float(flags.float().mean())


def complication_rate(severity: torch.Tensor, threshold: float = 7.0) -> float:
    flagged = (severity >= threshold).float().mean()
    return 100.0 * float(flagged)
