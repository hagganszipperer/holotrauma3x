from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str, model: nn.Module, optimizer: torch.optim.Optimizer, seed: int, epoch: int
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "seed": seed,
        "epoch": epoch,
    }
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str, model: nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
