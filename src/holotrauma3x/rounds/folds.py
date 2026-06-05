from __future__ import annotations

import torch


def patient_folds(size: int, folds: int, seed: int) -> list[tuple[list[int], list[int]]]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(size, generator=generator).tolist()
    splits: list[list[int]] = [order[i::folds] for i in range(folds)]
    result: list[tuple[list[int], list[int]]] = []
    for index in range(folds):
        validation = sorted(splits[index])
        training = sorted(i for j, split in enumerate(splits) if j != index for i in split)
        result.append((training, validation))
    return result


def leave_one_center_out(centers: list[int]) -> list[tuple[list[int], list[int]]]:
    unique = sorted(set(centers))
    result: list[tuple[list[int], list[int]]] = []
    for held in unique:
        training = [i for i, center in enumerate(centers) if center != held]
        validation = [i for i, center in enumerate(centers) if center == held]
        result.append((training, validation))
    return result
