from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from holotrauma3x.arrival.cohort import Sample
from holotrauma3x.pathway import HoloTrauma3x
from holotrauma3x.protocol import TrainingSpec, TriadInputs
from holotrauma3x.rounds.devices import select_device
from holotrauma3x.rounds.seeding import seeded_generator


@dataclass
class TrainHistory:
    step_losses: list[float] = field(default_factory=list)
    epochs: int = 0

    @property
    def final(self) -> float:
        return self.step_losses[-1] if self.step_losses else float("nan")


class Trainer:
    def __init__(self, spec: TrainingSpec) -> None:
        self.spec = spec
        self.device = select_device(spec.device)

    def loss(self, model: HoloTrauma3x, batch: Sample) -> torch.Tensor:
        inputs = TriadInputs(
            volume=batch["volume"].to(self.device),
            points=batch["points"].to(self.device),
            radiograph=batch["radiograph"].to(self.device),
        )
        output = model(inputs)
        injury = output.reasoning.injury
        return (
            F.mse_loss(output.maxilla, batch["maxilla"].to(self.device))
            + F.mse_loss(output.mandible, batch["mandible"].to(self.device))
            + F.mse_loss(output.condyle, batch["condyle"].to(self.device))
            + 0.01 * F.mse_loss(injury.airway_area, batch["airway_area"].to(self.device))
            + 0.01 * F.mse_loss(injury.airway_volume, batch["airway_volume"].to(self.device))
            + F.mse_loss(injury.occlusion, batch["occlusion"].to(self.device))
            + 0.01 * F.mse_loss(injury.fiss, batch["fiss"].to(self.device))
            + F.mse_loss(injury.fracture_offsets, batch["fracture_offsets"].to(self.device))
        )

    def fit(self, model: HoloTrauma3x, dataset: Dataset[Sample]) -> TrainHistory:
        model.to(self.device)
        loader: DataLoader[Sample] = DataLoader(
            dataset,
            batch_size=self.spec.batch_size,
            shuffle=True,
            drop_last=True,
            generator=seeded_generator(self.spec.seed),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.spec.learning_rate)
        history = TrainHistory()
        best = float("inf")
        stale = 0
        steps = 0
        for epoch in range(self.spec.epochs):
            model.train()
            epoch_loss = 0.0
            batches = 0
            for batch in loader:
                optimizer.zero_grad()
                value = self.loss(model, batch)
                torch.autograd.backward(value)
                optimizer.step()
                history.step_losses.append(float(value.item()))
                epoch_loss += float(value.item())
                batches += 1
                steps += 1
                if self.spec.max_steps and steps >= self.spec.max_steps:
                    history.epochs = epoch + 1
                    return history
            mean_loss = epoch_loss / max(1, batches)
            if mean_loss + 1e-4 < best:
                best = mean_loss
                stale = 0
            else:
                stale += 1
            if stale >= self.spec.patience:
                history.epochs = epoch + 1
                return history
        history.epochs = self.spec.epochs
        return history
