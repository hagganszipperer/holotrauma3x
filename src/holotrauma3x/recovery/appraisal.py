from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from holotrauma3x.arrival.cohort import Sample
from holotrauma3x.pathway import HoloTrauma3x
from holotrauma3x.protocol import TriadInputs
from holotrauma3x.recovery.accuracy import (
    airway_error,
    condyle_mae,
    occlusal_class_i,
    positional_mae,
    rotational_mae,
)
from holotrauma3x.recovery.outcomes import airway_rescue_rate


class Appraisal:
    def __init__(self, device: torch.device, critical: float = 50.0) -> None:
        self.device = device
        self.critical = critical

    def evaluate(
        self, model: HoloTrauma3x, dataset: Dataset[Sample], batch_size: int = 8
    ) -> dict[str, float]:
        loader: DataLoader[Sample] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        maxilla_p, maxilla_t = [], []
        mandible_p, mandible_t = [], []
        condyle_p, condyle_t = [], []
        occlusion_p = []
        area_p, volume_p, volume_t = [], [], []
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for batch in loader:
                inputs = TriadInputs(
                    volume=batch["volume"].to(self.device),
                    points=batch["points"].to(self.device),
                    radiograph=batch["radiograph"].to(self.device),
                )
                output = model(inputs)
                maxilla_p.append(output.maxilla.cpu())
                maxilla_t.append(batch["maxilla"])
                mandible_p.append(output.mandible.cpu())
                mandible_t.append(batch["mandible"])
                condyle_p.append(output.condyle.cpu())
                condyle_t.append(batch["condyle"])
                occlusion_p.append(output.reasoning.injury.occlusion.cpu())
                area_p.append(output.reasoning.injury.airway_area.cpu())
                volume_p.append(output.reasoning.injury.airway_volume.cpu())
                volume_t.append(batch["airway_volume"])
        maxilla = (torch.cat(maxilla_p), torch.cat(maxilla_t))
        mandible = (torch.cat(mandible_p), torch.cat(mandible_t))
        return {
            "maxilla_mae_mm": positional_mae(*maxilla),
            "maxilla_rotation_deg": rotational_mae(*maxilla),
            "mandible_mae_mm": positional_mae(*mandible),
            "mandible_rotation_deg": rotational_mae(*mandible),
            "condyle_mae_mm": condyle_mae(torch.cat(condyle_p), torch.cat(condyle_t)),
            "airway_volume_mae_cm3": airway_error(torch.cat(volume_p), torch.cat(volume_t)),
            "occlusal_class_i_pct": occlusal_class_i(torch.cat(occlusion_p)),
            "airway_rescue_pct": airway_rescue_rate(torch.cat(area_p), self.critical),
        }
