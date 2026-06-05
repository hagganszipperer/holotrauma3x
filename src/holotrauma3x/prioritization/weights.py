from __future__ import annotations

from dataclasses import dataclass

from holotrauma3x.protocol import OptimizationSpec


@dataclass(frozen=True)
class PriorityWeights:
    airway: float
    stabilization: float
    function: float
    longterm: float

    @classmethod
    def from_spec(cls, spec: OptimizationSpec) -> PriorityWeights:
        if spec.priority_aware:
            return cls(
                airway=spec.weight_airway,
                stabilization=spec.weight_stabilization,
                function=spec.weight_function,
                longterm=spec.weight_longterm,
            )
        return cls(airway=1.0, stabilization=1.0, function=1.0, longterm=1.0)

    def is_ordered(self) -> bool:
        return self.airway > self.stabilization > self.function >= self.longterm
