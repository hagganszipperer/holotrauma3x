from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ArrivalSpec:
    cohort_size: int = 64
    volume_size: int = 32
    point_count: int = 1024
    radiograph_size: int = 64
    modalities: tuple[str, ...] = ("cbct", "dental", "cephalogram")
    fps_points: int = 16384
    outlier_k: int = 20
    outlier_std: float = 2.5
    augment_prob: float = 0.5


@dataclass
class VolumetricSpec:
    patch: int = 8
    layers: int = 10
    dim: int = 512
    heads: int = 8
    feedforward: int = 2048
    dropout: float = 0.15


@dataclass
class PointSpec:
    radii: tuple[float, ...] = (0.12, 0.25, 0.5, 1.0)
    samples: tuple[int, ...] = (512, 256, 128, 64)
    out_dim: int = 256


@dataclass
class RadiographSpec:
    out_dim: int = 256
    reduction: int = 16


@dataclass
class FusionSpec:
    weight_volume: float = 1.5
    weight_dental: float = 0.8
    weight_cephalogram: float = 1.2
    enc_dim: int = 1024
    heads: int = 8


@dataclass
class EncoderSpec:
    volumetric: VolumetricSpec = field(default_factory=VolumetricSpec)
    point: PointSpec = field(default_factory=PointSpec)
    radiograph: RadiographSpec = field(default_factory=RadiographSpec)
    fusion: FusionSpec = field(default_factory=FusionSpec)
    fusion_mode: str = "cross_modal"


@dataclass
class CoReasoningSpec:
    clip_dim: int = 1024
    proj_dim: int = 768
    decoder_layers: int = 6
    decoder_heads: int = 8
    decoder_hidden: int = 512
    vocab: int = 4800
    knowledge_size: int = 15400
    top_m: int = 7
    confidence_threshold: float = 0.75
    engine: str = "grammar"
    temperature: float = 0.3
    enabled: bool = True


@dataclass
class OptimizationSpec:
    weight_airway: float = 10.0
    weight_stabilization: float = 5.0
    weight_function: float = 2.0
    weight_longterm: float = 1.0
    airway_beta: tuple[float, float] = (5.0, 1.8)
    airway_critical: float = 50.0
    stabilization_gamma: tuple[float, float] = (3.0, 2.0)
    occlusion_alpha: tuple[float, float, float, float] = (2.0, 1.5, 1.0, 1.0)
    learning_rate: float = 2e-3
    beta1: float = 0.9
    beta2: float = 0.999
    max_iterations: int = 120
    priority_aware: bool = True


@dataclass
class RobotSpec:
    dof: int = 7
    exploration: float = 0.35
    safety_margin: float = 5.0
    alpha_distance: float = 1.0
    alpha_time: float = 2.0
    max_iterations: int = 3500
    connection_radius: float = 0.5
    step_size: float = 0.1
    goal_threshold: float = 0.05
    policy_layers: int = 4
    policy_units: int = 256
    policy_dropout: float = 0.25
    obstacle_grid: int = 64
    uncertainty_grid: int = 32
    velocity_limits: tuple[float, ...] = (1.73, 1.73, 1.73, 2.30, 2.30, 2.88, 2.88)
    acceleration_limits: tuple[float, ...] = (2.20, 2.20, 2.20, 3.30, 3.30, 4.40, 4.40)
    neural_guidance: bool = True


@dataclass
class TrainingSpec:
    epochs: int = 150
    patience: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-4
    seed: int = 42
    folds: int = 5
    centers: int = 12
    device: str = "cpu"
    max_steps: int = 0


@dataclass
class Protocol:
    experiment: str = "main"
    arrival: ArrivalSpec = field(default_factory=ArrivalSpec)
    encoder: EncoderSpec = field(default_factory=EncoderSpec)
    coreasoning: CoReasoningSpec = field(default_factory=CoReasoningSpec)
    optimization: OptimizationSpec = field(default_factory=OptimizationSpec)
    robot: RobotSpec = field(default_factory=RobotSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)


@dataclass
class TriadInputs:
    volume: torch.Tensor
    points: torch.Tensor
    radiograph: torch.Tensor


@dataclass
class InjuryState:
    airway_area: torch.Tensor
    airway_volume: torch.Tensor
    fracture_offsets: torch.Tensor
    occlusion: torch.Tensor
    fiss: torch.Tensor


@dataclass
class SurgicalPlan:
    maxilla: torch.Tensor
    mandible: torch.Tensor
    condyle: torch.Tensor
    airway_volume: torch.Tensor
    airway_area: torch.Tensor
    trajectory: torch.Tensor
    confidence: torch.Tensor
    narrative: list[str]
