# Implementation Map

Every numbered equation, algorithm box, reported table, ablation, and metric mapped to the
file and module that carries it. The package is organized as the clinical pathway of an
emergency trauma patient: `arrival -> workup -> consult -> prioritization -> intervention
-> recovery`, with `rounds` (training) and `protocol`/`orders` as cross-cutting concerns.

(All paper provenance lives here. Source files carry no comments and no docstrings.)

## Problem definition and objective

| paper item | ref | file | module symbol | notes |
|---|---|---|---|---|
| Emergency plan P*, SE(3) maxilla/mandible, condyle theta, trajectory tau | Methods Problem Def. | `protocol.py` | `SurgicalPlan`, `TriadInputs` | typed plan + input containers |
| Priority hierarchy w_L>>w_A>w_F>w_LT | Methods Problem Def.; Eq.1/Eq.8 | `prioritization/weights.py` | `PriorityWeights` | w_L=10, w_A=5, w_F=2, w_LT=1 |
| Eq.1 arg-min weighted objective | Eq.1 | `prioritization/planner.py` | `PriorityPlanner.solve` | Adam, hard-constraint guards |

## Triadic anatomical encoder (Section: Rapid Triadic Anatomical Encoder)

| paper item | ref | file | module symbol | notes |
|---|---|---|---|---|
| Eq.2 3D ViT encoder stack | Eq.2 | `workup/volumetric.py` | `VolumetricTransformer` | patch 8^3, L_v=10, D=512, 8 heads, FFN 2048, dropout 0.15 |
| Eq.3 injury-aware attention pooling | Eq.3 | `workup/volumetric.py` | `InjuryAttentionPool` | hierarchical pooling every floor(L_v/3) |
| Eq.4 PointNet++ set abstraction | Eq.4 | `workup/pointcloud.py` | `PointNetTrauma`, `SetAbstraction` | radii [0.12,0.25,0.5,1.0], N [512,256,128,64], MLP ladder; max-pool D_d=256 |
| Eq.5 ResNet-34 + spatial attention | Eq.5 | `workup/radiographic.py` | `CephalometricEncoder`, `SpatialAttention` | reduction 16, trauma bias, D_c=256 |
| Eq.6 cross-modal transformer fusion | Eq.6 | `workup/fusion.py` | `CrossModalFusion` | w_v=1.5, w_d=0.8, w_c=1.2, D_enc=1024 |
| Triadic encoder assembly | Methods | `workup/encoder.py` | `TriadicEncoder` | composes the three pathways + fusion |

## VLM-LLM co-reasoning (Algorithm 2; Section: Co-Reasoning Module)

| paper item | ref | file | module symbol | notes |
|---|---|---|---|---|
| Alg.2 co-reasoning pipeline | Algorithm 2 | `consult/coreasoner.py` | `CoReasoner.run` | 3 stages + parameter extraction |
| Stage 1 CLIP projection 1024->768 | Alg.2 L2; Methods | `consult/projection.py` | `ClipProjection` | LayerNorm + dropout 0.15 |
| Stage 1 vision-to-language decoder | Methods | `consult/narrator.py` | `VisionToLanguageDecoder` | 6-layer decoder, hidden 512, 8 heads, vocab 4800 |
| Injury severity + FISS scoring | Alg.2 L4; Methods | `consult/severity.py` | `severity_score`, `airway_compromise` | severity + airway class |
| Eq.7 dense passage retrieval | Eq.7 | `consult/knowledge.py` | `TraumaKnowledgeBase.retrieve` | PubMedBERT cosine, top-M=7, KB 15,400 (numpy exact; D1) |
| Stage 3 priority-ranked reasoning | Alg.2 L12-14 | `consult/deliberation.py` | `ReasoningEngine`, `GrammarReasoner`, `ApiReasoner` | protocol + deterministic grammar engine + guarded GPT-4-Turbo (D2) |
| Confidence threshold 0.75 review flag | Alg.2 L17; Methods | `consult/severity.py` | `confidence_review` | flags C_conf<0.75 |

## Priority-aware holistic optimization (Section: Priority-Aware Holistic Planning)

| paper item | ref | file | module symbol | notes |
|---|---|---|---|---|
| Eq.8 total emergency objective | Eq.8 | `prioritization/objectives.py` | `total_objective` | sum of weighted terms + L_joint |
| Eq.9 airway loss (cubic shortfall) | Eq.9 | `prioritization/objectives.py` | `airway_loss` | beta=[5.0,1.8], A_crit=50 mm^2 |
| Eq.10 stabilization loss | Eq.10 | `prioritization/objectives.py` | `stabilization_loss` | gamma=[3.0,2.0] |
| Occlusion / symmetry / aesthetic / joint | Methods (coeffs) | `prioritization/objectives.py` | `occlusion_loss`, `symmetry_loss`, `aesthetic_loss`, `joint_loss` | occlusion alpha=[2.0,1.5,1.0,1.0] |
| Adam optimization loop N_max=120 | Methods | `prioritization/planner.py` | `PriorityPlanner` | lr=2e-3, betas=(0.9,0.999) |

## Emergency-adapted robot motion planning (Algorithm 1; Section: Robotic Motion Planning Decoder)

| paper item | ref | file | module symbol | notes |
|---|---|---|---|---|
| Alg.1 Emergency Neural RRT* | Algorithm 1 | `intervention/rrtstar.py` | `EmergencyNeuralRRTStar.plan` | omega=0.35, N_max=3500, conn 0.5, goal 0.05 |
| Neural sampling policy | Methods | `intervention/policy.py` | `EmergencyPolicy` | 4x256 ReLU+BN, dropout 0.25, Gaussian head; inputs 7D+64^3+32^3+7D+1D |
| Time-weighted cost / steer | Methods | `intervention/kinematics.py` | `cost_time`, `steer`, `JointLimits` | alpha_time=2.0; velocity/accel tables |
| Swept-volume collision + 5mm margin | Methods | `intervention/collision.py` | `collision_free` | safety margin 5 mm |

## Evaluation metrics and statistics (Sections: Evaluation Metrics; Statistical Analysis)

| paper item | ref | file | module symbol | notes |
|---|---|---|---|---|
| Positional / rotational / condyle MAE | Eval Metrics 1-2,5 | `recovery/accuracy.py` | `positional_mae`, `rotational_mae`, `condyle_mae` | mm / degrees |
| Airway volume & min-CSA error | Eval Metrics 3 | `recovery/accuracy.py` | `airway_error` | cm^3 / mm^2 |
| Occlusal Class I rate | Eval Metrics 4 | `recovery/accuracy.py` | `occlusal_class_i` | percent |
| Time-to-intervention / efficiency | Eval Metrics 6-7 | `recovery/timing.py` | `planning_time`, `time_to_airway` | minutes/hours |
| Airway rescue / ICU LOS / outcomes | Eval Metrics 8-9 | `recovery/outcomes.py` | `airway_rescue_rate`, `icu_los` | emergency endpoints |
| BCa bootstrap CI (1000) | Statistical Analysis | `recovery/statistics.py` | `bca_bootstrap_ci` | bias-corrected accelerated |
| Permutation test (10,000) + Holm | Statistical Analysis | `recovery/statistics.py` | `permutation_test`, `holm_bonferroni` | two-sided |
| Eval loop assembling endpoints | Results | `recovery/appraisal.py` | `Appraisal.evaluate` | report dict |

## Training, folds, reproducibility (Section: Experimental Setup)

| paper item | ref | file | module symbol | notes |
|---|---|---|---|---|
| Trainer (150 ep, early stop 20) | Setup | `rounds/loop.py` | `Trainer.fit` | early-stopping patience 20 |
| Patient-level 5-fold CV + LOCO | Setup | `rounds/folds.py` | `patient_folds`, `leave_one_center_out` | stratified by patient |
| Atomic checkpoint + stored seed | R4 | `rounds/checkpoint.py` | `save_checkpoint`, `load_checkpoint` | tmp + os.replace |
| set_seed utility, seeds {17,23,42} | Setup/R4 | `rounds/seeding.py` | `set_seed` | restored on resume |
| Device / logging | R4 | `rounds/devices.py`, `rounds/logbook.py` | `select_device`, `get_logger` | logging module |

## Data and preprocessing (Section: Dataset)

| paper item | ref | file | module symbol | notes |
|---|---|---|---|---|
| Synthetic trauma cohort + dataset | Dataset | `arrival/cohort.py` | `TraumaCohort`, `CohortSpec` | recoverable injury signal, FISS strata (synthetic; D3) |
| CBCT/cast/ceph preprocessing | Dataset preprocessing | `arrival/acquisition.py` | `prepare_cbct`, `prepare_cast`, `prepare_cephalogram` | denoise/clip/z-score; FPS 16,384; CLAHE |
| Trauma-specific augmentation | Dataset augmentation | `arrival/augment.py` | `TraumaAugment` | per-modality, p=0.5 |

## Ablations and tables coverage

| table / ablation | ref | covered by | notes |
|---|---|---|---|
| Table 1-2 overall accuracy | Results | `recovery/accuracy.py` + `configs/experiment/main.yaml` | reported endpoints |
| Table 3 trauma severity strata | Results | `arrival/cohort.py` (FISS strata) + `recovery/appraisal.py` | mild/moderate/severe/critical |
| Table 4-7 timing/robotics/ICU/confidence | Results | `recovery/{timing,outcomes}.py` | endpoint reports |
| Table 9 component ablation | Ablations | `configs/experiment/ablation_*.yaml` | encoder / co-reasoning / optimization / robot toggles |
| Table 10 fusion-strategy ablation | Ablations | `configs/experiment/ablation_fusion_*.yaml` + `workup/fusion.py` | concat / early / late / cross-modal |
| Modality ablation (CBCT/dental/ceph) | Ablations | `configs/experiment/ablation_modality_*.yaml` + `protocol.py` toggles | drop-modality |
| Table 11 subgroup analysis | Results | `recovery/appraisal.py` strata | skeletal class / age / complexity |

## Deviations index (full text in `deviations.md`)

- D1: FAISS dense index -> numpy exact cosine retrieval (small synthetic KB).
- D2: GPT-4-Turbo API -> deterministic grammar `ReasoningEngine` with a guarded API adapter.
- D3: restricted clinical data -> deterministic synthetic trauma-cohort generator.
