# Repository Plan

## Organizing principle

The package mirrors the **clinical pathway** of an emergency maxillofacial trauma patient.
Each top-level subpackage is a stage of care; cross-cutting concerns (the study `protocol`,
operator `orders`, and training `rounds`) sit alongside it.

    arrival  ->  workup  ->  consult  ->  prioritization  ->  intervention  ->  recovery
    (imaging)   (encode)    (reason)     (plan)               (robot)           (outcomes)

## Directory tree

    holotrauma3x/
    |- README.md
    |- LICENSE
    |- pyproject.toml
    |- requirements.txt
    |- environment.yml
    |- Dockerfile
    |- Makefile
    |- .gitignore
    |- .pre-commit-config.yaml
    |- .github/workflows/ci.yml
    |- configs/
    |  |- protocol.yaml                 # base study protocol (defaults)
    |  |- experiment/
    |     |- main.yaml
    |     |- ablation_encoder.yaml
    |     |- ablation_coreasoning.yaml
    |     |- ablation_optimization.yaml
    |     |- ablation_robot.yaml
    |     |- ablation_fusion_concat.yaml
    |     |- ablation_fusion_early.yaml
    |     |- ablation_fusion_late.yaml
    |     |- ablation_modality_cbct.yaml
    |     |- ablation_modality_cbct_dental.yaml
    |     |- supplementary_loco.yaml
    |     |- supplementary_threshold_sweep.yaml
    |     |- _smoke.yaml                 # pytest only; do not use for reporting
    |- src/holotrauma3x/
    |  |- __init__.py
    |  |- protocol.py                    # typed config dataclasses + plan/input containers
    |  |- pathway.py                     # HoloTrauma3x model assembling the four modules
    |  |- orders.py                      # jsonargparse CLI (train/evaluate/plan/export)
    |  |- arrival/{__init__,acquisition,cohort,augment}.py
    |  |- workup/{__init__,volumetric,pointcloud,radiographic,fusion,encoder}.py
    |  |- consult/{__init__,projection,narrator,knowledge,severity,deliberation,coreasoner}.py
    |  |- prioritization/{__init__,weights,objectives,planner}.py
    |  |- intervention/{__init__,policy,kinematics,collision,rrtstar}.py
    |  |- recovery/{__init__,accuracy,timing,outcomes,statistics,appraisal}.py
    |  |- rounds/{__init__,seeding,devices,logbook,folds,checkpoint,loop}.py
    |- scripts/
    |  |- launch_train.sh
    |  |- launch_eval.sh
    |  |- prepare_data.sh
    |- tests/
    |  |- conftest.py
    |  |- test_shapes.py
    |  |- test_invariants.py
    |  |- test_objectives.py
    |  |- test_retrieval.py
    |  |- test_reasoning.py
    |  |- test_planner_overfit.py
    |  |- test_rrtstar.py
    |  |- test_statistics.py
    |  |- test_gradients.py
    |  |- test_determinism.py
    |  |- test_house_rules.py
    |  |- test_training_smoke.py
    |- docs/{project-context,implementation-map,repo-plan,deviations}.md
    |- assets/

## Module responsibilities

- `protocol.py` — frozen dataclasses for every config block (`ArrivalSpec`, `EncoderSpec`,
  `CoReasoningSpec`, `OptimizationSpec`, `RobotSpec`, `TrainingSpec`, `Protocol`), plus the
  data containers `TriadInputs` and `SurgicalPlan`. These are what jsonargparse binds.
- `pathway.py` — `HoloTrauma3x` wires `TriadicEncoder -> CoReasoner -> PriorityPlanner ->
  EmergencyNeuralRRTStar`; `build_pathway(protocol)` factory.
- `orders.py` — single jsonargparse CLI exposing `train`, `evaluate`, `plan`, `export`
  subcommands, each typed against `Protocol`; console script `holotrauma3x`.
- `arrival/*` — synthetic cohort, deterministic preprocessing, trauma augmentation.
- `workup/*` — the four encoder equations and their assembly.
- `consult/*` — projection, decoder, retrieval, severity/confidence, reasoning engine,
  Algorithm 2 orchestration.
- `prioritization/*` — priority weights, the six loss terms, the Adam planning loop.
- `intervention/*` — neural policy, kinematics/limits, collision, Algorithm 1.
- `recovery/*` — metrics, timing/outcome endpoints, statistics, the evaluation loop.
- `rounds/*` — seeding, device, logging, fold splitters, atomic checkpoints, the trainer.

## Dependencies (pinned ranges)

Runtime: `torch>=2.1,<3`, `numpy>=1.26`, `scipy>=1.11`, `jsonargparse[signatures]>=4.40`,
`pyyaml>=6`. Optional `transformers>=4.40` behind a guarded import for real CLIP/PubMedBERT;
optional `openai>=1.30` behind the guarded reasoning adapter. Dev: `pytest`, `ruff`,
`black`, `isort`, `mypy`, `pre-commit`.

## Test matrix (varied, not smoke-only)

| file | kind | asserts |
|---|---|---|
| test_shapes.py | shape | encoder/fusion/decoder output dims (512/256/256/1024/768) |
| test_invariants.py | physics/invariant | airway cubic penalty monotone, attention rows sum to 1, A_min>=0, weight order w_L>>w_A>w_F>w_LT |
| test_objectives.py | metric correctness | Eq.9/Eq.10 against closed-form references |
| test_retrieval.py | correctness | top-M=7, cosine ranking vs numpy reference |
| test_reasoning.py | determinism/structure | grammar reasoner emits ranked plan + confidence; review flag <0.75 |
| test_planner_overfit.py | overfit-single-batch | planner drives airway shortfall to ~0 on one case |
| test_rrtstar.py | correctness | trajectory collision-free; timeout escalates |
| test_statistics.py | numerical regression | BCa CI / permutation p vs scipy reference |
| test_gradients.py | gradient-flow | losses backprop to encoder params |
| test_determinism.py | determinism | same seed -> identical cohort + forward |
| test_house_rules.py | style guard | AST+tokenize: no comments, no docstrings, no forbidden phrases/emoji |
| test_training_smoke.py | e2e smoke | 2 steps on `_smoke.yaml`, loss decreases |

## Verification gates

`ruff check .`, `black --check .`, `isort --check .`, `mypy --strict src/holotrauma3x`,
`pytest -q`, and a `_smoke` end-to-end CLI run must all pass. `docker build` is documented
but may be unverifiable without a local daemon.
