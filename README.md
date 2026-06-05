# HoloTrauma 3X — Clinical Study Protocol for the Code Release

This README is written as a study protocol. Each numbered section mirrors the structure of a
clinical investigation: the code is the instrument, the synthetic cohort is the study
population, and the reported endpoints are the measurements the instrument is built to make.

## Protocol synopsis

| Field | Value |
|---|---|
| Protocol ID | HOLOTRAUMA-3X / holotrauma3x |
| Title | Triadic VLM-LLM co-reasoning with priority-aware planning and robot trajectory decoding for emergency maxillofacial trauma |
| Framework | PyTorch 2.x, plain `torch.nn` |
| Investigational modalities | emergency CBCT volume, intraoral/cast point cloud, lateral cephalogram |
| Primary endpoints | maxilla / mandible / condyle MAE; airway volume error; occlusal Class I rate |
| Secondary endpoints | airway rescue rate; trajectory success rate; planning time |
| Design | single-pipeline, 5-fold patient-level cross-validation + leave-one-center-out |
| Population | deterministic synthetic trauma cohort (clinical data is controlled-access) |
| Status | reference implementation; runs end-to-end on CPU |

## 1. Background and rationale

Emergency maxillofacial trauma forces a surgical team to settle airway patency, skeletal
integrity, and occlusal function under a strict time budget. HoloTrauma 3X treats the
occlusion-bone-airway triad as one problem: a triadic anatomical encoder reads the three
imaging modalities, a vision-language / language co-reasoning stage grounds the findings in a
trauma knowledge base, a priority-aware optimizer resolves competing objectives by clinical
urgency, and an emergency-adapted neural RRT* decoder returns a collision-free robot
trajectory. This repository is the team's own implementation of that pipeline.

## 2. Objectives and endpoints

Primary objective: given the triad of inputs, predict skeletal repositioning and airway state
accurate enough to drive a robot-executable plan.

| Endpoint | Module | Reported target (manuscript, real cohort) |
|---|---|---|
| Maxilla MAE | `recovery/accuracy.py:positional_mae` | 0.42 mm |
| Mandible MAE | `recovery/accuracy.py:positional_mae` | 0.38 mm |
| Condyle MAE | `recovery/accuracy.py:condyle_mae` | 0.51 mm |
| Airway volume error | `recovery/accuracy.py:airway_error` | 1.47 cm^3 |
| Occlusal Class I | `recovery/accuracy.py:occlusal_class_i` | 94.7 % |
| Airway adequacy / rescue | `recovery/outcomes.py:airway_rescue_rate` | 96.2 % |
| Trajectory success | `intervention/rrtstar.py` | 98.6 % |

The reported targets are the manuscript figures on the clinical cohort. The synthetic cohort
shipped here exercises every code path but does not reproduce the clinical numbers.

## 3. Study design and setup

Install one of three ways.

pip:

    python -m pip install -r requirements.txt
    python -m pip install --no-deps -e .

conda:

    conda env create -f environment.yml
    conda activate holotrauma3x
    pip install --no-deps -e .

Docker:

    docker build -t holotrauma3x .
    docker run --rm holotrauma3x --help

## 4. Materials (study population)

| Group | Source | License | Access |
|---|---|---|---|
| G1 | Public craniomaxillofacial imaging, 12 institutions | CC BY 4.0 / CC BY-NC-SA 4.0 | open per source |
| G2-G4 | Three participating hospitals, clinical records | institutional DUA | controlled |

The clinical cohort (8,427 patients, ~1.2 TB) is controlled-access. The release generates a
deterministic synthetic trauma cohort at runtime, so no download is required:

    bash scripts/prepare_data.sh

## 5. Procedures

Each experiment is one config under `configs/experiment/`. The console entry point is
`holotrauma3x`.

    holotrauma3x train    --config configs/experiment/main.yaml
    holotrauma3x evaluate --config configs/experiment/main.yaml
    holotrauma3x plan     --config configs/experiment/main.yaml
    holotrauma3x export   --config configs/experiment/main.yaml

Ablations follow the same form, e.g.:

    holotrauma3x evaluate --config configs/experiment/ablation_optimization.yaml
    holotrauma3x evaluate --config configs/experiment/ablation_fusion_concat.yaml
    holotrauma3x evaluate --config configs/experiment/ablation_modality_cbct.yaml

A fast end-to-end pass for verification:

    holotrauma3x train --config configs/experiment/_smoke.yaml

## 6. Outcome measures

| Procedure | Command | What it reports |
|---|---|---|
| Primary endpoints | `holotrauma3x evaluate --config configs/experiment/main.yaml` | maxilla/mandible/condyle MAE, airway volume error, occlusal Class I, airway rescue |
| Planning demonstration | `holotrauma3x plan --config configs/experiment/main.yaml` | priority-ranked reasoning trace, confidence, trajectory waypoint count |
| Component ablations | `holotrauma3x evaluate --config configs/experiment/ablation_*.yaml` | endpoint shift when a module is removed |

## 7. Statistical analysis plan

Aggregate endpoints are reported as mean over 5 folds x 3 seeds {17, 23, 42}. Interval
estimates use bias-corrected and accelerated bootstrap
(`recovery/statistics.py:bca_bootstrap_ci`, 1000 resamples). Between-method comparisons use
two-sided permutation tests (`permutation_test`, 10,000 permutations) with Holm-Bonferroni
correction (`holm_bonferroni`).

## 8. Compute and safety

| Resource | Value |
|---|---|
| Accelerator | NVIDIA A100 80GB |
| Cross-validation | 5-fold patient-level + leave-one-center-out |
| Schedule | 150 epochs, early stopping patience 20 |
| Cost | ~168 GPU-hours per fold |
| Seeds | {17, 23, 42} |

Checkpoints are written atomically (temp file + `os.replace`) and carry their seed for exact
resume. Trained weights for the clinical cohort are available from the corresponding authors
on reasonable request.

## 9. Monitoring and quality control

    make lint     # ruff + isort + black
    make type     # mypy --strict
    make test     # pytest

The suite spans shape, physics-invariant, overfit-single-batch, gradient-flow, metric-vs-scipy
regression, retrieval ranking, RRT* collision/timeout, determinism, and house-style guards, in
addition to the end-to-end training smoke test.

## 10. Ethics

The underlying study was a retrospective analysis of fully de-identified records; institutional
review boards at all three participating hospitals waived approval. Public datasets were used
under their open-access licenses. Language-model calls in the clinical system used only
de-identified, structured descriptors under PIPL / GDPR / HIPAA-compliant infrastructure.

## 11. Data availability

The de-identified, multi-institutional dataset is available for controlled access under
institutional data-use agreements; requests are directed to the corresponding authors. The
synthetic cohort used by this repository is generated deterministically and needs no external
download.
