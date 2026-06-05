# HoloTrauma 3x — Project Context

    project_name       : holotrauma3x                                  [HIGH]
    domain             : emergency maxillofacial trauma planning —      [HIGH]
                         multimodal imaging + VLM/LLM co-reasoning +
                         robot trajectory planning
    framework          : PyTorch 2.x, plain torch.nn                    [HIGH]
    venue              : npj Digital Medicine                           [HIGH]
    primary_datasets   : 4 source groups (see §6)                       [HIGH]
    compute_target     : NVIDIA A100 80GB, ~168 GPU-h/fold             [HIGH]
    hparams_reference  : Results §Setup + Methods §Encoder/Reasoning/   [HIGH]
                         Optimization/Robotics
    supp_path          : none (Supplementary Notes 1-3 and Appendix B
                         referenced in text; no standalone SI file)
    extra_signals      : seeds {17,23,42}; GPT-4-Turbo external API;
                         2 algorithm boxes (Neural RRT*, co-reasoning);
                         code availability = "upon reasonable request";
                         retrospective IRB-waived ethics

    NEEDS_USER_DECISION: 0
    Status             : Turn 0 resolved; values approved.

## 1. project_name

`holotrauma3x` — derived from the framework name "Holo-Trauma 3x" used throughout the
title, abstract, and methods. Snake-friendly identifier, three content words collapsed
to a single token. Source: Title; Introduction ("we present Holo-Trauma 3x"). [HIGH]

## 2. supp_path

`none`. The manuscript references Supplementary Note 1 (preprocessing and provenance),
Supplementary Note 2 (airway-threshold ROC validation), Supplementary Note 3 (loss-term
coefficients), and Appendix B (full architecture/hyperparameters), but no separate
supplementary file accompanies the source. All values needed were recoverable from the
main text. [HIGH]

## 3. domain

Emergency maxillofacial trauma surgical planning. The system jointly assesses the
occlusion-bone-airway triad from emergency CBCT, intraoral/cast point clouds, and lateral
cephalometric radiographs, performs vision-language co-reasoning grounded in a trauma
knowledge base, runs priority-aware multi-objective planning, and decodes a collision-free
robot trajectory. Source: Abstract; Methods §Problem Definition. [HIGH]

## 4. framework

PyTorch 2.x with plain `torch.nn`. The manuscript states the framework was built in
PyTorch 2.1 on NVIDIA A100 80GB GPUs. The architecture (3D volumetric transformer,
PointNet++, ResNet-34, cross-modal transformer, CLIP projection, transformer decoder,
neural RRT* policy) is implemented directly in `torch.nn` so every paper detail is
expressible without a higher-level wrapper. Source: Results §Experimental Setup. [HIGH]

## 5. venue

npj Digital Medicine. Confirmed. The structured manuscript (Data Availability, Code
Availability, Author Contributions, Competing Interests, numbered references, line
numbering) and the clinical-AI subject matter match the journal's format. [HIGH]

## 6. primary_datasets

The combined cohort is 8,427 patients spanning January 2018-December 2024 across three
continents, assembled from four access groups:

| group | name | version | license | access |
|---|---|---|---|---|
| G1 | Public craniomaxillofacial imaging (12 institutions) | aggregate 2018-2024 | CC BY 4.0 / CC BY-NC-SA 4.0 | open-access per source |
| G2 | Affiliated Hospital of Nantong University — clinical | retrospective | institutional DUA | controlled |
| G3 | Shanghai Ninth People's Hospital — clinical | retrospective | institutional DUA | controlled |
| G4 | Stomatological Hospital of Tongji University — clinical | retrospective | institutional DUA | controlled |

Sub-cohorts: 1,243 orthognathic-arthroplasty emergency cases; 847 acute trauma encounters;
347 robot-assisted cases with full intraoperative trajectories. Approx. 1.2 TB imaging,
3.4M annotated landmarks, 127k occlusal contacts, 47.3k fracture segments, 18.6k airway
measures. The de-identified multi-institutional dataset is available for controlled access
under institutional data-use agreements. Source: Methods §Dataset; Data Availability. [HIGH]

This release ships a deterministic synthetic trauma-cohort generator (recoverable injury
signal, FISS strata, airway compromise levels) so the full pipeline runs without restricted
clinical data; real backbones load through guarded adapters when available.

## 7. compute_target

NVIDIA A100 80GB. Patient-level 5-fold stratified cross-validation plus
leave-one-center-out; three global seeds {17,23,42}; 150 epochs per fold with early
stopping (patience 20); approximately 168 GPU-hours per fold. Triadic encoder inference
8.3+/-1.2 s; co-reasoning 12.7+/-2.4 s; optimization 2.3+/-0.6 min; trajectory planning
3.2+/-0.8 s; end-to-end planning median 3.8 min. Source: Results §Experimental Setup;
§Time-Critical Planning. [HIGH]

## 8. hparams_reference

Consolidated from Results §Experimental Setup and Methods §Encoder / §Co-Reasoning /
§Optimization / §Robotics, with architecture figures.

Triadic encoder: 3D volumetric transformer patch 8^3, L_v=10, D_v=512, 8 heads, FFN 2048,
dropout 0.15; PointNet++ 4 SA layers, radii [0.12,0.25,0.5,1.0], samples [512,256,128,64],
MLP [[32,32,64],[64,64,128],[128,128,256],[256,256,512]], D_d=256; ResNet-34 spatial
attention reduction 16, D_c=256; cross-modal fusion modality weights w_v=1.5, w_d=0.8,
w_c=1.2, D_enc=1024.

Co-reasoning: CLIP ViT-L/14 projection 1024->768 (LayerNorm + dropout 0.15); 6-layer
transformer decoder hidden 512, 8 heads, trauma vocabulary 4,800; PubMedBERT (110M,
max length 512), dense passage retrieval top-M=7, knowledge base 15,400 documents;
GPT-4-Turbo (gpt-4-turbo-2024-04-09) temperature 0.3; confidence review threshold 0.75.

Optimization: priority weights w_L=10.0, w_A=5.0, w_F=2.0, w_LT=1.0; airway critical
threshold 50 mm^2, airway coefficients beta=[5.0,1.8] with cubic shortfall penalty;
stabilization gamma=[3.0,2.0]; occlusion alpha=[2.0,1.5,1.0,1.0]; Adam lr=2e-3,
betas=(0.9,0.999), max 120 iterations.

Robotics (Neural RRT*): exploration 0.35, safety margin 5 mm, time cost weight 2.0,
max 3500 iterations, connection radius 0.5 rad, step 0.1 rad, goal threshold 0.05 rad;
policy 4 layers x 256 units, ReLU, batch-norm, dropout 0.25; inputs config 7D + obstacle
field 64^3 + soft-tissue uncertainty 32^3 + goal 7D + urgency 1D; Gaussian output;
joint velocity limits [1.73,1.73,1.73,2.30,2.30,2.88,2.88] rad/s; acceleration limits
[2.20,2.20,2.20,3.30,3.30,4.40,4.40] rad/s^2; e-stop deceleration 5 rad/s^2.

Training: CLIP fine-tune 50 epochs lr 5e-5 batch 128; ResNet-34 lr 1e-4. Statistics: BCa
bootstrap (1000), permutation tests (10,000) with Holm-Bonferroni, mixed-effects models,
1:1 propensity-score matching (caliper 0.2 SD). [HIGH]

## 9. extra_signals

- Released checkpoints: none stated; weights "upon reasonable request".
- Algorithm boxes: 2 (Algorithm 1 Emergency Neural RRT*; Algorithm 2 VLM-LLM co-reasoning).
- External dependency: GPT-4-Turbo API for the reasoning stage (guarded adapter here).
- Code availability (verbatim): "The source code for the HoloTrauma 3X framework,
  including model architectures, training scripts, prompts, and evaluation pipelines,
  will be made available upon reasonable request to the corresponding authors. All
  experiments were conducted using PyTorch 2.1 with random seeds {17,23,42} for
  reproducibility."
- Ethics: retrospective, IRB-waived across the three participating hospitals; de-identified
  data only; PIPL/GDPR/HIPAA-compliant API calls with de-identified inputs.
- Reported headline numbers (for evaluation tolerances): maxilla MAE 0.42 mm, mandible MAE
  0.38 mm, airway volume MAE 1.47 cm^3, condyle MAE 0.51 mm, occlusal Class I 94.7%,
  airway adequacy 96.2%, trajectory success 98.6%, operative time 3.2 h, complications 5.7%.
