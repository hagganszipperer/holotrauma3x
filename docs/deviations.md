# Deviations

Intentional departures from the manuscript, each with a paper anchor and justification. The
science (architectures, equations, losses, schedules, priority weights) is implemented as
written; the items below concern restricted external dependencies and clinical data that
cannot ship with a public release.

## D1 — Dense passage retrieval index

- Paper anchor: Co-Reasoning Module; Eq.7 ("dense passage indexing with FAISS").
- Manuscript: FAISS index over 15,400 PubMedBERT passage embeddings.
- Release: exact cosine top-M retrieval in `consult/knowledge.py` over an in-memory passage
  matrix. The ranking is identical to an exact FAISS flat index; only the ANN data structure
  differs. Top-M=7 and the cosine relevance of Eq.7 are preserved.

## D2 — LLM reasoning backend

- Paper anchor: Co-Reasoning Module; Algorithm 2 ("GPT-4-Turbo, temperature 0.3").
- Manuscript: structured reasoning produced by the GPT-4-Turbo API.
- Release: a deterministic, grammar-driven `GrammarReasoner` emits the same priority-ranked
  trace (life-saving / stabilization / functional / risk) with confidence scoring, so the
  pipeline runs offline and reproducibly. A guarded `ApiReasoner` adapter wires the real API
  when credentials are present. The confidence-review threshold (0.75) and top-M=7 grounding
  are unchanged.

## D3 — Clinical cohort

- Paper anchor: Dataset; Data Availability (controlled-access, institutional DUAs).
- Manuscript: 8,427 patients of multi-institutional CBCT, intraoral, and cephalometric data
  (~1.2 TB), available only under data-use agreements.
- Release: a deterministic synthetic trauma-cohort generator (`arrival/cohort.py`) with a
  recoverable injury signal and FISS strata. Volumetric, point-cloud, and radiograph grids
  are reduced from the clinical acquisition shapes so the full pipeline runs on CPU; the
  encoder hyperparameters (patch 8, L_v=10, D_v=512, PointNet++ ladder, ResNet-34) match the
  paper. Real backbones load through guarded adapters when available.

## D4 — Cross-modal fusion modality weights

- Paper anchor: Eq.6 and surrounding prose ("w_v=1.5 for CBCT, w_d=0.8 for dental,
  w_c=1.2 for cephalometric").
- The printed Eq.6 concatenation subscripts the modality weights inconsistently with the
  prose that assigns each weight to its own modality. `workup/fusion.py` follows the prose:
  each modality feature is scaled by its own weight before fusion. Values are unchanged.

## D5 — Policy network volumetric inputs

- Paper anchor: Robotic Motion Planning Decoder ("inputs: configuration 7D, obstacle distance
  field 64^3, soft-tissue uncertainty map 32^3, goal 7D, urgency 1D").
- The 64^3 and 32^3 fields are reduced by adaptive average pooling to compact descriptors
  before the policy MLP in `intervention/policy.py`; feeding the raw 262k/32k-dimensional
  grids as a flat input is impractical. All other inputs and the Gaussian output head match.
