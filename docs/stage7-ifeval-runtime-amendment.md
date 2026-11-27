# Stage 7 IFEval runtime amendment

Date: 2026-08-15

This amendment was frozen before any Stage 7 IFEval score or raw model output
was inspected. The only observed IFEval information was process health,
wall-clock duration, and atomic checkpoint counts.

The initial Stage 7 IFEval launch inherited the generic Stage 2 generation cap
of 4,096 tokens. Its first Qwen3-8B arm required about 76 minutes, implying that
the full 8B/14B/32B ladder would take many hours to days. That is not an
operationally viable piggyback experiment.

The frozen Stage 7 test split was inspected only for explicit constraint
magnitudes. It contains five requests requiring at least 800 words and no
request requiring more than 900 words. All five are English-language requests.

The canonical Stage 7 IFEval execution therefore uses:

- `max_new_tokens = 2048` for every model and every IFEval arm;
- the unchanged 541-item test split, eight prompt arms, and four open-model
  steering/control arms;
- the unchanged fixed 30-cell Holm family and coherence wall;
- new `stage7_ifeval_*_test_v3` paths;
- `stage7_campaign_v3` and `stage7_scale_invariant_aggregate_v3` terminal
  orchestration paths.

All partial `stage7_ifeval_*_test_v2` artifacts are noncanonical and excluded
from validation, aggregation, and the terminal manifest. They remain on EC2 for
audit. Stage 7 v2 calibration, difficulty locks, held-out admissions, panel
locks, and safety treatment artifacts remain canonical and are reused without
modification.
