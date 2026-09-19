Goal: improve the sign embedding model's validation accuracy on ASL Citizen as far as possible,
through model, augmentation, loss, optimization and domain-specific changes.

Read CLAUDE.md, README.md and ROADMAP.md first (step 7 "Strong model", the decisions and the
training findings), so you don't repeat experiments that are already done or rejected.

Metric and baseline
- Primary metric: val top-1 at k = 1 (the full val evaluation a run writes, 5 reference draws).
  Guard metric: val EER at k = 1 must not get worse beyond noise. Report k = 5 as well.
- Baseline: gru_wlasl_twins (val top-1 78.9% / EER 0.0540 at k = 1), trained with
  uv run scripts/train.py --name <name> --prepared data/prepared/asl_citizen-d314433a
  data/prepared/mm_wlauslan-d314433a data/prepared/wlasl-d314433a
  --set phonology_input=pooled --set phonology_weight=1
- A change counts as a gain only if its paired 95% bootstrap CI over val signs excludes 0 against
  the current best. Gains under about 1 point also need a second seed to count. Val has 290
  signs, so many experiments on it will overfit val: prefer changes with a clear reason over
  lucky configurations.

Rules
- Never touch the test split during the search. Use the splits from splits.py; no held-out signer
  or sign may reach training. Twin handling (twins.csv, SignBank twins) stays as it is.
- The GPU is shared with other agents: check nvidia-smi before each run, and run one training at
  a time.
- Every new option defaults to the current behavior, so earlier runs stay reproducible, and new
  code gets a small test (uv run pytest).
- Budget: at most 40 training runs. Stop early after 8 runs in a row without a counted gain.

Directions (roughly in order of expected value; justify each run in one line before starting it)
- Domain-specific inputs: hand shape relative to its own wrist and scale, velocities, bone
  angles, 3D coordinates (n_coords=3), lips/face for mouthing, a two-stream model (local hand
  shape plus global trajectory).
- Augmentation: hide hands below a random depth (the open framing decision in ROADMAP),
  per-hand and per-finger noise, temporal crops, stronger or weaker versions of the current
  augmentations, mixup between clips of one sign.
- Loss: ArcFace margin and scale, sub-center ArcFace, supervised contrastive alone or combined,
  an extra margin between near-minimal pairs (ROADMAP's next step for phonology).
- Architecture and optimization: GRU width and depth, attention pooling, the conv-transformer
  with heavy augmentation, a temporal convolutional network (TCN); learning rate and schedule,
  EMA or SWA of the weights, longer training once the model regularizes better.
- Inference: test-time augmentation (averaging embeddings over mirrored or augmented copies), and
  ensembles of the best runs.

Logging and wrap-up
- After each run, add its outcome to the ROADMAP findings (the run name, the change, and val
  top-1 and EER at k = 1 and 5 with paired CIs against the current best). Commit code changes as
  atomic commits.
- At the end, retrain the best configuration with 2 more seeds. Then evaluate it once on the test
  split (scripts/evaluate_test.py), on Slovo, and on the Swedish recordings
  (scripts/evaluate_recordings.py with the lexicon glossary sts_lexikon-234f4575). Summarize
  the gains against the baseline in ROADMAP.
