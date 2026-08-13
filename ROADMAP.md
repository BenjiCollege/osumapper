# osumapper full-set quality roadmap

## Central product goal

Given one user-selected song, produce one playable osu!standard `.osz` containing
Easy, Normal, Hard, Insane, Expert, and Expert+. The set must share timing and
audio, feel intentionally related across tiers, and place the strongest quality
emphasis on Expert and Expert+.

Generated output is for local use and human editing. Current osu! ranking criteria
do not permit generative tooling for ranking-bound objects, timing, or hitsounds.

## Release gates

- Mean no-mod star error below 0.15★ and every difficulty within 0.25★.
- Zero duplicate object ticks, partially off-screen objects, or invalid object gaps.
- Identical uninherited timing across all six difficulties.
- Exactly six standard difficulties and one shared audio file per package.
- Lower rhythm-density error than the frozen V4 benchmark.
- Better Expert/Expert+ precision at equivalent recall than V4.
- Human review on at least 30 held-out songs per tier and across varied genres,
  BPM ranges, time signatures, lengths, and song structures.
- No promotion when any objective quality gate regresses.

## Phase 1 — FullSet-v1 foundation (implemented)

- One `--full-set` CLI and Studio workflow.
- One isolated workspace and cached audio features for all six generations.
- Fixed targets: 1.50★, 2.35★, 3.35★, 4.65★, 5.90★, and 7.00★.
- Bounded density correction plus lower-tier spacing reduction and separate
  Expert/Expert+ spatial-strain correction, with strict per-map and set-level
  star gates.
- Safe playfield geometry, shared timing verification, exact archive counts, and
  deterministic packaging.
- Preservation of referenced background, video, and storyboard assets for
  explicit-map input, deterministic preview fallback, and usable combo colours.
- Per-tier criteria reports plus a set-level machine-readable quality report.

FullSet-v1 deliberately uses independent per-tier passes. It provides the product
and validation boundary needed to measure a new rhythm or placement model
honestly; it does not by itself claim one-pass shared inference, section planning,
or measured placement quality. The per-package quality report names the rhythm
architecture and placement engine that actually produced it, so its stated
limitations follow the models used rather than a fixed sentence.

## Phase 2 — Curated training data (prototype established)

- Mark only trusted human-authored maps as GOOD.
- Start with 50–100 distinct songs per lower tier and 200–300 trusted Expert and
  Expert+ difficulties.
- Exclude converted, broken, unreviewed, and automatically generated maps.
- Balance genres, BPM, meter, song structure, length, and respected mapping styles.
- Freeze song-grouped train/validation/test splits at seed 2026.
- Never train on uncorrected model output.

The first frozen prototype contains 30 user-approved songs and 225 GOOD standard
maps at seed 2026. Its benchmark manifest intentionally reports
`curated-prototype`: it validates the workflow but does not meet the production
quantity or untouched-test coverage gates above.

## Phase 3 — Conformer-v5 Full-Set (initial six-head model implemented)

- One shared audio encoder evaluated once per song.
- Six difficulty-specific rhythm heads with larger Expert/Expert+ capacity.
- Cross-tier event-importance and nesting objective.
- Auxiliary beat, downbeat, section, intensity, density, and emphasis targets.
- Per-tier calibration and hard-negative mining for higher precision.
- Extra Expert/Expert+ loss weight without starving lower tiers.
- Mixed precision, precomputed shards, XLA where stable, AdamW, scheduling, and
  early stopping for RTX 4070 throughput.

V5 is accepted only when it beats frozen V4 on the same split and review rubric.
Frozen MERT features are an optional controlled ablation after the core V5 result.

The initial implementation covers the shared difficulty-independent encoder, six
tier-specific rhythm heads, larger Expert heads, tier-weighted loss, deterministic
window shards, and expanded ±20/±35 ms/per-tier evaluation. One-pass full-set
inference, auxiliary musical targets, and controlled V4/V5 human A/B review remain
open and are not claimed by the prototype.

Conformer-v6 closes the cross-tier nesting item architecturally: a multi-scale
audio stem feeds six heads whose logits accumulate non-negative increments, so a
timestamp's probability cannot fall as the requested tier and target stars move
from Easy toward Expert+, and a hard-negative focal objective concentrates
learning on confident false positives and missed events. Each tier is still
generated in its own inference pass, so the full-set quality report records
`one_pass_full_set_inference: false`. V6 advances only by beating frozen V5 on the
same split and review rubric.

## Phase 4 — Placement (v4 current, measured against v1/v2/v3)

- Predict relative jump distance/direction and long-range pattern continuation.
- Predict circles, sliders, spinners, slider geometry/duration/repeats, and combos.
- Model tier-appropriate spacing, overlaps, streams, bursts, jumps, anchors, and
  slider patterns.
- Keep off-screen geometry, duplicate ticks, impossible gaps, and other hard rules
  in deterministic post-processing.

The implemented model conditions every prediction on the requested tier and target
star rating, adds measure-level musical context, predicts absolute playfield
position alongside the relative step, and predicts slider repeats and hold length.
Reconstruction anchors direction to the predicted absolute position while
preserving the calibrated step distance, rotates to the nearest legal direction
instead of shortening jumps at the playfield edge, and emits repeats only when the
traversal fits before the next object. A five-block Conformer encoder replaces the
three plain attention blocks, and the objective is rebalanced with Huber terms,
class-weighted object types, and a positive-weighted combo loss.

### Measured result, 66 held-out test songs, seed 2026

Every placement architecture was trained on the same frozen split with the same
schedule and seed, so the differences are attributable to architecture and
objective only. Placement-v4 supersedes v3; its distance comparison is reported
under Phase 4b because average error is the wrong acceptance metric for it.

| metric | v1 | v2 | v3 |
|---|---:|---:|---:|
| jump distance MAE | 50.85 px | 53.84 px | **48.09 px** |
| spacing correlation | 0.677 | 0.652 | **0.717** |
| prediction spread (target 93.4 px) | 72.6 px | 67.9 px | 71.2 px |
| spinner recall | 0.2664 | **0.7828** | 0.7541 |
| new combo accuracy | 0.7878 | 0.8350 | **0.8439** |

Placement-v1 could never train before this work: a loss term reduced the step
axis, so every run aborted in its first epoch. V2 then beat the repaired V1 on
object types, combos, turn angle, and slider length, but regressed jump distance
and shrank its prediction spread, because a Huber delta of 0.05 is 32px while
typical distance errors were ~54px. V3 restores squared error on distance only,
demotes the weak absolute-position term, and now leads on jump distance, turn
angle, slider length, object type, and combo accuracy.

Long-range pattern continuation, learned overlap and anchor structure, and
non-linear slider geometry are still open. A promoted placement model must beat
its predecessors and PatternPlanner-v1 on the frozen test split **and** in human
review; the version number alone is not evidence, and no human A/B review has
been run yet.

### Resolved since the first full-set measurement

- **Slider and spinner tails were unsnapped** (508 occurrences across six
  difficulties, up to 71% of holds on a tier). Object starts were always snapped;
  only ends drifted, because their duration came from a continuous predicted
  pixel length. Reconstruction now snaps the duration to an editor division and
  derives the length from it. Now zero.
- **Placement was conditioned on the wrong density.** Training uses each map's
  real objects/second; generation passed the tier's nominal target, which
  measured about 0.6x the median human density of that tier. Tier targets are now
  the measured medians, and inference derives density from the timestamps being
  placed.
- **Generated sets bought stars with spacing.** Expert+ reached its target with
  1,142 objects at 95% jumps and 238px median spacing, against a human map using
  1,590 objects at 50px. With correct densities it now uses 1,560 objects at 61px.
- **No generated map contained a stack**, on any tier, in any version through v3.
  Placement-v4's mixture density head reproduces the human stacking rate.
- **Short sliders were impossible at Expert+ density.** Reconstruction refused any
  slider under 35px, but at 8 objects/second the median gap allows 0.42 beats
  while a 35px slider at that song's velocity needs 0.50, so 31 of 200 predicted
  sliders became circles. Across 28,180 sliders in dense human Expert+ maps, p1 is
  23.8px and only 3.0% fall under 35px, so the floor is now 20px. Note the
  measurement that prompted this also retired the claim that the type head is
  biased: conditioned on density, human Expert+ slider use falls from 32.4% at
  4.5-6/s to 15.6% at 7.5-9/s, and the model predicts 12.8% at 8.1/s. The
  apparent 13-point gap came from comparing against one atypical mapset rather
  than the density-conditioned population.
- **Stream notes were pruned preferentially.** They carry lower predicted
  probability than isolated notes (0.800 against 0.832), so one flat per-tier
  cutoff removed them first: stream recall trailed non-stream recall by 0.25 on
  Insane. Selection now applies hysteresis, admitting a lower-confidence candidate
  only where it continues an already-selected run. The continuation ratio was
  chosen by sweep as the strongest setting costing no F1.

### Known quality gaps in generated full sets

The first V6 + Placement-v3 six-difficulty package hit every star target (mean
error 0.0145★, maximum 0.0285★) while still failing to look human-authored:

- Mid tiers are slider-dominated (Insane came out 85% sliders), so tier
  conditioning reaches the distance head far more effectively than the type head.
- Expert+ made 96% of its transitions jumps over 160px with zero stacks, because
  star calibration buys difficulty with spacing inflation rather than musical
  structure.
- Sub-tick snapping errors concentrate in sparse tiers (83/208 Easy objects,
  150/327 Normal), which alone would fail visual review.

These are ranked ahead of further architecture work: snapping first, then
constraining spacing-only calibration, then per-tier object-type priors, then
curating GOOD-rated training data instead of the current unrated split.

## Phase 5 — Complete arrangement and polish

- Shared musical-section and six-tier difficulty plan.
- Hitsound, combo, break, kiai, preview-time, and metadata planning.
- Iterative rhythm/placement adjustment to meet star targets without crude density
  inflation.
- Per-tier and set-level playability reports surfaced in Studio.
- Refuse structurally broken packages and import accepted sets into osu!lazer.

## Benchmark discipline

Preserve every promoted model, dataset manifest, split hash, seed, configuration,
calibration, evaluation, and review rubric. Retraining the same data is not assumed
to improve quality. A model advances only through measured held-out and human-review
gains against the frozen predecessor.
