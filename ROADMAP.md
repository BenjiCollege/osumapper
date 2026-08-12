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

FullSet-v1 deliberately uses independent V4 passes. It provides the product and
validation boundary needed to measure V5 honestly; it does not claim learned
cross-tier nesting, section planning, or Placement-v2 quality.

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
inference, cross-tier nesting, auxiliary musical targets, and controlled V4/V5
human A/B review remain open and are not claimed by the prototype.

## Phase 4 — Placement-v2

- Predict relative jump distance/direction and long-range pattern continuation.
- Predict circles, sliders, spinners, slider geometry/duration/repeats, and combos.
- Model tier-appropriate spacing, overlaps, streams, bursts, jumps, anchors, and
  slider patterns.
- Keep off-screen geometry, duplicate ticks, impossible gaps, and other hard rules
  in deterministic post-processing.

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
