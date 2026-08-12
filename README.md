# osumapper — modernized for osu!lazer

> [!IMPORTANT]
> This repository is a community-maintained modernization of the original
> [osumapper project](https://github.com/kotritrona/osumapper), created by
> [kotritrona](https://github.com/kotritrona). The original model architecture,
> generation concepts, trained model assets, presets, and legacy implementation
> belong to that upstream work. The 2026 changes focus on compatibility,
> packaging, safety, testing, and usability—not claiming authorship of the
> original project.

osumapper generates osu! beatmaps using the original v7 rhythm models through a
modern Python application. It accepts exported `.osz` packages, individual `.osu`
difficulties, or audio files and creates complete `.osz` packages that can be
imported into osu!lazer.

## Product goal

Give osumapper one favorite song and receive one playable osu!standard `.osz`
containing Easy, Normal, Hard, Insane, Expert, and Expert+, with the strongest
quality emphasis on Expert and Expert+. The current **FullSet-v1** implementation
establishes the six-map packaging and validation boundary using Conformer-v4.
Conformer-v5 Full-Set, Placement-v2, section-aware mapping, and learned
cross-difficulty nesting remain research milestones—not completed features. See
[`ROADMAP.md`](ROADMAP.md) for the phased quality plan and release gates.

The original 2020 implementations are preserved unchanged under [`v6.2/`](v6.2/)
and [`v7.0/`](v7.0/). See [`legacy/README.md`](legacy/README.md) for the Git safety
baseline and compatibility policy.

## What the modernization adds

- A Python 3.12 package under `src/osumapper` with a straightforward CLI.
- A locked, reproducible environment managed by [uv](https://docs.astral.sh/uv/).
- Safe `.osz` extraction into isolated temporary directories.
- Complete, deterministic `.osz` export and optional opening in osu!lazer.
- A standard-only Conformer-v4 workflow with fixed star-rated difficulty tiers;
  historical taiko, catch, and mania generation remains available through the
  preserved legacy compatibility path.
- A professional local queue UI with multi-file/folder drag-and-drop, training
  controls, progress, retry/clear actions, and per-package Lazer import.
- Cross-platform timing estimation using librosa instead of `TimingAnlyz.exe`.
- Pure-Python beatmap conversion in the modern path instead of Node.js.
- Typed configuration, `pathlib`, checked subprocesses, progress output, and
  actionable errors.
- Fourteen migrated `.keras` rhythm models with deterministic prediction-parity
  manifests.
- An opt-in local osu!standard dataset, safe curated `.osz` ingestion,
  Transformer-v1, Conformer-v2, faster Conformer-v3, and star-conditioned
  Conformer-v4 training,
  validation-only density calibration, held-out evaluation, and deterministic
  human-review packages.
- Mixed-precision RTX training, optional XLA, float16 window shards, per-song
  balancing, AdamW, learning-rate reduction, and early stopping.
- A Placement-v1 learner for flow, object types, slider lengths, and combo
  changes, plus a standalone placement/playability analyzer.
- An osu!standard ranking-criteria audit that reports objective rules,
  measurable guidelines, and mandatory manual checks without claiming rankability.
- Representative fixtures, golden-output tests, safe-archive tests, and
  cross-ruleset smoke coverage.

## Requirements

- Windows, macOS, or Linux.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Enough free space for the Python environment and TensorFlow dependencies.
- osu!lazer is optional unless `--open` is used.

You do not need to install Python manually. The checked-in `.python-version`
selects the tested Python 3.12 runtime, and uv can manage it for the project.

## Install

Clone or download this repository, open a terminal in its root directory, and run:

```powershell
uv sync --locked
uv run osumapper doctor
```

`doctor` reports the active Python version, required libraries, available legacy
models, and the detected osu!lazer executable. A healthy installation reports
Python 3.12 and all dependencies as available.

## Quick start

After installation, generation is one command:

```powershell
uv run osumapper generate song.osz --preset default --open
```

The generated package is placed in `output/` unless `--output` is provided.
`--open` asks the installed osu!lazer executable to import the finished package.
It does not modify Lazer's private `client.realm` database or hashed file storage.

## Supported inputs

| Input | Behavior |
| --- | --- |
| `.osz` | Safely extracts the package and generates from a selected difficulty. |
| `.osu` | Uses the selected difficulty and its adjacent audio, or audio supplied with `--audio`. |
| Audio | Estimates timing and creates a source difficulty before generation. Supported formats include MP3, OGG, WAV, FLAC, M4A, AAC, and Opus. |

When an `.osz` contains several difficulties, use `--difficulty` with a unique
substring of the desired difficulty name.

## Fixed osu!standard difficulty tiers

Conformer-v4 trains and generates osu!standard only. Its model-level difficulty
input contains exactly two values: the requested no-mod star rating and the fixed
difficulty tier. OD, AR, CS, density, taiko, catch, and mania are not V4 model
conditioning inputs. Map difficulty settings are applied from the selected profile,
while [`rosu-pp-py`](https://pypi.org/project/rosu-pp-py/) calculates the real
no-mod standard star value used for dataset labels and generated-output validation.

| Tier | Accepted star range | Default target |
|---|---:|---:|
| Easy | 0.0★–1.99★ | 1.50★ |
| Normal | 2.0★–2.69★ | 2.35★ |
| Hard | 2.7★–3.99★ | 3.35★ |
| Insane | 4.0★–5.29★ | 4.65★ |
| Expert | 5.3★–6.49★ | 5.90★ |
| Expert+ | 6.5★ and above | 7.00★ |

Choose the tier and an exact target inside that tier:

```powershell
uv run osumapper generate song.osz --rhythm-engine modern --modern-model models/modern/rhythm-conformer-v4-standard-stars --difficulty-tier hard --target-stars 3.50 --open
```

The generated `.osu` is recalculated after rhythm and placement. The package is
created only when its real star value is inside the selected band and within
±0.25★ of the requested target; otherwise the command reports the measured value
and asks for a density adjustment or more V4 training data. This prevents a
filename from claiming a difficulty the map did not actually achieve.

## Generate a complete six-difficulty set

FullSet-v1 analyzes/caches the source audio in one isolated workspace, runs the
star-conditioned V4 model for all six fixed targets, and writes exactly six `.osu`
files with one shared audio file. Each tier receives up to four bounded correction
attempts. Easy and Normal reduce density and spacing together; Expert and Expert+
adjust spatial strain separately from density so difficulty is not controlled only
by flooding or starving a map of objects. Referenced background, video, and
storyboard assets are preserved for explicit `.osu` input; audio-only input receives
a neutral packaged background. Export is refused unless every difficulty is inside
its band and within 0.25★ of target, mean star error is below 0.15★, timing sections
are identical, and deterministic gameplay-safety checks pass. FullSet-v1 aims for
0.18★ per tier during refinement and uses the wider 0.25★ boundary only after all
six attempts are exhausted.

```powershell
uv run osumapper generate song.osz --rhythm-engine modern --modern-model models/modern/rhythm-conformer-v4-standard-stars --flow-engine deterministic --full-set --output output/song-full-set.osz --open
```

The command writes `song-full-set.full-set.json` plus one
`song-full-set.<tier>.criteria.json` report per difficulty. Each report includes
circle, slider, and spinner counts plus heuristic jump, burst, stream, stack, and
position-diversity measurements. FullSet-v1 runs V4 independently for each tier;
learned event nesting and one-pass shared inference require Conformer-v5. The
output is a local draft and is not eligible for ranking.

### PatternPlanner-v1

Modern generation with `--flow-engine deterministic` or `auto` uses the seeded,
difficulty-aware PatternPlanner-v1 after Conformer-v4 selects rhythm timestamps.
It creates a controlled mixture of:

- circles and new-combo boundaries;
- linear sliders whose duration leaves a legal gap before the next object;
- spinners only when a musical rest is long enough for that difficulty;
- wide jumps, curved compact streams and bursts, and sparse readable stacks; and
- playfield-clamped positions and slider endpoints.

The same seed, input, model, and controls reproduce the same plan. Easy through
Hard favor readability, near-stacks, and more sliders. Expert and Expert+ permit
larger movement, tighter streams, and sparse exact stacks. The planner never adds
rhythm timestamps: timing precision remains the responsibility of Conformer-v4,
and star calibration plus the criteria audit still gate export. This is a bounded
heuristic placement system, not the future learned Placement-v2 model; every map
still requires listening, test play, and human editing.

## Command examples

Generate from an exported osu!lazer package:

```powershell
uv run osumapper generate song.osz --preset sota --seed 2026 --open
```

Select one difficulty from a package:

```powershell
uv run osumapper generate song.osz --difficulty "Insane" --preset default
```

Generate from an explicit `.osu` file and audio file:

```powershell
uv run osumapper generate map.osu --audio audio.mp3 --preset default
```

Generate standard mode directly from audio, overriding automatic timing:

```powershell
uv run osumapper generate song.mp3 --mode standard --bpm 174 --offset 250
```

Generate a 7K mania difficulty:

```powershell
uv run osumapper generate song.mp3 --mode mania --keys 7 --preset mania-highkey
```

Generate taiko or catch:

```powershell
uv run osumapper generate song.osz --mode taiko --preset taiko
uv run osumapper generate song.osz --mode catch --preset catch
```

Use the fully seeded flow generator instead of per-map legacy GAN training:

```powershell
uv run osumapper generate song.osz --flow-engine deterministic --seed 2026
```

Choose an exact destination:

```powershell
uv run osumapper generate song.osz --output "output/My Generated Map.osz"
```

## Local drag-and-drop interface

Start the local interface with:

```powershell
uv run osumapper ui
```

The **Generate queue** tab accepts individual `.osz`, `.osu`, and audio files, any
combination of multiple files, or whole folders. Folder scanning queues one
representative difficulty per beatmap folder by default; enable **Every
difficulty** to queue all `.osu` files. Duplicate paths are ignored and existing
output names receive a safe numeric suffix instead of being overwritten.

When Studio runs under WSL, Windows paths such as `C:\Users\...\Downloads\song.mp3`
are translated automatically to `/mnt/c/...`. The file picker starts in the
Windows Downloads directory. If WSLg does not deliver an Explorer drag event,
choose **Paste path** and paste the Windows path directly.

Use **Clear queue** when you are ready for another song. The queue shows each
item's state and output, can be stopped, and can retry failed or stopped items.
Generation controls expose the preset, ruleset, seed, flow/rhythm engines, modern
model, validation-calibrated threshold override, density, difficulty, BPM,
offset, and mania keys. **Import each completed package into osu!lazer** opens
every successful `.osz` as it finishes.

Enable **Generate complete Easy–Expert+ set** to create one validated six-map
package for every queued song. Full-set mode manages tier density and star targets
itself, so the individual Density, Output difficulty, and Target stars values are
not passed to generation.

The **Training lab** tab provides safe `.osz` ingestion, explicit GOOD-map
curation, scan/statistics/split/feature/window controls, Conformer-v3 and
Placement-v1 GPU training, calibration, held-out evaluation, review packages,
and placement analysis. Both sides scroll independently on smaller screens. The
**Activity** tab keeps detailed process output and actionable errors.

## Presets and rulesets

List every installed preset with:

```powershell
uv run osumapper presets
```

| Ruleset | Presets |
| --- | --- |
| Standard | `default`, `sota`, `vtuber`, `flower`, `inst`, `lowbpm`, `tvsize`, `hard`, `normal` |
| Taiko | `taiko` |
| Catch | `catch` |
| Mania | `mania-lowkey`, `mania-highkey` |

The source map's ruleset is used automatically unless `--mode` is supplied. For
audio-only input, the default is standard mode. Mania key count is controlled with
`--keys`.

## Generation controls

The most useful `generate` options are:

| Option | Purpose |
| --- | --- |
| `--preset NAME` | Select the model and generation parameters. |
| `--mode MODE` | Select a ruleset for legacy compatibility; Conformer-v4 accepts only `standard`. |
| `--difficulty TEXT` | Select a difficulty from a multi-map `.osz`. |
| `--difficulty-tier NAME` | Select `easy`, `normal`, `hard`, `insane`, `expert`, or `expert-plus` for V4 output. |
| `--target-stars NUMBER` | Fix the requested no-mod standard star target inside the selected tier. |
| `--full-set` | Generate all six fixed standard tiers in one validated `.osz`. |
| `--seed INTEGER` | Set the deterministic seed; the default is `2026`. |
| `--rhythm-engine legacy` | Use the preserved v7 rhythm path; this remains the default. |
| `--rhythm-engine modern` | Use a locally trained modern osu!standard rhythm model. |
| `--modern-model PATH` | Select a modern model directory or `.keras` file. |
| `--placement-model PATH` | Select a trained Placement-v1 directory or `.keras` file. |
| `--rhythm-threshold NUMBER` | Override the modern model's hit-probability threshold. |
| `--target-density NUMBER` | Cap modern output at a target number of objects per second. |
| `--flow-engine auto` | With modern rhythm, use PatternPlanner-v1; legacy rhythm retains compatible legacy flow. |
| `--flow-engine legacy` | Require the legacy per-map flow model. |
| `--flow-engine deterministic` | Use seeded PatternPlanner-v1 for modern standard maps. |
| `--flow-engine placement` | Use learned Placement-v1 flow with the modern rhythm engine. |
| `--bpm NUMBER` | Override timing estimation for audio-only input. |
| `--offset MS` | Set the first timing-point offset for audio-only input. |
| `--keys NUMBER` | Set mania key count for audio-only input. |
| `--output PATH` | Choose the output `.osz` path. |
| `--open` | Launch the completed package in osu!lazer. |
| `--quiet` | Suppress normal progress messages. |

Run `uv run osumapper generate --help` for the authoritative command reference.

## osu!lazer workflow

1. Export a beatmap from osu!lazer as an `.osz` package.
2. Pass the package to `osumapper generate`.
3. Review the `.osz` produced under `output/`.
4. Import it with `--open`, double-click it, or drag it into osu!lazer.
5. Open the generated difficulty in the editor and review timing, object placement,
   hitsounds, metadata, and playability before publishing.

Generated maps are starting points for private/local review. The current general
osu! ranking criteria require ranking-bound hit objects, hitsounds, and timing to
be created exclusively through direct human input, so an osumapper-generated map
is **not eligible for the ranking process**. Do not submit generated output as a
rankable beatmap. See the official [general ranking criteria](https://osu.ppy.sh/wiki/en/Ranking_criteria)
and [osu!standard criteria](https://osu.ppy.sh/wiki/en/Ranking_criteria/osu%21).

## Ranking-criteria audit

Every generated osu!standard package now receives a sibling report: generating
`song.osz` also writes `song.criteria.json`. The report checks the deterministic subset of the
criteria, including object timing gaps, duplicate timestamps, common beat-snap
errors, partially off-screen geometry, timing-point collisions, kiai on the first
uninherited point, minimum drain time, audio extension, preview point, background,
combo-colour count, difficulty-setting guidelines, and difficulty-specific spinner
and overlap limits.

Audit an existing standard difficulty directly:

```powershell
uv run osumapper criteria check "C:\Maps\review\difficulty.osu" --output "output\difficulty-criteria.json"
```

Add `--strict` for CI or scripting that should return a non-zero exit status when
objective structural errors are found. A clean automated result is not a
rankability certificate. Musical timing, permissions, authorship, hitsound
audibility, visual safety/readability, slider clarity, full-mapset spread, and
playability still require human review and tools such as Mapset Verifier. The
implemented policy snapshot is dated 2026-08-11; always consult the current wiki
before relying on it.

## Optional osu!stable adapter

The old Songs-folder workflow is retained as an optional adapter. It does not parse
Lazer storage and does not require the Windows registry or `osu!.db`:

```powershell
uv run osumapper stable-scan "C:\osu!" --mode standard --output maplist.txt
```

## Training a modern rhythm model locally

The modern rhythm engine is an **opt-in, osu!standard-only experiment**. No
pretrained modern model is shipped, and the legacy rhythm engine remains the
default. Training output is stored under ignored local directories so it cannot
replace the original models accidentally. A low training loss is not evidence of
beatmap quality; use the held-out test-song evaluation and human review before
drawing conclusions.

The default workspace is `training_data/`. You may place it elsewhere by passing
the same `--data-root PATH` to every dataset, training, evaluation, and analysis
command. Audio is read from your Songs directory and cached as numerical features;
the source audio and beatmaps are not copied or modified.

### NVIDIA GPU training on Windows with WSL2

Current TensorFlow releases do not provide CUDA training through the native
Windows package. Windows users with an NVIDIA GPU should run training inside
WSL2. Generation can continue to run from the normal Windows installation.

Install Ubuntu from an elevated PowerShell window, restart Windows when prompted,
and verify that the distribution is using WSL2:

```powershell
wsl --install -d Ubuntu
wsl.exe --update
wsl.exe --list --verbose
wsl.exe -d Ubuntu
```

Commands beginning with `wsl.exe` are Windows commands. After the final command,
the prompt is inside Ubuntu and Linux commands should be used. Install the project
in a separate Linux workspace; do not reuse the Windows `.venv` through `/mnt/c`,
because Windows and Linux virtual environments are incompatible:

```bash
sudo apt update
sudo apt install -y git curl build-essential ffmpeg python3-tk
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
git clone https://github.com/BenjiCollege/osumapper.git "$HOME/osumapper"
cd "$HOME/osumapper"
uv sync --locked --extra gpu
```

The `gpu` extra keeps the CUDA-enabled TensorFlow dependencies associated with
the WSL project environment. Confirm both WSL GPU passthrough and TensorFlow GPU
detection before starting a long run:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 uv run --extra gpu python -c "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); assert g, 'NVIDIA GPU unavailable'; print('TensorFlow:', tf.__version__); print('GPU:', g)"
```

Use `--device gpu` for every GPU training command. Unlike `--device auto`, this
fails immediately if TensorFlow cannot see a GPU instead of silently training on
the CPU. `CUDA_VISIBLE_DEVICES=0` selects the first NVIDIA GPU; on a one-GPU
machine that is the installed card. Audio decoding and feature extraction still
use the CPU, while TensorFlow model training uses the GPU.

Do not copy `dataset.parquet` from Windows and use it unchanged in WSL: it contains
absolute audio and map paths. Re-scan the Songs directory using its WSL mount path,
then recreate the deterministic split. In interactive Bash, single-quote a path
containing `osu!` (or run `set +H`) so `!` is not treated as history expansion:

```bash
cd "$HOME/osumapper"
set +H
uv run --extra gpu osumapper dataset scan '/mnt/e/Games/osu!/Songs' --data-root "$HOME/osumapper/training_data"
uv run --extra gpu osumapper dataset split --data-root "$HOME/osumapper/training_data" --seed 2026 --include-unrated
uv run --extra gpu osumapper dataset features --data-root "$HOME/osumapper/training_data"
```

Only use `--include-unrated` for an explicitly experimental dataset. A successful
feature pass must report `"failed": 0`. Decoder warnings about damaged ID3 or MPEG
metadata can be non-fatal when extraction still finishes with zero failures.

### 1. Scan and curate a local dataset

Point the scanner at a mixed song source. One scan now reads both loose legacy
`.osu` maps and every `.osz` package below the selected folder. Packages are safely
extracted into the ignored, content-addressed `training_data/imported_osz/` cache;
the original `.osz` files and legacy song folders are never modified. Duplicate
`.osu` content is indexed once. Malformed maps, unsafe or invalid packages, missing
audio, converted maps, and maps outside the configured quality bounds are recorded
in `training_data/skipped.jsonl` instead of stopping the rest of the scan.

```powershell
uv run osumapper dataset scan "E:\Games\osu!\Songs"
uv run osumapper dataset stats
```

Use `--no-osz` only when you intentionally want to ignore archives and scan loose
legacy `.osu` files alone.

The scanner writes scalar metadata to `training_data/dataset.parquet` and detailed
per-map timing/object data under `training_data/maps/`. It supports UTF-8 and
common legacy encodings, inherited timing points, timing changes, sliders,
spinners, hitsounds, and the standard mapping statistics used by preprocessing.

Review maps and label them explicitly. Unrated maps are not assumed to be good:

```powershell
uv run osumapper dataset rate "E:\Games\osu!\Songs\123 Artist - Song\map.osu" good
uv run osumapper dataset rate "E:\Games\osu!\Songs\456 Artist - Song\map.osu" bad
uv run osumapper dataset rate "E:\Games\osu!\Songs\789 Artist - Song\map.osu" ignore
```

Ratings persist in `training_data/ratings.json` and update the current Parquet
index. By default, only maps marked `good` are eligible for training. Use
`dataset split --include-unrated` only when you deliberately want unrated maps.

When `.osz` packages are already below the folder passed to `dataset scan`, no
second import command is needed. The explicit `import-osz` command remains useful
for appending a separate curated download folder without rescanning the main song
source:

```powershell
uv run osumapper dataset import-osz "C:\Users\bcten\Downloads\osz"
```

If, and only if, every package in that folder has been reviewed and is a trusted
mapping reference, mark those imported maps GOOD in the same operation:

```powershell
uv run osumapper dataset import-osz "C:\Users\bcten\Downloads\osz" --rating good
```

The WSL equivalent for this repository is:

```bash
uv run --extra gpu osumapper dataset import-osz \
  '/mnt/c/Users/bcten/Downloads/osz' \
  --data-root "$HOME/osumapper/training_data" \
  --rating good
```

`--rating good` is an explicit trust decision, not an automatic quality claim.
After importing or changing ratings, always recreate the split and features.

### 2. Split songs and cache features

```powershell
uv run osumapper dataset split --seed 2026
uv run osumapper dataset features
uv run osumapper dataset windows --sequence-length 512 --audio-context-radius 0
```

The split is deterministic and groups every difficulty from the same song/mapset
together, preventing song leakage across the 80% training, 10% validation, and
10% test partitions. The manifest includes the dataset hash and refuses stale
splits after the dataset changes.

Feature extraction uses 22.05 kHz audio by default and caches normalized
log-mel, onset-strength, RMS-energy, spectral-flux, and beat-pulse arrays with
their exact frame timestamps. Cache keys include audio content and feature
configuration. Candidate hit positions are built from the map timing grid at
1/1, 1/2, 1/3, 1/4, 1/6, and 1/8 subdivisions; each human object is matched to at
most one candidate within the configured tolerance.

`dataset windows` writes deterministic float16 shards under
`training_data/window_shards/`. Training creates or reuses the same shards
automatically. This moves JSON/NPZ parsing out of repeated epochs so the GPU is
fed a shard at a time. Use `--rebuild` only after deliberately changing the
preprocessing implementation; configuration and map hashes otherwise select the
correct cache automatically.

### 3. Train standard-star Conformer-v4

V4 changes the dataset schema because each standard map receives a real star
rating and one of the six fixed tiers. Re-scan and rebuild the split, features, and
V4 window shards. Existing ratings remain associated with map hashes, and existing
V1–V3 model folders are not modified.

```bash
cd "$HOME/osumapper"
set +H
uv sync --locked --extra gpu
uv run --extra gpu osumapper dataset scan \
  '/mnt/e/Games/osu!/Songs' \
  --data-root "$HOME/osumapper/training_data"
uv run --extra gpu osumapper dataset stats \
  --data-root "$HOME/osumapper/training_data"
uv run --extra gpu osumapper dataset split \
  --data-root "$HOME/osumapper/training_data" \
  --seed 2026 \
  --include-unrated
uv run --extra gpu osumapper dataset features \
  --data-root "$HOME/osumapper/training_data"
uv run --extra gpu osumapper dataset windows \
  --data-root "$HOME/osumapper/training_data" \
  --architecture conformer-v4 \
  --sequence-length 512 \
  --audio-context-radius 0
```

`dataset stats` reports the star distribution and map count in every tier. The
command above includes unrated maps to match the requested full collection; omit
`--include-unrated` for the recommended accuracy run after enough maps in every
tier have been reviewed and marked GOOD.

Train the isolated V4 model on the RTX 4070:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra gpu osumapper train rhythm \
  --data-root "$HOME/osumapper/training_data" \
  --output "$HOME/osumapper/models/modern/rhythm-conformer-v4-standard-stars-mixed-20260811" \
  --architecture conformer-v4 \
  --epochs 75 \
  --batch-size 32 \
  --learning-rate 0.0005 \
  --sequence-length 512 \
  --audio-context-radius 0 \
  --device gpu \
  --precision mixed-float16 \
  --xla auto \
  --window-cache auto \
  --early-stopping-patience 8 \
  --lr-patience 3 \
  --lr-factor 0.5 \
  --weight-decay 0.0001 \
  --seed 2026
```

Calibrate per-tier validation thresholds and evaluate once on held-out songs:

```bash
uv run --extra gpu osumapper train calibrate rhythm \
  --data-root "$HOME/osumapper/training_data" \
  --model "$HOME/osumapper/models/modern/rhythm-conformer-v4-standard-stars-mixed-20260811"
uv run --extra gpu osumapper train evaluate rhythm \
  --data-root "$HOME/osumapper/training_data" \
  --model "$HOME/osumapper/models/modern/rhythm-conformer-v4-standard-stars-mixed-20260811"
```

The Studio defaults to standard mode, modern rhythm, Conformer-v4, Hard, and
3.35★. Changing the tier updates its default target automatically. The CLI retains
the older rulesets only so legacy snapshots and compatibility tests remain usable.

#### Preserved V1–V3 baselines

```powershell
uv run osumapper train rhythm --architecture transformer-v1 --epochs 50 --batch-size 16 --seed 2026 --device auto
```

Transformer-v1 combines a small Conv1D audio encoder with timing-grid and
difficulty features, then applies three Transformer attention blocks to predict
`P(hit)` for each candidate position. Positive-class weighting handles sparse
labels. Training uses only training and validation partitions and writes the best
checkpoint, a last-epoch resume checkpoint, deployment model, per-epoch state,
configuration, history, dataset manifest, TensorBoard logs, and CSV logs to
`models/modern/rhythm/`.

The project now also provides Conformer-v2. It adds local depthwise convolution
to global self-attention and uses a nine-frame audio context around every musical
grid candidate. Its default output is deliberately separate, so it cannot
overwrite Transformer-v1:

```powershell
uv run osumapper train rhythm --architecture conformer-v2 --sequence-length 512 --audio-context-radius 4 --epochs 50 --batch-size 16 --seed 2026 --device auto --output models/modern/rhythm-conformer-v2
```

Both architectures use the exact existing song-level split manifest. Compare
them only when the dataset hash and train/validation/test assignments match.

The completed 50-epoch WSL model at
`$HOME/osumapper/models/modern/rhythm-conformer-v2-wsl-seed-2026-50` is the
Conformer-v2 baseline. Preserve it; do not resume or overwrite it. Conformer-v3
uses a new output directory. It removes the nine-frame input expansion used by
v2 and learns local temporal context inside the Conformer, masks padded
attention positions, uses a gated convolution module, AdamW, per-song balancing,
mixed precision, cached shards, adaptive learning rate, and early stopping.

After adding trusted rated maps, rebuild the split without `--include-unrated` so
only GOOD maps train the accuracy run:

```bash
cd "$HOME/osumapper"
CUDA_VISIBLE_DEVICES=0 uv run --extra gpu osumapper dataset split \
  --data-root "$HOME/osumapper/training_data" \
  --seed 2026
CUDA_VISIBLE_DEVICES=0 uv run --extra gpu osumapper dataset features \
  --data-root "$HOME/osumapper/training_data"
CUDA_VISIBLE_DEVICES=0 uv run --extra gpu osumapper dataset windows \
  --data-root "$HOME/osumapper/training_data" \
  --sequence-length 512 \
  --audio-context-radius 0
```

Start the isolated Conformer-v3 run on the RTX 4070:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra gpu osumapper train rhythm \
  --data-root "$HOME/osumapper/training_data" \
  --output "$HOME/osumapper/models/modern/rhythm-conformer-v3-rated-seed-2026" \
  --architecture conformer-v3 \
  --epochs 75 \
  --batch-size 32 \
  --learning-rate 0.0005 \
  --sequence-length 512 \
  --audio-context-radius 0 \
  --device gpu \
  --precision mixed-float16 \
  --xla auto \
  --window-cache auto \
  --early-stopping-patience 8 \
  --lr-patience 3 \
  --lr-factor 0.5 \
  --weight-decay 0.0001 \
  --seed 2026
```

The probability head stays float32 for stable loss and calibration even when
the encoder uses Tensor Cores. `--device gpu` fails instead of silently using
CPU. If batch 32 exhausts the RTX 4070's memory, start a new output folder with
batch 16. Do not change batch size, precision, split, architecture, or
preprocessing while resuming an existing run.

To continue an interrupted v3 run, repeat its full command unchanged and add
`--resume`. `--epochs 75` is the total target, not 75 additional epochs. Resume
validation rejects changed batch size, learning rate, seed, precision, XLA,
scheduler, balancing, weight decay, architecture, context, or split.

For baseline reproduction only, the completed WSL2 Conformer-v2 command was:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra gpu osumapper train rhythm \
  --data-root "$HOME/osumapper/training_data" \
  --output "$HOME/osumapper/models/modern/rhythm-conformer-v2-wsl-seed-2026-50" \
  --architecture conformer-v2 \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --sequence-length 512 \
  --audio-context-radius 4 \
  --device gpu \
  --seed 2026
```

Training writes `last.keras`, `best.keras`, `training_state.json`, history, CSV,
and TensorBoard data after each completed epoch. To continue after closing the
terminal or restarting Windows, enter Ubuntu again and inspect the saved state:

```powershell
wsl.exe -d Ubuntu
```

```bash
cd "$HOME/osumapper"
source "$HOME/.local/bin/env"
nvidia-smi
cat "$HOME/osumapper/models/modern/rhythm-conformer-v2-wsl-seed-2026-50/training_state.json"
```

Then use the identical command and add `--resume`:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra gpu osumapper train rhythm \
  --data-root "$HOME/osumapper/training_data" \
  --output "$HOME/osumapper/models/modern/rhythm-conformer-v2-wsl-seed-2026-50" \
  --architecture conformer-v2 \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --sequence-length 512 \
  --audio-context-radius 4 \
  --device gpu \
  --seed 2026 \
  --resume
```

`--epochs 50` means 50 total epochs, not 50 additional epochs. A run with six
completed epochs therefore resumes at epoch seven. If interruption occurs during
an epoch, the last completed checkpoint is retained and that incomplete epoch is
repeated. Do not scan, rate, or split the dataset while a run is in progress;
resume validation intentionally rejects a changed split. Feature-cache validation
may run again before the model loads, which is expected.

Monitor utilization from a second Ubuntu terminal while training is active:

```bash
watch -n 2 nvidia-smi
```

Do not resume a model merely to increase its recorded epoch count. Each epoch is
now regression-tested to contain non-zero training and validation work. Historical
runs affected by the former alternating empty-epoch bug should remain snapshots;
start a clean corrected run in a new output directory.

`--device auto` selects an available GPU and otherwise uses CPU. For long GPU
runs, use `--device gpu` so a broken CUDA environment cannot fall back unnoticed,
or use `--device cpu` to require CPU execution. TensorFlow training can need
substantial memory and time on a large collection.

### 4. Calibrate on validation, then evaluate once on unseen songs

Choose the operating threshold without looking at test songs:

```powershell
uv run osumapper train calibrate rhythm --model models/modern/rhythm
```

The command maximizes candidate-level F1 using validation songs only, breaks ties
toward the higher/safer threshold, also calibrates low-, medium-, and
high-density bands, writes `threshold_calibration.json`, and makes generation,
analysis, and evaluation use the matching threshold by default. An explicit
`--threshold` or `--rhythm-threshold` still overrides it.

For the new v3 run, calibrate and evaluate with the exact paths:

```bash
uv run --extra gpu osumapper train calibrate rhythm \
  --data-root "$HOME/osumapper/training_data" \
  --model "$HOME/osumapper/models/modern/rhythm-conformer-v3-rated-seed-2026"
uv run --extra gpu osumapper train evaluate rhythm \
  --data-root "$HOME/osumapper/training_data" \
  --model "$HOME/osumapper/models/modern/rhythm-conformer-v3-rated-seed-2026"
```

```powershell
uv run osumapper train evaluate rhythm --model models/modern/rhythm
```

Evaluation is hard-wired to the held-out test partition. It reports precision,
recall, F1, PR-AUC, matched-event timing error, and density error. Treat these as
comparative diagnostics, not proof that a generated map is enjoyable or rankable.
The command refuses a dataset/split manifest that differs from the one recorded
during training, so the reported test songs cannot silently overlap the run's
training or validation songs.

Create real packages from a deterministic sample of held-out songs for human
review in the osu! editor:

```powershell
uv run osumapper train review rhythm --count 5 --seed 2026 --output output/held-out-review
```

The review selector samples distinct test songs and chooses a median-density
difficulty from each selected set. It writes complete `.osz` packages plus a
`review_manifest.json`. Add `--open` only when you want each package imported into
osu!lazer immediately.

### Current Transformer-v1 baseline

The first completed local baseline is preserved under
`models/modern/baselines/transformer-v1-seed-2026-unrated-50-recorded-26-effective/`.
It recorded 50 history rows but only 26 non-empty epochs because of the corrected
finite-dataset exhaustion bug. It also used unrated maps. Therefore its results
demonstrate learning, not mapping quality, and the snapshot must not be resumed.

At threshold `0.5`, held-out F1 was `0.7038`, precision `0.5705`, and mean density
error `1.835 objects/sec`. Validation-only calibration selected threshold
`0.806874`; on the unchanged held-out test songs this improved F1 to `0.7368`,
precision to `0.6808`, mean timing error to `0.514 ms`, and mean density error to
`0.692 objects/sec`. PR-AUC remained `0.7898`, as expected because calibration
changes the decision threshold rather than probability ranking.

Five deterministic review packages from this baseline are available locally at
`output/held-out-review-transformer-v1/` with their exact source-map manifest.

### 5. Train Placement-v1 and analyze flow

Rhythm decides *when* objects occur. Placement-v1 is a separate model that
learns rotationally relative jump distance/turn patterns, circle/slider/spinner
type, slider length, and combo changes from the same song-level split. It is kept
separate so rhythm-v3 can be evaluated before placement changes the output.

Train it on the RTX 4070 after the rated-only split is stable:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra gpu osumapper train placement \
  --data-root "$HOME/osumapper/training_data" \
  --output "$HOME/osumapper/models/modern/placement-v1-rated-seed-2026" \
  --epochs 75 \
  --batch-size 32 \
  --learning-rate 0.0005 \
  --sequence-length 256 \
  --device gpu \
  --precision mixed-float16 \
  --xla auto \
  --early-stopping-patience 8 \
  --weight-decay 0.0001 \
  --seed 2026
```

Evaluate jump distance, turn angle, object type, slider length, and combo
prediction on held-out test songs:

```bash
uv run --extra gpu osumapper train evaluate placement \
  --data-root "$HOME/osumapper/training_data" \
  --model "$HOME/osumapper/models/modern/placement-v1-rated-seed-2026"
```

Placement inference constrains generated objects and slider endpoints to the
playfield and caps jump distance using the available time. It remains an
experimental learned starting point; review every map in the osu! editor.

Analyze any standard `.osu` map without training a placement model:

```powershell
uv run osumapper placement analyze "C:\Users\bcten\Downloads\reviewed-map.osu"
```

The JSON report includes offscreen objects/slider points, stacking, edge use,
position diversity, jump distance and speed percentiles, turn angles, a
transition timeline, and a clearly labeled heuristic score. These diagnostics
measure obvious flow/playability problems; they are not proof of map quality.

### Recommended research order

Preserve and evaluate the completed V4 benchmark first. Curate GOOD-only maps,
then train Conformer-v5 Full-Set with a shared encoder, six tier heads, Expert and
Expert+ capacity/weighting, auxiliary musical targets, hard negatives, and
per-tier calibration. Placement-v2 follows after rhythm wins on the frozen split.
Optional frozen MERT features are an ablation only after those controlled
baselines are stable. The complete sequence and acceptance targets are maintained
in [`ROADMAP.md`](ROADMAP.md).

Inspect one human map against the model and export a timestamp-by-timestamp JSON
and CSV timeline:

```powershell
uv run osumapper analyze "E:\Games\osu!\Songs\123 Artist - Song\map.osu" --format both
```

The report includes human hits, predicted hits, matched/missed/extra events, timing
error, beat position, onset strength, and prediction probability.

### 6. Generate with the trained models

```powershell
uv run osumapper generate song.osz --rhythm-engine modern --flow-engine deterministic --open
```

The modern engine predicts osu!standard timestamps, then passes them through the
existing placement, metadata, validation, packaging, and Lazer-import boundary.
Use `--modern-model PATH` for a non-default model, `--rhythm-threshold` to adjust
selection, and `--target-density` to cap objects per second. It exits with an
actionable error if the model selects no usable events. Omit `--rhythm-engine
modern` to retain the original behavior.

Use both learned models in WSL after training:

```bash
uv run --extra gpu osumapper generate song.osz \
  --rhythm-engine modern \
  --modern-model "$HOME/osumapper/models/modern/rhythm-conformer-v3-rated-seed-2026" \
  --flow-engine placement \
  --placement-model "$HOME/osumapper/models/modern/placement-v1-rated-seed-2026" \
  --target-density 2.0 \
  --seed 2026 \
  --open
```

## Model migration and verification

The repository contains the fourteen original extensionless Keras/HDF5 rhythm
models and parity-checked `.keras` replacements. The default legacy rhythm engine
uses these compatibility copies; they are separate from a locally trained modern
rhythm model, and migration is not part of routine setup.

Verify the tracked models and parity-manifest hashes:

```powershell
uv run osumapper models verify
```

Rebuild every `.keras` model from the preserved originals:

```powershell
uv run osumapper models migrate --seed 2026
uv run osumapper models verify
```

Migration loads each original through an isolated HDF5 compatibility path, runs a
fixed prediction probe, transfers the weights into a public Keras 3 architecture,
saves and reloads it, and checks prediction parity. A replacement is installed only
when its predictions remain within tolerance. Each manifest records SHA-256 hashes,
probe shapes, runtime versions, and maximum absolute error.

## Determinism and safety

- Python, NumPy, and TensorFlow receive the configured seed.
- TensorFlow deterministic operations are enabled when supported.
- Archive entries are sorted and receive fixed timestamps during packaging.
- `.osz` extraction rejects path traversal, drive paths, symbolic links, excessive
  entry counts, and excessive expanded sizes.
- Every run uses its own temporary directory; fixed temporary filenames and global
  working-directory changes are isolated behind the legacy compatibility boundary.
- The modern application never writes directly into osu!lazer storage.

Exact generated object placement can still vary if platform-level TensorFlow or
audio-decoder behavior differs. Keep the same lockfile, runtime, source package,
models, options, and seed when comparing results.

## Troubleshooting

### `uv` is not recognized

Install uv using its
[official installation instructions](https://docs.astral.sh/uv/getting-started/installation/),
restart the terminal, and run `uv sync --locked` again.

### osu!lazer is not found

Run `uv run osumapper doctor`. Generation still works without Lazer, but omit
`--open` and import the resulting `.osz` manually.

### The beatmap references missing audio

Place the referenced audio beside the `.osu` file or pass it explicitly:

```powershell
uv run osumapper generate map.osu --audio "path/to/audio.mp3"
```

### A package contains multiple difficulties

Select one with a unique name fragment:

```powershell
uv run osumapper generate mapset.osz --difficulty "Hard"
```

### Audio is too short

The rhythm model needs usable audio after the first timing point. Use a longer
source file or correct the BPM and offset supplied for audio-only generation.

### TensorFlow prints compatibility warnings

The preserved v7 flow pipeline can emit deprecation messages while loading its
historical graph. Use `--flow-engine deterministic` to avoid per-map legacy GAN
training. Model hash and prediction parity can be checked with
`uv run osumapper models verify`.

### Dataset splitting reports no eligible maps

The default split accepts only maps you explicitly rated `good`. Run
`dataset stats`, label reviewed maps with `dataset rate`, and split again. The
`--include-unrated` option is available for intentional exploratory training.

### Modern generation selects no events

First confirm that training completed and held-out evaluation produced sensible
results. Then inspect the map with `osumapper analyze`; if appropriate, retry with
a lower `--rhythm-threshold`. Very low thresholds can create excessive or
musically poor output, so review the result in the osu! editor.

## Development

Create the locked environment and run all checks:

```powershell
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run osumapper models verify
```

The fixture suite covers standard, taiko, catch, and mania. Tests include Unicode
metadata, negative offsets, safe archive extraction, deterministic package bytes,
22.05 kHz audio feature dimensions, golden hit-object output, preset selection,
model-parity manifests, malformed training maps, song-level split isolation,
feature-cache determinism, timing-grid alignment, a CPU training smoke run, model
reload, held-out evaluation, analysis output, and modern-engine generation.

Project layout:

```text
src/osumapper/           Modern application package
src/osumapper/training/  Local dataset, preprocessing, model, and evaluation code
models/                  Migrated legacy .keras models and parity manifests
models/modern/rhythm/    Ignored local modern-model output after training
models/modern/rhythm-conformer-v2/  Separate ignored Conformer-v2 output
models/modern/rhythm-conformer-v3/  Faster isolated Conformer-v3 output
models/modern/rhythm-conformer-v4-standard-stars/  Standard star-conditioned V4 output
models/modern/placement-v1/         Learned flow/object-type output
training_data/           Ignored metadata, imports, splits, features, and window shards
tests/                   Fixtures, golden outputs, and regression tests
legacy/                  Preservation and compatibility documentation
v6.2/                    Untouched historical v6.2 implementation
v7.0/                    Untouched historical v7.0 implementation
```

## Credits and project lineage

- **Original creator:** [kotritrona](https://github.com/kotritrona)
- **Original repository:** [github.com/kotritrona/osumapper](https://github.com/kotritrona/osumapper)
- **Original mapping guide:** [Complete guide: creating beatmap using osumapper](https://github.com/kotritrona/osumapper/wiki/Complete-guide:-creating-beatmap-using-osumapper)
- **2026 modernization:** Gerardo (Benji) Colegio and contributors
- **TimingAnlyz attribution from upstream:**
  [statementreply](https://osu.ppy.sh/users/126198). The historical executable is
  preserved for archaeology but is not executed by the modern package.

This derivative exists because of kotritrona's original research, implementation,
trained models, presets, documentation, and public release. Please retain this
credit and the repository's [`NOTICE`](NOTICE) when redistributing modified copies.

The same attribution is available from the installed package:

```powershell
uv run osumapper credits
```

## License and trademarks

The project remains available under the [Apache License 2.0](LICENSE). See
[`NOTICE`](NOTICE) for attribution and modification notices.

`osu!` is a trademark of ppy Pty Ltd. This community project is not affiliated with
or endorsed by ppy Pty Ltd.
