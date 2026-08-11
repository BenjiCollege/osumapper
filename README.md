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

The original 2020 implementations are preserved unchanged under [`v6.2/`](v6.2/)
and [`v7.0/`](v7.0/). See [`legacy/README.md`](legacy/README.md) for the Git safety
baseline and compatibility policy.

## What the modernization adds

- A Python 3.12 package under `src/osumapper` with a straightforward CLI.
- A locked, reproducible environment managed by [uv](https://docs.astral.sh/uv/).
- Safe `.osz` extraction into isolated temporary directories.
- Complete, deterministic `.osz` export and optional opening in osu!lazer.
- Standard, taiko, catch, and mania generation paths.
- A professional local queue UI with multi-file/folder drag-and-drop, training
  controls, progress, retry/clear actions, and per-package Lazer import.
- Cross-platform timing estimation using librosa instead of `TimingAnlyz.exe`.
- Pure-Python beatmap conversion in the modern path instead of Node.js.
- Typed configuration, `pathlib`, checked subprocesses, progress output, and
  actionable errors.
- Fourteen migrated `.keras` rhythm models with deterministic prediction-parity
  manifests.
- An opt-in local osu!standard dataset, feature extraction, Transformer-v1 and
  Conformer-v2 training, validation-only calibration, held-out evaluation, and
  deterministic human-review packages.
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

Use **Clear queue** when you are ready for another song. The queue shows each
item's state and output, can be stopped, and can retry failed or stopped items.
Generation controls expose the preset, ruleset, seed, flow/rhythm engines, modern
model, validation-calibrated threshold override, density, difficulty, BPM,
offset, and mania keys. **Import each completed package into osu!lazer** opens
every successful `.osz` as it finishes.

The **Training lab** tab provides the same scan, statistics, split, feature,
training, calibration, held-out evaluation, and review-package commands described
below. The **Activity** tab keeps detailed process output and actionable errors.

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
| `--mode MODE` | Select `standard`, `taiko`, `catch`, or `mania`. |
| `--difficulty TEXT` | Select a difficulty from a multi-map `.osz`. |
| `--seed INTEGER` | Set the deterministic seed; the default is `2026`. |
| `--rhythm-engine legacy` | Use the preserved v7 rhythm path; this remains the default. |
| `--rhythm-engine modern` | Use a locally trained modern osu!standard rhythm model. |
| `--modern-model PATH` | Select a modern model directory or `.keras` file. |
| `--rhythm-threshold NUMBER` | Override the modern model's hit-probability threshold. |
| `--target-density NUMBER` | Cap modern output at a target number of objects per second. |
| `--flow-engine auto` | Use the compatible legacy flow model when available. |
| `--flow-engine legacy` | Require the legacy per-map flow model. |
| `--flow-engine deterministic` | Use the fast, seeded cross-platform flow implementation. |
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

Generated maps are starting points for human review. Automatic generation does not
guarantee ranking quality, accessibility, or compliance with current osu! ranking
criteria.

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

### 1. Scan and curate a local dataset

Point the scanner at an osu!stable-style Songs directory. Malformed maps, missing
audio, converted maps, duplicates, and maps outside the configured quality bounds
are recorded in `training_data/skipped.jsonl` instead of stopping the scan.

```powershell
uv run osumapper dataset scan "E:\Games\osu!\Songs"
uv run osumapper dataset stats
```

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

### 2. Split songs and cache features

```powershell
uv run osumapper dataset split --seed 2026
uv run osumapper dataset features
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

### 3. Preserve Transformer-v1 and train Conformer-v2

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

To continue an interrupted run with the same architecture, context radius, split,
and output directory:

```powershell
uv run osumapper train rhythm --resume --epochs 50 --seed 2026
```

Do not resume a model merely to increase its recorded epoch count. Each epoch is
now regression-tested to contain non-zero training and validation work. Historical
runs affected by the former alternating empty-epoch bug should remain snapshots;
start a clean corrected run in a new output directory.

`--device auto` selects an available GPU and otherwise uses CPU. You can require
one explicitly with `--device gpu` or `--device cpu`. TensorFlow training can need
substantial memory and time on a large collection.

### 4. Calibrate on validation, then evaluate once on unseen songs

Choose the operating threshold without looking at test songs:

```powershell
uv run osumapper train calibrate rhythm --model models/modern/rhythm
```

The command maximizes candidate-level F1 using validation songs only, breaks ties
toward the higher/safer threshold, writes `threshold_calibration.json`, and makes
generation, analysis, and evaluation use that threshold by default. An explicit
`--threshold` or `--rhythm-threshold` still overrides it.

```powershell
uv run osumapper train evaluate rhythm
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

### Recommended research order

For meaningful quality gains, curate and rate maps before scaling model size.
Train a rated-only Conformer-v2 on the unchanged seed-2026 split, calibrate on
validation songs, and compare the same test metrics and review-song rubric against
Transformer-v1. Review timing, rhythm choice, density, pattern readability,
spacing, and playability separately; one aggregate score can hide regressions.

After that comparison, the highest-value extensions are difficulty-conditioned
thresholds/density, per-song sampling so large mapsets cannot dominate, continuous
audio-frame prediction before timing-grid projection, and an optional maintained
beat/downbeat front end. On Windows, use TensorFlow through WSL2 for RTX GPU
training; native Windows TensorFlow remains CPU-only in this tested setup. Keep
third-party audio encoders optional until their license permits the way you intend
to distribute the resulting model.

Inspect one human map against the model and export a timestamp-by-timestamp JSON
and CSV timeline:

```powershell
uv run osumapper analyze "E:\Games\osu!\Songs\123 Artist - Song\map.osu" --format both
```

The report includes human hits, predicted hits, matched/missed/extra events, timing
error, beat position, onset strength, and prediction probability.

### 5. Generate with the trained model

```powershell
uv run osumapper generate song.osz --rhythm-engine modern --flow-engine deterministic --open
```

The modern engine predicts osu!standard timestamps, then passes them through the
existing placement, metadata, validation, packaging, and Lazer-import boundary.
Use `--modern-model PATH` for a non-default model, `--rhythm-threshold` to adjust
selection, and `--target-density` to cap objects per second. It exits with an
actionable error if the model selects no usable events. Omit `--rhythm-engine
modern` to retain the original behavior.

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
training_data/           Ignored local metadata, splits, and feature cache
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
