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
- A local drag-and-drop/file-picker interface.
- Cross-platform timing estimation using librosa instead of `TimingAnlyz.exe`.
- Pure-Python beatmap conversion in the modern path instead of Node.js.
- Typed configuration, `pathlib`, checked subprocesses, progress output, and
  actionable errors.
- Fourteen migrated `.keras` rhythm models with deterministic prediction-parity
  manifests.
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

Drop an `.osz`, `.osu`, or supported audio file onto the window, or use the file
picker. Select a preset and seed, choose whether to open the result in osu!lazer,
and press **Generate beatmap**. Progress and diagnostic messages appear in the
window.

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

## Model migration and verification

The repository contains the fourteen original extensionless Keras/HDF5 rhythm
models and parity-checked `.keras` replacements. Normal generation uses the modern
models; migration is not part of routine setup.

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
and model-parity manifests.

Project layout:

```text
src/osumapper/     Modern application package
models/            Migrated .keras models and parity manifests
tests/             Fixtures, golden outputs, and regression tests
legacy/            Preservation and compatibility documentation
v6.2/              Untouched historical v6.2 implementation
v7.0/              Untouched historical v7.0 implementation
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
