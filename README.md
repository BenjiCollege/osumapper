# osumapper

osumapper creates osu! beatmaps with the original v7 rhythm models and a modern,
reproducible Python application. The current application packages complete `.osz`
files for osu!lazer and never modifies Lazer's private `client.realm` or hashed file
storage.

The 2020 implementations remain untouched under [`v6.2/`](v6.2/) and
[`v7.0/`](v7.0/). See [`legacy/README.md`](legacy/README.md) for the preserved Git
baseline.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then from this directory run:

```powershell
uv sync
uv run osumapper doctor
uv run osumapper generate song.osz --preset default --open
```

`uv` reads `.python-version` and installs the tested Python 3.12 runtime when it is
not already available.

Inputs may be:

- an exported `.osz` package;
- a selected `.osu` difficulty, with `--audio` if its audio is not alongside it;
- an audio file, which uses the maintained cross-platform timing estimator.

Examples:

```powershell
# Generate from an exported Lazer package
uv run osumapper generate song.osz --preset sota --seed 2026 --open

# Generate from an explicit beatmap and audio
uv run osumapper generate map.osu --audio audio.mp3 --preset default

# Generate a 7K mania map directly from audio
uv run osumapper generate song.mp3 --mode mania --keys 7 --preset mania-highkey

# Use the deterministic flow fallback instead of per-map legacy GAN training
uv run osumapper generate song.osz --flow-engine deterministic

# Open the local drag-and-drop/file-picker interface
uv run osumapper ui
```

Generated packages are written under `output/` by default. They contain the source
package assets plus the newly generated difficulty. `--open` starts the installed
osu!lazer executable with the finished package.

## Presets and rulesets

Run `uv run osumapper presets` for the full list. Standard presets include
`default`, `sota`, `vtuber`, `flower`, `inst`, `lowbpm`, `tvsize`, `hard`, and
`normal`. `taiko`, `catch`, `mania-lowkey`, and `mania-highkey` select their
corresponding ruleset pipelines.

Generation is seeded (`2026` by default). The same package, model set, parameters,
runtime, and seed produce deterministic preprocessing and packaging. TensorFlow
deterministic operations are enabled where the platform supports them.

## osu!lazer workflow

Use Lazer's export feature to obtain `.osz` input. Import the generated `.osz` by
using `--open`, double-clicking it, or dragging it into Lazer. Do not copy files into
Lazer's application-data directory.

The old osu!stable map-list workflow remains available without registry or
`osu!.db` parsing:

```powershell
uv run osumapper stable-scan C:\osu! --mode standard --output maplist.txt
```

## Model migration and parity

The repository retains all 14 original extensionless Keras/HDF5 rhythm models.
Migrate them with:

```powershell
uv run osumapper models migrate
uv run osumapper models verify
```

Each model is loaded from a temporary `.h5` compatibility path, evaluated with a
fixed deterministic probe, saved as `.keras`, reloaded, and evaluated again. The
new model is installed only when the predictions match within tolerance. SHA-256
hashes, probe shapes, runtime versions, and maximum error are recorded beside each
model.

## Development

```powershell
uv sync
uv run pytest
uv run ruff check .
```

The fixtures cover standard, taiko, catch, and mania. Tests cover parser behavior,
negative offsets, Unicode metadata, safe `.osz` extraction, deterministic package
bytes, the 22.05 kHz audio feature shape, golden hit-object output, preset
resolution, and model parity manifests.

The modern package uses isolated temporary workspaces, `pathlib`, typed
configuration, checked subprocesses, explicit errors, and optional progress output.
Legacy modules are loaded only through the compatibility engine while their
algorithms are incrementally replaced behind regression tests.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
