from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from tkinterdnd2 import DND_FILES, TkinterDnD

from osumapper.difficulty import (
    STANDARD_DIFFICULTY_KEYS,
    resolve_standard_difficulty,
    standard_difficulty,
)
from osumapper.errors import InputError
from osumapper.paths import project_root
from osumapper.presets import preset_names
from osumapper.training.config import DatasetPaths
from osumapper.training.storage import read_json

SUPPORTED_INPUTS = {
    ".osz",
    ".osu",
    ".mp3",
    ".ogg",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".opus",
}
AUDIO_INPUTS = SUPPORTED_INPUTS - {".osz", ".osu"}

BACKGROUND = "#0b1020"
PANEL = "#141b2d"
PANEL_ALT = "#1b2438"
TEXT = "#eef2ff"
MUTED = "#9aa6bd"
ACCENT = "#5b8cff"
ACCENT_ACTIVE = "#79a2ff"
SUCCESS = "#42c89a"
ERROR = "#ff6b7a"
DEFAULT_RHYTHM_MODEL = "rhythm-conformer-v5-curated-run1-seed-2026"
DEFAULT_V6_TRAINING_MODEL = "rhythm-conformer-v6-curated-657-songs-seed-2026"
DEFAULT_PLACEMENT_MODEL = "placement-v1"


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    preset: str = "default"
    mode: str = "auto"
    seed: int = 2026
    flow_engine: str = "auto"
    rhythm_engine: str = "legacy"
    modern_model: Path | None = None
    placement_model: Path | None = None
    threshold: float | None = None
    density: float | None = None
    difficulty: str | None = None
    difficulty_tier: str = "hard"
    target_stars: float = 3.35
    full_set: bool = False
    star_precision: float = 0.03
    calibration_attempts: int = 24
    star_calculator: str = "auto"
    bpm: float | None = None
    offset_ms: int | None = None
    key_count: int = 4
    open_in_lazer: bool = True


@dataclass(slots=True)
class QueueEntry:
    source: Path
    status: str = "Ready"
    output: Path | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessResult:
    exit_code: int
    error: str | None = None


_WINDOWS_DRIVE_PATH = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<tail>.*)$")
_STAR_CALIBRATION_ERROR = re.compile(
    r"could not reach (?P<tier>.+?) (?P<target>\d+(?:\.\d+)?)★.*?"
    r"Best result was (?P<best>\d+(?:\.\d+)?)★",
    re.IGNORECASE,
)


def summarize_process_error(message: str | None) -> str:
    """Turn a verbose CLI error into an actionable queue-row summary."""

    if not message:
        return "Generation failed; open Activity for details"
    normalized = " ".join(message.removeprefix("error:").strip().split())
    calibration = _STAR_CALIBRATION_ERROR.search(normalized)
    if calibration:
        return (
            f"{calibration.group('tier')}: best {calibration.group('best')}★ / "
            f"target {calibration.group('target')}★"
        )
    return normalized if len(normalized) <= 110 else f"{normalized[:107]}…"


def default_generation_models(root: Path) -> tuple[Path, Path]:
    modern_root = root / "models" / "modern"
    configured_rhythm = os.environ.get("OSUMAPPER_RHYTHM_MODEL")
    if configured_rhythm:
        rhythm = Path(configured_rhythm).expanduser()
    else:
        trained_v6 = modern_root / DEFAULT_V6_TRAINING_MODEL
        v6_model, v6_config = model_bundle_paths(trained_v6)
        rhythm = (
            trained_v6
            if v6_model.is_file() and v6_config.is_file()
            else modern_root / DEFAULT_RHYTHM_MODEL
        )
    placement = Path(
        os.environ.get("OSUMAPPER_PLACEMENT_MODEL", modern_root / DEFAULT_PLACEMENT_MODEL)
    ).expanduser()
    return rhythm, placement


def default_training_model(root: Path) -> Path:
    return Path(
        os.environ.get(
            "OSUMAPPER_TRAINING_MODEL",
            root / "models" / "modern" / DEFAULT_V6_TRAINING_MODEL,
        )
    ).expanduser()


def model_bundle_paths(model: Path) -> tuple[Path, Path]:
    if model.suffix.casefold() == ".keras":
        return model, model.parent / "config.json"
    return model / "model.keras", model / "config.json"


def _running_under_wsl() -> bool:
    return bool(os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"))


def normalize_input_path(value: str | Path) -> Path:
    raw = str(value).strip().strip('"')
    if not _running_under_wsl():
        return Path(raw)
    normalized = raw.replace("\\", "/")
    match = _WINDOWS_DRIVE_PATH.match(normalized)
    if match:
        return Path("/mnt") / match.group("drive").casefold() / match.group("tail")
    lowered = normalized.casefold()
    for prefix in ("//wsl.localhost/", "//wsl$/"):
        if lowered.startswith(prefix):
            parts = normalized[len(prefix) :].split("/", 1)
            return Path("/" + parts[1]) if len(parts) == 2 else Path("/")
    return Path(normalized)


def _default_input_directory() -> Path | None:
    if not _running_under_wsl():
        return None
    users = Path("/mnt/c/Users")
    if not users.is_dir():
        return None
    ignored = {"default", "default user", "public", "all users"}
    candidates = [
        path / "Downloads"
        for path in users.iterdir()
        if path.is_dir() and path.name.casefold() not in ignored and (path / "Downloads").is_dir()
    ]
    return sorted(candidates, key=lambda path: str(path).casefold())[0] if candidates else None


def discover_inputs(paths: list[str | Path], *, all_difficulties: bool = False) -> list[Path]:
    """Expand dropped files/folders into a stable, duplicate-free generation queue."""
    discovered: list[Path] = []
    for candidate in paths:
        path = normalize_input_path(candidate).expanduser().resolve()
        if path.is_file():
            if path.suffix.casefold() in SUPPORTED_INPUTS:
                discovered.append(path)
            continue
        if not path.is_dir():
            continue
        directories = [path, *(item for item in path.rglob("*") if item.is_dir())]
        for directory in sorted(directories, key=lambda item: str(item).casefold()):
            files = sorted(
                (item for item in directory.iterdir() if item.is_file()),
                key=lambda item: item.name.casefold(),
            )
            packages = [item for item in files if item.suffix.casefold() == ".osz"]
            maps = [item for item in files if item.suffix.casefold() == ".osu"]
            audio = [item for item in files if item.suffix.casefold() in AUDIO_INPUTS]
            discovered.extend(packages)
            if maps:
                discovered.extend(maps if all_difficulties else maps[:1])
            elif not packages:
                discovered.extend(audio)
    unique: dict[str, Path] = {}
    for path in discovered:
        unique.setdefault(os.path.normcase(str(path)), path)
    return list(unique.values())


def unique_output_path(path: Path, reserved: set[Path] | None = None) -> Path:
    reserved_paths = {item.resolve() for item in (reserved or set())}
    candidate = path.expanduser().resolve()
    number = 2
    while candidate.exists() or candidate in reserved_paths:
        candidate = path.with_name(f"{path.stem}-{number}{path.suffix}").expanduser().resolve()
        number += 1
    return candidate


def build_generate_command(source: Path, output: Path, options: GenerationOptions) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "osumapper",
        "generate",
        str(source),
        "--output",
        str(output),
        "--preset",
        options.preset,
        "--seed",
        str(options.seed),
        "--flow-engine",
        options.flow_engine,
        "--rhythm-engine",
        options.rhythm_engine,
        "--keys",
        str(options.key_count),
        "--star-calculator",
        options.star_calculator,
    ]
    if options.mode != "auto":
        command.extend(("--mode", options.mode))
    if options.modern_model is not None:
        command.extend(("--modern-model", str(options.modern_model)))
    if options.placement_model is not None:
        command.extend(("--placement-model", str(options.placement_model)))
    if options.threshold is not None:
        command.extend(("--rhythm-threshold", str(options.threshold)))
    if options.density is not None:
        command.extend(("--target-density", str(options.density)))
    if options.difficulty:
        command.extend(("--difficulty", options.difficulty))
    if options.full_set:
        command.extend(
            (
                "--full-set",
                "--star-precision",
                str(options.star_precision),
                "--calibration-attempts",
                str(options.calibration_attempts),
            )
        )
    else:
        command.extend(("--difficulty-tier", options.difficulty_tier))
        command.extend(("--target-stars", str(options.target_stars)))
    if options.bpm is not None:
        command.extend(("--bpm", str(options.bpm)))
    if options.offset_ms is not None:
        command.extend(("--offset", str(options.offset_ms)))
    if options.open_in_lazer:
        command.append("--open")
    return command


class OsumapperStudio:
    def __init__(self, root: Any) -> None:
        self.root = root
        self.root.title("osumapper Studio for osu!lazer")
        self.root.geometry("1120x780")
        self.root.minsize(940, 660)
        self.root.configure(background=BACKGROUND)
        self.events: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self.entries: dict[str, QueueEntry] = {}
        self.active_process: subprocess.Popen[str] | None = None
        self.stop_requested = False
        self.busy = False
        self._configure_style()
        self._variables()
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._poll)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Root.TFrame", background=BACKGROUND)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Card.TFrame", background=PANEL_ALT)
        style.configure("TLabel", background=PANEL, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED, background=BACKGROUND)
        style.configure(
            "Header.TLabel", background=BACKGROUND, foreground=TEXT, font=("Segoe UI Semibold", 23)
        )
        style.configure(
            "Subheader.TLabel", background=BACKGROUND, foreground=MUTED, font=("Segoe UI", 10)
        )
        style.configure(
            "CardTitle.TLabel",
            background=PANEL_ALT,
            foreground=TEXT,
            font=("Segoe UI Semibold", 11),
        )
        style.configure("CardText.TLabel", background=PANEL_ALT, foreground=MUTED)
        style.configure(
            "TButton", background=PANEL_ALT, foreground=TEXT, padding=(12, 7), borderwidth=0
        )
        style.map("TButton", background=[("active", "#28344e"), ("disabled", PANEL)])
        style.configure(
            "Accent.TButton", background=ACCENT, foreground="white", font=("Segoe UI Semibold", 10)
        )
        style.map("Accent.TButton", background=[("active", ACCENT_ACTIVE), ("disabled", "#334265")])
        style.configure("Danger.TButton", background="#4a2530", foreground="#ffb4bd")
        style.map("Danger.TButton", background=[("active", "#62303d")])
        style.configure(
            "TEntry",
            fieldbackground="#0f1627",
            foreground=TEXT,
            insertcolor=TEXT,
            bordercolor="#2b3855",
        )
        style.configure(
            "TCombobox",
            fieldbackground="#0f1627",
            background=PANEL_ALT,
            foreground=TEXT,
            arrowcolor=TEXT,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#0f1627")],
            selectbackground=[("readonly", "#0f1627")],
            selectforeground=[("readonly", TEXT)],
        )
        style.configure("TCheckbutton", background=PANEL_ALT, foreground=TEXT)
        style.map("TCheckbutton", background=[("active", PANEL_ALT)])
        style.configure("TNotebook", background=BACKGROUND, borderwidth=0)
        style.configure(
            "TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(18, 9), borderwidth=0
        )
        style.map(
            "TNotebook.Tab", background=[("selected", PANEL_ALT)], foreground=[("selected", TEXT)]
        )
        style.configure(
            "Treeview",
            background="#0f1627",
            fieldbackground="#0f1627",
            foreground=TEXT,
            rowheight=30,
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background=PANEL_ALT,
            foreground=MUTED,
            font=("Segoe UI Semibold", 9),
            borderwidth=0,
        )
        style.map(
            "Treeview", background=[("selected", "#29477b")], foreground=[("selected", "white")]
        )
        style.configure(
            "Horizontal.TProgressbar", troughcolor="#0f1627", background=ACCENT, borderwidth=0
        )

    def _variables(self) -> None:
        root = project_root()
        paths = DatasetPaths.at()
        dataset_config = read_json(paths.config, default={})
        default_rhythm_model, default_placement_model = default_generation_models(root)
        self.output_dir = tk.StringVar(value=str(root / "output"))
        self.preset = tk.StringVar(value="default")
        self.mode = tk.StringVar(value="standard")
        self.seed = tk.StringVar(value="2026")
        self.flow_engine = tk.StringVar(value="deterministic")
        self.rhythm_engine = tk.StringVar(value="modern")
        self.modern_model = tk.StringVar(value=str(default_rhythm_model))
        self.placement_model = tk.StringVar(value=str(default_placement_model))
        self.threshold = tk.StringVar()
        self.density = tk.StringVar()
        self.difficulty = tk.StringVar()
        self.difficulty_tier = tk.StringVar(value="hard")
        self.target_stars = tk.StringVar(value="3.35")
        self.difficulty_tier.trace_add("write", self._sync_target_stars)
        self.full_set = tk.BooleanVar(value=True)
        self.star_precision = tk.StringVar(value="0.03")
        self.calibration_attempts = tk.StringVar(value="24")
        self.star_calculator = tk.StringVar(value="lazer")
        self.bpm = tk.StringVar()
        self.offset = tk.StringVar()
        self.keys = tk.StringVar(value="4")
        # Import is deliberately opt-in: lazer does not replace local sets by
        # matching an audio filename or title, so unattended batches could
        # otherwise coexist with older V4 drafts.
        self.open_lazer = tk.BooleanVar(value=False)
        self.all_difficulties = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Ready")

        self.data_root = tk.StringVar(value=str(paths.root))
        self.songs_root = tk.StringVar(value=str(dataset_config.get("songs_root", "")))
        self.include_unrated = tk.BooleanVar(value=False)
        self.trust_imported_osz = tk.BooleanVar(value=False)
        self.train_architecture = tk.StringVar(value="conformer-v6")
        self.train_output = tk.StringVar(value=str(default_training_model(root)))
        self.placement_output = tk.StringVar(value=str(default_placement_model))
        self.epochs = tk.StringVar(value="50")
        self.batch_size = tk.StringVar(value="32")
        self.learning_rate = tk.StringVar(value="0.0005")
        self.sequence_length = tk.StringVar(value="512")
        self.context_radius = tk.StringVar(value="0")
        self.device = tk.StringVar(value="auto")
        self.precision = tk.StringVar(value="auto")
        self.xla = tk.StringVar(value="auto")
        self.window_cache = tk.StringVar(value="auto")
        self.balance_songs = tk.BooleanVar(value=True)
        self.early_stopping = tk.StringVar(value="8")
        self.weight_decay = tk.StringVar(value="0.0001")
        self.resume = tk.BooleanVar(value=False)
        self.review_count = tk.StringVar(value="5")

    def _build(self) -> None:
        shell = ttk.Frame(self.root, style="Root.TFrame", padding=(22, 16, 22, 16))
        shell.pack(fill="both", expand=True)
        header = ttk.Frame(shell, style="Root.TFrame")
        header.pack(fill="x", pady=(0, 13))
        ttk.Label(header, text="osumapper Studio", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Queue, train, review, and import osu!lazer beatmaps locally.",
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        notebook = ttk.Notebook(shell)
        notebook.pack(fill="both", expand=True)
        self.generate_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=14)
        self.training_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=14)
        self.activity_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=14)
        notebook.add(self.generate_tab, text="Generate queue")
        notebook.add(self.training_tab, text="Training lab")
        notebook.add(self.activity_tab, text="Activity")
        self._build_generate_tab()
        self._build_training_tab()
        self._build_activity_tab()
        footer = ttk.Frame(shell, style="Root.TFrame")
        footer.pack(fill="x", pady=(11, 0))
        ttk.Label(footer, textvariable=self.status, style="Muted.TLabel").pack(side="left")
        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100, length=260)
        self.progress.pack(side="right")

    def _build_generate_tab(self) -> None:
        pane = ttk.Panedwindow(self.generate_tab, orient="horizontal")
        pane.pack(fill="both", expand=True)
        queue_card = ttk.Frame(pane, style="Card.TFrame", padding=14)
        settings_shell = ttk.Frame(pane, style="Card.TFrame", width=330)
        actions = ttk.Frame(settings_shell, style="Card.TFrame", padding=(14, 9, 14, 14))
        actions.pack(fill="x", side="bottom")
        settings = self._scrollable_card(settings_shell)
        pane.add(queue_card, weight=3)
        pane.add(settings_shell, weight=2)

        ttk.Label(queue_card, text="Generation queue", style="CardTitle.TLabel").pack(anchor="w")
        self.drop_zone = ttk.Label(
            queue_card,
            text="Drop files or folders here  •  .osz, .osu, and audio",
            style="CardText.TLabel",
            anchor="center",
            padding=(12, 16),
        )
        self.drop_zone.pack(fill="x", pady=(10, 10))
        controls = ttk.Frame(queue_card, style="Card.TFrame")
        controls.pack(fill="x", pady=(0, 9))
        ttk.Button(controls, text="Add files", command=self._choose_files).pack(side="left")
        ttk.Button(controls, text="Add folder", command=self._choose_folder).pack(
            side="left", padx=6
        )
        ttk.Button(controls, text="Paste path", command=self._paste_path).pack(side="left")
        ttk.Button(controls, text="Remove", command=self._remove_selected).pack(side="left")
        ttk.Button(controls, text="Clear queue", command=self._clear_queue).pack(
            side="left", padx=6
        )
        ttk.Checkbutton(
            controls,
            text="Every difficulty",
            variable=self.all_difficulties,
            style="TCheckbutton",
        ).pack(side="right")
        tree_frame = ttk.Frame(queue_card, style="Card.TFrame")
        tree_frame.pack(fill="both", expand=True)
        self.queue_tree = ttk.Treeview(
            tree_frame, columns=("source", "type", "status", "output"), show="headings"
        )
        self.queue_tree.heading("source", text="INPUT")
        self.queue_tree.heading("type", text="TYPE")
        self.queue_tree.heading("status", text="STATUS")
        self.queue_tree.heading("output", text="OUTPUT")
        self.queue_tree.column("source", width=270, minwidth=170)
        self.queue_tree.column("type", width=60, anchor="center")
        self.queue_tree.column("status", width=85, anchor="center")
        self.queue_tree.column("output", width=190, minwidth=120)
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.queue_tree.yview)
        self.queue_tree.configure(yscrollcommand=scrollbar.set)
        self.queue_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._section(settings, "Output and import")
        self._path_row(settings, "Output folder", self.output_dir, self._choose_output_folder)
        ttk.Checkbutton(
            settings,
            text="Import completed packages into osu!lazer (adds; does not replace by title)",
            variable=self.open_lazer,
        ).pack(anchor="w", pady=(5, 10))
        ttk.Label(
            settings,
            text=(
                "Safe V4 replacement: generate the batch with import disabled, remove the old "
                "local set inside lazer, then import the validated V5 .osz. osumapper never "
                "edits lazer's hash store or client database."
            ),
            style="CardText.TLabel",
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))
        self._section(settings, "Generation controls")
        self._combo_row(settings, "Preset", self.preset, preset_names())
        self._combo_row(settings, "Mode", self.mode, ("standard",))
        self._combo_row(
            settings,
            "Flow",
            self.flow_engine,
            ("auto", "placement", "deterministic", "legacy"),
        )
        ttk.Label(
            settings,
            text=(
                "Modern auto/deterministic uses PatternPlanner-v1 for circles, "
                "sliders, spinners, jumps, streams, and stacks."
            ),
            style="Muted.TLabel",
            wraplength=390,
            justify="left",
        ).pack(fill="x", pady=(0, 5))
        self._combo_row(settings, "Rhythm", self.rhythm_engine, ("legacy", "modern"))
        self._entry_row(settings, "Seed", self.seed)
        self._entry_row(settings, "Source map selector", self.difficulty)
        ttk.Checkbutton(
            settings,
            text="Generate complete Easy–Expert+ set",
            variable=self.full_set,
        ).pack(anchor="w", pady=(4, 6))
        self._entry_row(settings, "Maximum star error", self.star_precision)
        self._entry_row(settings, "Calibration attempts", self.calibration_attempts)
        self._combo_row(
            settings,
            "Star calculator",
            self.star_calculator,
            ("auto", "lazer", "rosu"),
        )
        ttk.Label(
            settings,
            text=(
                "Precision automation first calibrates density, then locks the rhythm "
                "and fine-tunes PatternPlanner spacing against your installed osu!lazer. "
                "The rosu option is a legacy approximation for systems without lazer."
            ),
            style="Muted.TLabel",
            wraplength=390,
            justify="left",
        ).pack(fill="x", pady=(0, 5))
        self._combo_row(
            settings,
            "Output difficulty",
            self.difficulty_tier,
            STANDARD_DIFFICULTY_KEYS,
        )
        self._entry_row(settings, "Target stars", self.target_stars)
        ttk.Label(
            settings,
            text=(
                "Easy 0.0–1.99★  •  Normal 2.0–2.69★  •  Hard 2.7–3.99★\n"
                "Insane 4.0–5.29★  •  Expert 5.3–6.49★  •  Expert+ 6.5★+"
            ),
            style="CardText.TLabel",
            wraplength=390,
        ).pack(anchor="w", pady=(2, 7))
        self._entry_row(settings, "Modern model", self.modern_model)
        self._entry_row(settings, "Placement model (optional)", self.placement_model)
        ttk.Label(
            settings,
            text=(
                "Generated maps are local drafts. Current osu! ranking criteria do not "
                "permit generative tooling for ranking-bound objects, timing, or hitsounds. "
                "A criteria audit is written beside every standard .osz."
            ),
            style="CardText.TLabel",
            wraplength=390,
        ).pack(anchor="w", pady=(7, 3))
        advanced = ttk.LabelFrame(settings, text="Advanced overrides", padding=9)
        advanced.pack(fill="x", pady=(8, 10))
        self._entry_row(advanced, "Threshold", self.threshold)
        self._entry_row(advanced, "Density", self.density)
        self._entry_row(advanced, "BPM", self.bpm)
        self._entry_row(advanced, "Offset ms", self.offset)
        self.generate_button = ttk.Button(
            actions,
            text="Generate queued song(s)",
            style="Accent.TButton",
            command=self._start_queue,
        )
        self.generate_button.pack(side="left", fill="x", expand=True)
        self.stop_button = ttk.Button(
            actions, text="Stop", style="Danger.TButton", command=self._stop, state="disabled"
        )
        self.stop_button.pack(side="left", padx=(8, 0))

        for target in (self.root, self.drop_zone, self.queue_tree):
            target.drop_target_register(DND_FILES)
            target.dnd_bind("<<Drop>>", self._accept_drop)

    def _build_training_tab(self) -> None:
        columns = ttk.Panedwindow(self.training_tab, orient="horizontal")
        columns.pack(fill="both", expand=True)
        dataset_shell = ttk.Frame(columns, style="Card.TFrame")
        model_shell = ttk.Frame(columns, style="Card.TFrame")
        dataset = self._scrollable_card(dataset_shell)
        model = self._scrollable_card(model_shell)
        columns.add(dataset_shell, weight=1)
        columns.add(model_shell, weight=1)

        ttk.Label(dataset, text="Dataset workspace", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            dataset,
            text=(
                "Scan loose legacy .osu maps and .osz packages together without modifying "
                "your source folder."
            ),
            style="CardText.TLabel",
            wraplength=410,
        ).pack(anchor="w", pady=(3, 14))
        self._path_row(dataset, "Map source", self.songs_root, self._choose_songs_folder)
        self._path_row(dataset, "Data folder", self.data_root, self._choose_data_folder)
        ttk.Checkbutton(
            dataset,
            text="Include unrated maps (experimental; quality not guaranteed)",
            variable=self.include_unrated,
        ).pack(anchor="w", pady=(8, 12))
        ttk.Checkbutton(
            dataset,
            text="Mark imported .osz maps GOOD (only for reviewed/trusted maps)",
            variable=self.trust_imported_osz,
        ).pack(anchor="w", pady=(0, 8))
        dataset_buttons = ttk.Frame(dataset, style="Card.TFrame")
        dataset_buttons.pack(fill="x")
        for text, action in (
            ("1  Scan mixed source (.osu + .osz)", "scan"),
            ("2  Append reviewed .osz folder", "import-osz"),
            ("3  Stats", "stats"),
            ("4  Split", "split"),
            ("5  Freeze quality benchmark", "benchmark"),
            ("6  Features", "features"),
            ("7  Window shards", "windows"),
        ):
            ttk.Button(
                dataset_buttons,
                text=text,
                command=lambda selected=action: self._run_dataset_action(selected),
            ).pack(fill="x", pady=3)

        ttk.Separator(dataset).pack(fill="x", pady=18)
        ttk.Label(dataset, text="Validation and review", style="CardTitle.TLabel").pack(anchor="w")
        self._entry_row(dataset, "Review packages", self.review_count)
        ttk.Button(
            dataset, text="Calibrate threshold (validation only)", command=self._calibrate
        ).pack(fill="x", pady=3)
        ttk.Button(dataset, text="Evaluate held-out test split", command=self._evaluate).pack(
            fill="x", pady=3
        )
        ttk.Button(dataset, text="Generate held-out review packages", command=self._review).pack(
            fill="x", pady=3
        )
        ttk.Button(dataset, text="Analyze placement in a .osu map", command=self._placement).pack(
            fill="x", pady=3
        )
        ttk.Button(
            dataset,
            text="Audit osu!standard ranking criteria",
            command=self._criteria,
        ).pack(fill="x", pady=3)

        ttk.Label(model, text="Model training", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            model,
            text=(
                "Conformer-v5 shares one music encoder across six independently learned "
                "osu!standard rhythm heads, with extra Expert and Expert+ training weight."
            ),
            style="CardText.TLabel",
            wraplength=410,
        ).pack(anchor="w", pady=(3, 14))
        self._combo_row(
            model,
            "Architecture",
            self.train_architecture,
            (
                "conformer-v6",
                "conformer-v5",
                "conformer-v4",
                "conformer-v3",
                "conformer-v2",
                "transformer-v1",
            ),
        )

        self._path_row(model, "Model output", self.train_output, self._choose_train_output)
        self._entry_row(model, "Epochs", self.epochs)
        self._entry_row(model, "Batch size", self.batch_size)
        self._entry_row(model, "Learning rate", self.learning_rate)
        self._entry_row(model, "Sequence length", self.sequence_length)
        self._entry_row(model, "Audio context", self.context_radius)
        self._combo_row(model, "Device", self.device, ("auto", "cpu", "gpu"))
        performance = ttk.LabelFrame(model, text="GPU and data pipeline", padding=9)
        performance.pack(fill="x", pady=(8, 8))
        self._combo_row(
            performance,
            "Precision",
            self.precision,
            ("auto", "mixed-float16", "float32"),
        )
        self._combo_row(performance, "XLA", self.xla, ("auto", "on", "off"))
        self._combo_row(performance, "Window cache", self.window_cache, ("auto", "rebuild", "off"))
        self._entry_row(performance, "Early stop", self.early_stopping)
        self._entry_row(performance, "Weight decay", self.weight_decay)
        ttk.Checkbutton(
            performance,
            text="Balance songs (recommended)",
            variable=self.balance_songs,
        ).pack(anchor="w", pady=(4, 0))
        ttk.Checkbutton(model, text="Resume this exact run", variable=self.resume).pack(
            anchor="w", pady=(8, 12)
        )
        ttk.Label(
            model,
            text=(
                "V6 uses nested tier probabilities and hard-negative focal training. "
                "Resume only the exact same model output, architecture, and unchanged split. "
                "For changed data, leave Resume off and use this new V6 folder."
            ),
            style="CardText.TLabel",
            wraplength=410,
        ).pack(anchor="w", pady=(0, 12))
        self.train_button = ttk.Button(
            model, text="Start training", style="Accent.TButton", command=self._train
        )
        self.train_button.pack(fill="x", pady=3)
        ttk.Button(
            model, text="Stop active process", style="Danger.TButton", command=self._stop
        ).pack(fill="x", pady=3)
        ttk.Separator(model).pack(fill="x", pady=14)
        ttk.Label(model, text="Placement-v1", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            model,
            text=(
                "Learn relative flow, object types, slider lengths, and combo changes "
                "after rhythm-v5."
            ),
            style="CardText.TLabel",
            wraplength=410,
        ).pack(anchor="w", pady=(3, 8))
        self._entry_row(model, "Placement output", self.placement_output)
        ttk.Button(
            model,
            text="Start Placement-v1 training",
            style="Accent.TButton",
            command=self._train_placement,
        ).pack(fill="x", pady=3)
        ttk.Button(
            model,
            text="Evaluate Placement-v1 on test songs",
            command=self._evaluate_placement,
        ).pack(fill="x", pady=3)

    def _sync_target_stars(self, *_args: object) -> None:
        try:
            profile = standard_difficulty(self.difficulty_tier.get())
        except InputError:
            return
        self.target_stars.set(f"{profile.default_stars:.2f}")

    def _build_activity_tab(self) -> None:
        header = ttk.Frame(self.activity_tab, style="Panel.TFrame")
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="Run log").pack(side="left")
        ttk.Button(header, text="Clear log", command=self._clear_log).pack(side="right")
        ttk.Button(header, text="Open output folder", command=self._open_output_folder).pack(
            side="right", padx=6
        )
        self.log = tk.Text(
            self.activity_tab,
            background="#080d19",
            foreground="#d8e2f5",
            insertbackground=TEXT,
            selectbackground="#29477b",
            relief="flat",
            font=("Cascadia Mono", 9),
            wrap="word",
            padx=12,
            pady=12,
            state="disabled",
        )
        self.log.pack(fill="both", expand=True)
        self.log.tag_configure("error", foreground=ERROR)
        self.log.tag_configure("success", foreground=SUCCESS)
        self.log.tag_configure("muted", foreground=MUTED)

    @staticmethod
    def _section(parent: Any, text: str) -> None:
        ttk.Label(parent, text=text, style="CardTitle.TLabel").pack(anchor="w", pady=(0, 7))

    @staticmethod
    def _scrollable_card(parent: Any) -> ttk.Frame:
        canvas = tk.Canvas(
            parent,
            background=PANEL_ALT,
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        content = ttk.Frame(canvas, style="Card.TFrame", padding=16)
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window, width=event.width),
        )
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return content

    @staticmethod
    def _entry_row(parent: Any, label: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, style="CardText.TLabel", width=15).pack(side="left")
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)

    @staticmethod
    def _combo_row(parent: Any, label: str, variable: tk.StringVar, values: Any) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, style="CardText.TLabel", width=15).pack(side="left")
        ttk.Combobox(row, textvariable=variable, values=values, state="readonly").pack(
            side="left", fill="x", expand=True
        )

    @staticmethod
    def _path_row(parent: Any, label: str, variable: tk.StringVar, command: Any) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, style="CardText.TLabel", width=15).pack(side="left")
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="…", width=3, command=command).pack(side="left", padx=(5, 0))

    def _choose_files(self) -> None:
        initial = _default_input_directory()
        selected = filedialog.askopenfilenames(
            title="Add beatmaps or audio",
            initialdir=str(initial) if initial is not None else None,
            filetypes=[
                ("osu! and audio", "*.osz *.osu *.mp3 *.ogg *.wav *.flac *.m4a *.aac *.opus"),
                ("All files", "*.*"),
            ],
        )
        self._add_paths(list(selected))

    def _choose_folder(self) -> None:
        initial = _default_input_directory()
        selected = filedialog.askdirectory(
            title="Queue a folder",
            initialdir=str(initial) if initial is not None else None,
        )
        if selected:
            self._add_paths([selected])

    def _paste_path(self) -> None:
        selected = simpledialog.askstring(
            "Add Windows or WSL path",
            "Paste an audio, .osu, .osz, or folder path:",
            parent=self.root,
        )
        if selected:
            self._add_paths([selected])

    def _choose_output_folder(self) -> None:
        self._choose_directory(self.output_dir, "Choose generated-package folder")

    def _choose_songs_folder(self) -> None:
        self._choose_directory(self.songs_root, "Choose Songs or downloaded .osz folder")

    def _choose_data_folder(self) -> None:
        self._choose_directory(self.data_root, "Choose training-data folder")

    def _choose_train_output(self) -> None:
        self._choose_directory(self.train_output, "Choose model output folder")

    @staticmethod
    def _choose_directory(variable: tk.StringVar, title: str) -> None:
        selected = filedialog.askdirectory(title=title, initialdir=variable.get() or None)
        if selected:
            variable.set(selected)

    def _accept_drop(self, event: tk.Event) -> None:
        self._add_paths(list(self.root.tk.splitlist(event.data)))

    def _add_paths(self, paths: list[str | Path]) -> None:
        inputs = discover_inputs(paths, all_difficulties=self.all_difficulties.get())
        existing = {os.path.normcase(str(entry.source)) for entry in self.entries.values()}
        added = 0
        for source in inputs:
            key = os.path.normcase(str(source))
            if key in existing:
                continue
            identifier = self.queue_tree.insert(
                "", "end", values=(source.name, source.suffix[1:].upper(), "Ready", "—")
            )
            self.entries[identifier] = QueueEntry(source=source)
            existing.add(key)
            added += 1
        if not inputs:
            messagebox.showwarning(
                "No supported input", "No .osz, .osu, or supported audio files were found."
            )
        self.status.set(f"Added {added} item(s) • {len(self.entries)} queued")

    def _remove_selected(self) -> None:
        if self.busy:
            messagebox.showinfo("Queue is running", "Stop the active queue before removing items.")
            return
        for identifier in self.queue_tree.selection():
            self.entries.pop(identifier, None)
            self.queue_tree.delete(identifier)
        self.status.set(f"{len(self.entries)} item(s) queued")

    def _clear_queue(self) -> None:
        if self.busy:
            messagebox.showinfo("Queue is running", "Stop the active queue before clearing it.")
            return
        self.entries.clear()
        self.queue_tree.delete(*self.queue_tree.get_children())
        self.progress["value"] = 0
        self.status.set("Queue cleared • ready for another song")

    @staticmethod
    def _optional_float(value: str, label: str) -> float | None:
        if not value.strip():
            return None
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be a number.") from exc

    def _generation_options(self) -> GenerationOptions:
        try:
            seed = int(self.seed.get())
            keys = int(self.keys.get())
            offset = int(self.offset.get()) if self.offset.get().strip() else None
            calibration_attempts = int(self.calibration_attempts.get())
        except ValueError as exc:
            raise ValueError(
                "Seed, offset, and calibration attempts must be whole numbers."
            ) from exc
        if not 1 <= keys <= 18:
            raise ValueError("Mania keys must be between 1 and 18.")
        threshold = self._optional_float(self.threshold.get(), "Threshold")
        density = self._optional_float(self.density.get(), "Density")
        star_precision = self._optional_float(self.star_precision.get(), "Maximum star error")
        if star_precision is None or not 0.001 <= star_precision <= 0.25:
            raise ValueError("Maximum star error must be between 0.001 and 0.25 stars.")
        if not 1 <= calibration_attempts <= 30:
            raise ValueError("Calibration attempts must be between 1 and 30.")
        if self.full_set.get():
            if self.rhythm_engine.get() != "modern":
                raise ValueError("FullSet-v1 requires the modern rhythm engine.")
            if density is not None:
                raise ValueError("FullSet-v1 manages density per tier; clear Density.")
            resolved_difficulty = standard_difficulty("hard"), 3.35
        else:
            target_stars = self._optional_float(self.target_stars.get(), "Target stars")
            if target_stars is None:
                raise ValueError("Target stars is required for standard difficulty generation.")
            try:
                resolved_difficulty = resolve_standard_difficulty(
                    self.difficulty_tier.get(), target_stars
                )
            except InputError as exc:
                raise ValueError(str(exc)) from exc
            assert resolved_difficulty is not None
        if threshold is not None and not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1.")
        if density is not None and not 0 < density <= 20:
            raise ValueError("Density must be greater than 0 and at most 20.")
        model = (
            Path(self.modern_model.get()).expanduser().resolve()
            if self.modern_model.get().strip()
            else None
        )
        placement_model = (
            Path(self.placement_model.get()).expanduser().resolve()
            if self.placement_model.get().strip()
            else None
        )
        if self.rhythm_engine.get() == "modern":
            if model is None:
                raise ValueError("Choose the trained V5 rhythm model folder.")
            model_file, config_file = model_bundle_paths(model)
            if not model_file.is_file() or not config_file.is_file():
                raise ValueError(
                    f"The modern rhythm model is incomplete under {model}. "
                    "Expected model.keras and config.json."
                )
        if self.flow_engine.get() == "placement" and self.rhythm_engine.get() != "modern":
            raise ValueError("Placement-v1 requires the modern rhythm engine.")
        if self.flow_engine.get() == "placement":
            if placement_model is None:
                raise ValueError("Choose a trained Placement-v1 model folder.")
            placement_file, placement_config = model_bundle_paths(placement_model)
            if not placement_file.is_file() or not placement_config.is_file():
                raise ValueError(
                    f"Placement-v1 is not trained under {placement_model}. "
                    "Keep Flow set to deterministic or train Placement-v1 first."
                )
        return GenerationOptions(
            preset=self.preset.get(),
            mode=self.mode.get(),
            seed=seed,
            flow_engine=self.flow_engine.get(),
            rhythm_engine=self.rhythm_engine.get(),
            modern_model=model if self.rhythm_engine.get() == "modern" else None,
            placement_model=(placement_model if self.flow_engine.get() == "placement" else None),
            threshold=threshold,
            density=density,
            difficulty=self.difficulty.get().strip() or None,
            difficulty_tier=resolved_difficulty[0].key,
            target_stars=resolved_difficulty[1],
            full_set=self.full_set.get(),
            star_precision=star_precision,
            calibration_attempts=calibration_attempts,
            star_calculator=self.star_calculator.get(),
            bpm=self._optional_float(self.bpm.get(), "BPM"),
            offset_ms=offset,
            key_count=keys,
            open_in_lazer=self.open_lazer.get(),
        )

    def _start_queue(self) -> None:
        if self.busy:
            return
        ready = [
            (identifier, entry)
            for identifier, entry in self.entries.items()
            if entry.status in {"Ready", "Failed", "Stopped"}
        ]
        if not ready:
            messagebox.showinfo("Queue is empty", "Add files or a folder before running the queue.")
            return
        try:
            options = self._generation_options()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        output_root = Path(self.output_dir.get()).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        reserved: set[Path] = set()
        jobs: list[tuple[str, list[str], Path]] = []
        for identifier, entry in ready:
            if options.full_set:
                base = output_root / f"{entry.source.stem}-osumapper-full-set.osz"
            else:
                base = output_root / (
                    f"{entry.source.stem}-osumapper-{options.difficulty_tier}-"
                    f"{options.target_stars:.2f}stars.osz"
                )
            output = unique_output_path(base, reserved)
            reserved.add(output)
            entry.output = output
            jobs.append((identifier, build_generate_command(entry.source, output, options), output))
        self._set_busy(True, f"Running {len(jobs)} queued item(s)")
        threading.Thread(target=self._queue_worker, args=(jobs,), daemon=True).start()

    def _queue_worker(self, jobs: list[tuple[str, list[str], Path]]) -> None:
        completed = 0
        for identifier, command, output in jobs:
            if self.stop_requested:
                self.events.put(("item", identifier, "Stopped", output, None))
                continue
            self.events.put(("item", identifier, "Running", output, None))
            result = self._execute(command, label=f"Generate {Path(command[4]).name}")
            status = (
                "Completed"
                if result.exit_code == 0
                else "Stopped"
                if self.stop_requested
                else "Failed"
            )
            detail = summarize_process_error(result.error) if status == "Failed" else None
            self.events.put(("item", identifier, status, output, detail))
            completed += 1
            self.events.put(("progress", completed / len(jobs) * 100))
        self.events.put(("done", "Queue stopped" if self.stop_requested else "Queue finished"))

    def _execute(self, command: list[str], *, label: str) -> ProcessResult:
        self.events.put(("log", f"\n▶ {label}\n{subprocess.list2cmdline(command)}\n", "muted"))
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        last_error: str | None = None
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
            self.active_process = process
            assert process.stdout is not None
            for line in process.stdout:
                is_error = "error:" in line.casefold()
                if is_error:
                    last_error = line.strip()
                self.events.put(("log", line, "error" if is_error else None))
            code = process.wait()
        except OSError as exc:
            last_error = f"Could not start process: {exc}"
            self.events.put(("log", f"{last_error}\n", "error"))
            code = 1
        finally:
            self.active_process = None
        self.events.put(
            ("log", f"Process exited with code {code}.\n", "success" if code == 0 else "error")
        )
        return ProcessResult(code, last_error)

    def _run_single(self, command: list[str], label: str) -> None:
        if self.busy:
            messagebox.showinfo("Process active", "Stop or wait for the active process first.")
            return
        self._set_busy(True, label)

        def worker() -> None:
            result = self._execute(command, label=label)
            self.events.put(("progress", 100 if result.exit_code == 0 else 0))
            self.events.put(
                (
                    "done",
                    f"{label} completed" if result.exit_code == 0 else f"{label} failed",
                )
            )

        threading.Thread(target=worker, daemon=True).start()

    def _base_command(self) -> list[str]:
        return [sys.executable, "-m", "osumapper"]

    def _run_dataset_action(self, action: str) -> None:
        data = self.data_root.get().strip()
        if not data:
            messagebox.showerror("Missing data folder", "Choose a training-data folder.")
            return
        command = self._base_command() + ["dataset", action]
        if action in {"scan", "import-osz"}:
            songs = self.songs_root.get().strip()
            if not songs:
                messagebox.showerror(
                    "Missing source folder",
                    "Choose an osu!stable Songs folder or a folder containing .osz packages.",
                )
                return
            command.append(songs)
        command.extend(("--data-root", data))
        if action == "import-osz" and self.trust_imported_osz.get():
            command.extend(("--rating", "good"))
        if action == "split":
            command.extend(("--seed", self.seed.get()))
            if self.include_unrated.get():
                command.append("--include-unrated")
        if action == "windows":
            command.extend(
                (
                    "--sequence-length",
                    self.sequence_length.get(),
                    "--audio-context-radius",
                    self.context_radius.get(),
                    "--architecture",
                    self.train_architecture.get(),
                )
            )
        self._run_single(command, f"Dataset {action}")

    def _training_model_root(self) -> str:
        value = self.train_output.get().strip()
        if not value:
            raise ValueError("Choose a model output folder.")
        return value

    def _train(self) -> None:
        try:
            for value, label in (
                (self.epochs.get(), "Epochs"),
                (self.batch_size.get(), "Batch size"),
                (self.sequence_length.get(), "Sequence length"),
                (self.context_radius.get(), "Audio context"),
                (self.early_stopping.get(), "Early stopping"),
            ):
                parsed = int(value)
                if parsed <= 0 and label != "Audio context":
                    raise ValueError(f"{label} must be positive.")
                if label == "Audio context" and not 0 <= parsed <= 32:
                    raise ValueError("Audio context must be between 0 and 32.")
            if float(self.learning_rate.get()) <= 0:
                raise ValueError("Learning rate must be positive.")
            if float(self.weight_decay.get()) < 0:
                raise ValueError("Weight decay cannot be negative.")
            output = self._training_model_root()
        except ValueError as exc:
            messagebox.showerror("Invalid training settings", str(exc))
            return
        command = self._base_command() + [
            "train",
            "rhythm",
            "--data-root",
            self.data_root.get(),
            "--output",
            output,
            "--architecture",
            self.train_architecture.get(),
            "--epochs",
            self.epochs.get(),
            "--batch-size",
            self.batch_size.get(),
            "--learning-rate",
            self.learning_rate.get(),
            "--sequence-length",
            self.sequence_length.get(),
            "--audio-context-radius",
            self.context_radius.get(),
            "--device",
            self.device.get(),
            "--precision",
            self.precision.get(),
            "--xla",
            self.xla.get(),
            "--window-cache",
            self.window_cache.get(),
            "--early-stopping-patience",
            self.early_stopping.get(),
            "--weight-decay",
            self.weight_decay.get(),
            "--seed",
            self.seed.get(),
        ]
        if self.resume.get():
            command.append("--resume")
        if not self.balance_songs.get():
            command.append("--no-balance-songs")
        self._run_single(command, f"Train {self.train_architecture.get()}")

    def _placement(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose an osu!standard beatmap",
            filetypes=(("osu! beatmap", "*.osu"),),
        )
        if not selected:
            return
        output = Path(self.output_dir.get()) / f"{Path(selected).stem}-placement-analysis.json"
        command = self._base_command() + [
            "placement",
            "analyze",
            selected,
            "--output",
            str(output),
        ]
        self._run_single(command, "Placement analysis")

    def _criteria(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose an osu!standard beatmap",
            filetypes=(("osu! beatmap", "*.osu"),),
        )
        if not selected:
            return
        output = Path(self.output_dir.get()) / f"{Path(selected).stem}-criteria.json"
        command = self._base_command() + [
            "criteria",
            "check",
            selected,
            "--output",
            str(output),
        ]
        self._run_single(command, "osu!standard criteria audit")

    def _train_placement(self) -> None:
        output = self.placement_output.get().strip()
        if not output:
            messagebox.showerror("Missing placement output", "Choose a Placement-v1 output folder.")
            return
        command = self._base_command() + [
            "train",
            "placement",
            "--data-root",
            self.data_root.get(),
            "--output",
            output,
            "--epochs",
            self.epochs.get(),
            "--batch-size",
            self.batch_size.get(),
            "--learning-rate",
            self.learning_rate.get(),
            "--sequence-length",
            "256",
            "--device",
            self.device.get(),
            "--precision",
            self.precision.get(),
            "--xla",
            self.xla.get(),
            "--early-stopping-patience",
            self.early_stopping.get(),
            "--weight-decay",
            self.weight_decay.get(),
            "--seed",
            self.seed.get(),
        ]
        if self.resume.get():
            command.append("--resume")
        if not self.balance_songs.get():
            command.append("--no-balance-songs")
        self._run_single(command, "Train Placement-v1")

    def _evaluate_placement(self) -> None:
        model = self.placement_output.get().strip()
        if not model:
            messagebox.showerror("Missing placement model", "Choose a Placement-v1 folder.")
            return
        command = self._base_command() + [
            "train",
            "evaluate",
            "placement",
            "--data-root",
            self.data_root.get(),
            "--model",
            model,
        ]
        self._run_single(command, "Evaluate Placement-v1")

    def _calibrate(self) -> None:
        try:
            model = self._training_model_root()
        except ValueError as exc:
            messagebox.showerror("Missing model", str(exc))
            return
        command = self._base_command() + [
            "train",
            "calibrate",
            "rhythm",
            "--data-root",
            self.data_root.get(),
            "--model",
            model,
        ]
        self._run_single(command, "Validation threshold calibration")

    def _evaluate(self) -> None:
        try:
            model = self._training_model_root()
        except ValueError as exc:
            messagebox.showerror("Missing model", str(exc))
            return
        command = self._base_command() + [
            "train",
            "evaluate",
            "rhythm",
            "--data-root",
            self.data_root.get(),
            "--model",
            model,
        ]
        self._run_single(command, "Held-out test evaluation")

    def _review(self) -> None:
        try:
            model = self._training_model_root()
            count = int(self.review_count.get())
            if count <= 0:
                raise ValueError("Review count must be positive.")
        except ValueError as exc:
            messagebox.showerror("Invalid review settings", str(exc))
            return
        output = Path(self.output_dir.get()) / "held-out-review"
        command = self._base_command() + [
            "train",
            "review",
            "rhythm",
            "--data-root",
            self.data_root.get(),
            "--model",
            model,
            "--output",
            str(output),
            "--count",
            str(count),
            "--seed",
            self.seed.get(),
        ]
        if self.open_lazer.get():
            command.append("--open")
        self._run_single(command, "Generate held-out review packages")

    def _set_busy(self, busy: bool, status: str) -> None:
        self.busy = busy
        self.stop_requested = False if busy else self.stop_requested
        self.generate_button.configure(state="disabled" if busy else "normal")
        self.train_button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy else "disabled")
        self.status.set(status)
        if busy:
            self.progress["value"] = 0

    def _stop(self) -> None:
        if not self.busy:
            return
        self.stop_requested = True
        self.status.set("Stopping active process…")
        process = self.active_process
        if process is not None and process.poll() is None:
            process.terminate()

    def _append_log(self, text: str, tag: str | None = None) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text, tag or ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _open_output_folder(self) -> None:
        output = Path(self.output_dir.get()).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(output)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(output)])

    def _poll(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "log":
                    self._append_log(event[1], event[2])
                elif kind == "item":
                    _, identifier, status, output, detail = event
                    entry = self.entries.get(identifier)
                    if entry is not None:
                        entry.status = status
                        entry.output = output
                        entry.detail = detail
                        self.queue_tree.set(identifier, "status", status)
                        self.queue_tree.set(
                            identifier,
                            "output",
                            detail or Path(output).name,
                        )
                elif kind == "progress":
                    self.progress["value"] = event[1]
                elif kind == "done":
                    self._set_busy(False, event[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _close(self) -> None:
        process = self.active_process
        if process is not None and process.poll() is None:
            if not messagebox.askyesno("Process active", "Stop the active process and close?"):
                return
            process.terminate()
        self.root.destroy()


def launch() -> None:
    root = TkinterDnD.Tk()
    OsumapperStudio(root)
    root.mainloop()
