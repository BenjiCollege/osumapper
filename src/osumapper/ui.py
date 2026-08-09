from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from tkinterdnd2 import DND_FILES, TkinterDnD

from osumapper.presets import preset_names

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


def launch() -> None:
    root = TkinterDnD.Tk()
    root.title("osumapper for osu!lazer")
    root.geometry("720x500")
    root.minsize(620, 410)

    source = tk.StringVar()
    output = tk.StringVar()
    preset = tk.StringVar(value="default")
    seed = tk.StringVar(value="2026")
    should_open = tk.BooleanVar(value=True)
    messages: queue.Queue[str] = queue.Queue()

    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(6, weight=1)

    def set_source(path: Path) -> None:
        path = path.expanduser().resolve()
        if path.suffix.casefold() not in SUPPORTED_INPUTS:
            messagebox.showerror(
                "Unsupported input",
                "Drop an .osz, .osu, or supported audio file.",
            )
            return
        source.set(str(path))
        if not output.get():
            output.set(str(Path.cwd() / "output" / f"{path.stem}-osumapper.osz"))

    def accept_drop(event: tk.Event) -> None:
        dropped = root.tk.splitlist(event.data)
        if dropped:
            set_source(Path(dropped[0]))

    def choose_source() -> None:
        selected = filedialog.askopenfilename(
            title="Choose .osz, .osu, or audio",
            filetypes=[
                (
                    "osu! and audio",
                    "*.osz *.osu *.mp3 *.ogg *.wav *.flac *.m4a *.aac *.opus",
                ),
                ("All files", "*.*"),
            ],
        )
        if selected:
            set_source(Path(selected))

    def choose_output() -> None:
        selected = filedialog.asksaveasfilename(
            title="Save generated package",
            defaultextension=".osz",
            filetypes=[("osu! package", "*.osz")],
        )
        if selected:
            output.set(selected)

    drop_zone = ttk.Label(
        frame,
        text="Drop an .osz, .osu, or audio file here\n(or use Browse)",
        anchor="center",
        relief="groove",
        padding=14,
    )
    drop_zone.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))

    ttk.Label(frame, text="Input").grid(row=1, column=0, sticky="w", pady=4)
    input_entry = ttk.Entry(frame, textvariable=source)
    input_entry.grid(row=1, column=1, sticky="ew", padx=8)
    ttk.Button(frame, text="Browse...", command=choose_source).grid(row=1, column=2)
    ttk.Label(frame, text="Output").grid(row=2, column=0, sticky="w", pady=4)
    ttk.Entry(frame, textvariable=output).grid(row=2, column=1, sticky="ew", padx=8)
    ttk.Button(frame, text="Browse...", command=choose_output).grid(row=2, column=2)
    ttk.Label(frame, text="Preset").grid(row=3, column=0, sticky="w", pady=4)
    ttk.Combobox(frame, textvariable=preset, values=preset_names(), state="readonly").grid(
        row=3, column=1, sticky="w", padx=8
    )
    ttk.Label(frame, text="Seed").grid(row=4, column=0, sticky="w", pady=4)
    ttk.Entry(frame, textvariable=seed, width=14).grid(row=4, column=1, sticky="w", padx=8)
    ttk.Checkbutton(frame, text="Open in osu!lazer when finished", variable=should_open).grid(
        row=5, column=1, sticky="w", padx=8, pady=4
    )
    log = tk.Text(frame, height=12, state="disabled", wrap="word")
    log.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(12, 8))

    for target in (root, drop_zone, input_entry):
        target.drop_target_register(DND_FILES)
        target.dnd_bind("<<Drop>>", accept_drop)

    def append_log(text: str) -> None:
        log.configure(state="normal")
        log.insert("end", text)
        log.see("end")
        log.configure(state="disabled")

    def worker(command: list[str]) -> None:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert process.stdout is not None
        for line in process.stdout:
            messages.put(line)
        messages.put(f"\nProcess finished with exit code {process.wait()}.\n")
        messages.put("__DONE__")

    def start() -> None:
        if not source.get() or not output.get():
            messagebox.showerror("Missing input", "Choose an input and output file first.")
            return
        try:
            int(seed.get())
        except ValueError:
            messagebox.showerror("Invalid seed", "Seed must be an integer.")
            return
        command = [
            sys.executable,
            "-m",
            "osumapper",
            "generate",
            source.get(),
            "--output",
            output.get(),
            "--preset",
            preset.get(),
            "--seed",
            seed.get(),
        ]
        if should_open.get():
            command.append("--open")
        run_button.configure(state="disabled")
        append_log(f"Running: {' '.join(command)}\n\n")
        threading.Thread(target=worker, args=(command,), daemon=True).start()

    def poll() -> None:
        try:
            while True:
                message = messages.get_nowait()
                if message == "__DONE__":
                    run_button.configure(state="normal")
                else:
                    append_log(message)
        except queue.Empty:
            pass
        root.after(100, poll)

    run_button = ttk.Button(frame, text="Generate beatmap", command=start)
    run_button.grid(row=7, column=0, columnspan=3, pady=(4, 0))
    poll()
    root.mainloop()
