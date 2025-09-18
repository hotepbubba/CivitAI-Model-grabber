"""Utility script to build a standalone executable for the GUI application."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from PyInstaller.__main__ import run as pyinstaller_run


def build_executable(additional_args: Sequence[str] | None = None) -> None:
    """Build the GUI application into a standalone executable using PyInstaller."""

    script_dir = Path(__file__).resolve().parent
    entry_script = script_dir / "gui_app.py"
    if not entry_script.exists():
        raise FileNotFoundError(f"GUI entry script not found at {entry_script}")

    executable_name = "CivitaiModelDownloader"
    args = [
        "--noconfirm",
        "--onefile",
        "--windowed",
        f"--name={executable_name}",
        "--hidden-import=fetch_all_models",
        str(entry_script),
    ]

    if additional_args:
        args = list(additional_args) + args

    pyinstaller_run(args)

    suffix = ".exe" if os.name == "nt" else ""
    dist_path = script_dir / "dist" / f"{executable_name}{suffix}"
    print(f"Executable available at: {dist_path}")


if __name__ == "__main__":
    build_executable()
