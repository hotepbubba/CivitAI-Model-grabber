"""Build a Windows installer for the Civitai Model Downloader."""

from __future__ import annotations

from pathlib import Path

from nsist import InstallerBuilder
from nsist.configreader import get_installer_builder_args, read_and_validate


def build_installer(config_path: Path | str = "installer/windows_installer.cfg") -> Path:
    """Generate the Windows installer executable using Pynsist/NSIS.

    Parameters
    ----------
    config_path:
        Path to the installer configuration file. Defaults to
        ``installer/windows_installer.cfg`` relative to the project root.

    Returns
    -------
    Path
        The location of the generated installer executable.
    """

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Installer configuration not found: {config_file}")

    config = read_and_validate(str(config_file))
    builder_args = get_installer_builder_args(config)

    builder = InstallerBuilder(**builder_args)
    exit_code = builder.run()
    if exit_code:
        raise RuntimeError(f"NSIS build failed with exit code {exit_code}")

    build_dir = Path(builder_args.get("build_dir", "build/nsis"))
    installer_name = builder_args.get("installer_name", "installer.exe")
    return build_dir / installer_name


if __name__ == "__main__":
    output_path = build_installer()
    print(f"Installer written to: {output_path}")
