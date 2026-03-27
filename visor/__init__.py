"""
ViSOR package root
==================

Exposes only metadata and useful repo-level paths—no heavy
imports or side-effects.
"""

from importlib.metadata import version, PackageNotFoundError
import os
from pathlib import Path

# ------------------------------------------------------------------
# Package / project paths
# ------------------------------------------------------------------
PACKAGE_DIR: Path = Path(__file__).resolve().parent   # …/visor
PROJECT_ROOT: Path = PACKAGE_DIR.parent               # repo root


def _default_renders_dir() -> Path:
    override = os.environ.get("VISOR_RENDERS_DIR")
    if override:
        return Path(override).expanduser()

    renders1 = PROJECT_ROOT / "renders1"
    if renders1.is_dir():
        return renders1

    return PROJECT_ROOT / "renders"


RENDERS_DIR: Path = _default_renders_dir()

# ------------------------------------------------------------------
# Version string
# ------------------------------------------------------------------
try:
    __version__: str = version("visor")               # when pip-installed
except PackageNotFoundError:
    __version__ = "0.0.0+dev"                         # editable clone / source

# ------------------------------------------------------------------
# Public symbols
# ------------------------------------------------------------------
__all__ = [
    "__version__",
    "PROJECT_ROOT",
    "RENDERS_DIR",
]
