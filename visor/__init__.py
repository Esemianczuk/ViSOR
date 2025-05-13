"""
ViSOR package root
==================

Exposes only metadata and useful repo-level paths—no heavy
imports or side-effects.
"""

from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

# ------------------------------------------------------------------
# Package / project paths
# ------------------------------------------------------------------
PACKAGE_DIR: Path = Path(__file__).resolve().parent   # …/visor
PROJECT_ROOT: Path = PACKAGE_DIR.parent               # repo root
RENDERS_DIR: Path = PROJECT_ROOT / "renders"

# ------------------------------------------------------------------
# Version string
# ------------------------------------------------------------------
try:
    __version__: str = version("visor-3d")            # when pip-installed
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
