"""Resolve the in-tree extension package for standalone script execution."""

from __future__ import annotations

import sys
from pathlib import Path


def add_project_source() -> None:
    """Make the project package and resource helpers importable."""
    project_root = Path(__file__).resolve().parents[1]
    source_dir = project_root / "source" / "mirrorme_rl_train"
    for path in (project_root, source_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
