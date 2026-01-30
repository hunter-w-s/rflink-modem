from __future__ import annotations

import re
from pathlib import Path


def repo_root() -> Path:
    """
    Find the repository root by walking upward until we find pyproject.toml.
    This is robust regardless of where tests live.
    """
    start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError("repo_root(): could not find pyproject.toml walking upward")


def sample_assets_dir() -> Path:
    """
    Committed sample data for tests and users.
    """
    return repo_root() / "tests" / "sample_assets"


def outputs_unrev_dir() -> Path:
    """
    Base directory for generated (unreviewed) test artifacts.
    Not committed to git.
    """
    p = repo_root() / "tests" / "outputs" / "test_results_unrev"
    p.mkdir(parents=True, exist_ok=True)
    return p


def outputs_unrev_test_dir(request) -> Path:
    """
    Per-test output directory derived from pytest nodeid, safe for Windows paths.
    """
    base = outputs_unrev_dir()

    nodeid = request.node.nodeid
    safe = nodeid.replace("::", "__").replace("/", "_").replace("\\", "_")
    safe = re.sub(r"[\[\]]", "", safe)

    p = base / safe
    p.mkdir(parents=True, exist_ok=True)
    return p
