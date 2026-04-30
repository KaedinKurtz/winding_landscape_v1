"""Deterministic hashing for design IDs and version stamping.

Design IDs must be reproducible across runs: same parameters → same hash. We use
SHA-256 of canonical JSON (sorted keys, no whitespace) truncated to 16 hex chars
(64 bits) -- collision-safe for landscapes far larger than V1 will ever produce
(birthday-bound at ~4 billion designs).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any

from winding_landscape import __version__


def design_hash(params: dict[str, Any]) -> str:
    """Return a 16-hex-char deterministic hash of a design parameter dict.

    Parameters
    ----------
    params : dict
        Any JSON-serializable mapping of design parameters. Must be comparable
        across runs -- pass primitives, not numpy types or dataclass instances.
    """
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _json_default(obj: Any) -> Any:
    """Coerce numpy scalars and similar to JSON-friendly primitives."""
    # numpy scalar protocol
    if hasattr(obj, "item"):
        return obj.item()
    # numpy arrays
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Cannot serialize object of type {type(obj).__name__}")


def get_code_version() -> str:
    """Return a version stamp combining package version and git commit (if available)."""
    base = __version__
    commit = _try_git_commit()
    if commit:
        return f"{base}+g{commit}"
    return base


def _try_git_commit() -> str | None:
    """Best-effort git short-hash lookup. Returns None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


GEOMETRY_EXTRACTION_VERSION = "v1.0.0"
"""Bumped manually whenever geometry extraction logic changes meaningfully."""
