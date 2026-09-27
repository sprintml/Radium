"""Lightweight score-cache helpers.

Pulled out of ``pipeline.eval_common`` so completeness checks at planning
time don't transitively import watermark/diffusion/torch backends. Pure
filesystem + JSON; safe to import on a login node without GPU stack.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict


def score_cache_path(scores_dir: str, wm_name: str) -> Path:
    """Return the path for a single-model score cache file."""
    return Path(scores_dir) / f"{wm_name}.json"


# Score cache schema: {"<image_index>": {"score": <float>, ...}}.
# Detection scores every image in the source directory once. Per-seed
# random sub-sampling happens at evaluation time.


def _is_legacy_cache(data: Dict[str, Any]) -> bool:
    """Return True if `data` looks like the old per-seed schema."""
    if not data:
        return False
    sample = next(iter(data.values()))
    return isinstance(sample, dict) and isinstance(sample.get("scores"), list)


def load_score_map(path: Path) -> Dict[int, Dict[str, Any]]:
    """Load a score cache as {int_index: {"score": ..., ...}}."""
    data = json.loads(path.read_text())
    if _is_legacy_cache(data):
        raise ValueError(
            f"Legacy per-seed score cache at {path}. The schema changed to "
            "{image_index: {score, ...}}; delete the file and re-run detection."
        )
    return {int(k): v for k, v in data.items()}


def save_score_map(path: Path, entries: Dict[int, Dict[str, Any]]) -> None:
    """Merge `entries` into the on-disk score cache at `path`."""
    existing: Dict[int, Dict[str, Any]] = {}
    if path.exists():
        existing = load_score_map(path)
    existing.update({int(k): v for k, v in entries.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({str(k): existing[k] for k in sorted(existing)})
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as f:
        f.write(payload)
        tmp = f.name
    os.replace(tmp, path)


def summarize_score_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Score cache missing: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid score cache JSON: {path}") from exc
    if _is_legacy_cache(data):
        raise ValueError(
            f"Legacy per-seed score cache at {path}; delete and re-run detection."
        )
    return {"path": str(path), "num_scores": len(data)}
