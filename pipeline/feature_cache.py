"""Feature-cache helpers.

Detection writes raw per-image feature vectors verbatim; the aggregation
evaluation stage reads them back. Two storage modes are supported (only the
first is implemented):

  * ``NPZ_STACKED`` (default) — a single ``<wm>.npz`` with parallel
    ``indices`` (int32) and ``features`` (float32, stacked) arrays. Suited to
    fixed-shape numpy features (e.g. StegaStamp probabilities, TrustMark
    decoder logits).

  * ``PER_IMAGE_PT`` — a directory ``<wm>/`` holding one ``<idx>.pt`` per
    image. Suited to variable-shape torch tensors (e.g. TreeRing's per-image
    Fourier latents). API is reserved here; the implementation is deferred.

The active mode is selected per call via the ``mode`` argument; callers
typically read it from ``wm_cfg.feature_cache_mode`` and pass it through.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np


# Storage mode identifiers. Strings (not enums) so configs can declare the
# mode by name without importing this module.
NPZ_STACKED: str = "npz_stacked"
PER_IMAGE_PT: str = "per_image_pt"

_KNOWN_MODES = (NPZ_STACKED, PER_IMAGE_PT)


def _check_mode(mode: str) -> str:
    if mode not in _KNOWN_MODES:
        raise ValueError(
            f"unknown feature cache mode {mode!r}; expected one of {_KNOWN_MODES}"
        )
    return mode


def feature_cache_path(features_dir: str, wm_name: str, mode: str) -> Path:
    """Return the on-disk handle for the feature cache.

    For ``NPZ_STACKED`` this is a file path ``<features_dir>/<wm>.npz``.
    For ``PER_IMAGE_PT`` it's a directory ``<features_dir>/<wm>/`` that
    contains one ``<idx>.pt`` per image.
    """
    _check_mode(mode)
    base = Path(features_dir) / wm_name
    if mode == NPZ_STACKED:
        return base.with_suffix(".npz")
    if mode == PER_IMAGE_PT:
        return base
    raise AssertionError("unreachable")


def load_feature_map(path: Path, mode: str) -> Dict[int, Any]:
    """Load a feature cache as ``{int_index: feature}``.

    For ``NPZ_STACKED`` features are ``np.ndarray`` of fixed shape.
    For ``PER_IMAGE_PT`` features are ``torch.Tensor`` of arbitrary shape.
    """
    _check_mode(mode)
    if mode == NPZ_STACKED:
        with np.load(path) as data:
            indices = data["indices"]
            features = data["features"]
        return {int(i): features[k] for k, i in enumerate(indices)}
    if mode == PER_IMAGE_PT:
        raise NotImplementedError(
            f"load_feature_map(mode={PER_IMAGE_PT!r}) is not implemented yet; "
            "see pipeline/feature_cache.py."
        )
    raise AssertionError("unreachable")


def save_feature_map(path: Path, entries: Dict[int, Any], mode: str) -> None:
    """Merge ``entries`` into the on-disk feature cache at ``path``.

    For ``NPZ_STACKED`` all vectors (existing + new) must share the same
    shape; mixing shapes raises ``ValueError``. Atomic write via tempfile +
    ``os.replace``.

    For ``PER_IMAGE_PT`` each entry is written as ``<path>/<idx>.pt``;
    arbitrary tensor shapes are allowed (each image is independent).
    """
    _check_mode(mode)
    if not entries:
        return
    if mode == NPZ_STACKED:
        existing: Dict[int, np.ndarray] = {}
        if path.exists():
            existing = load_feature_map(path, mode=mode)
        existing.update({int(k): np.asarray(v, dtype=np.float32) for k, v in entries.items()})

        shapes = {v.shape for v in existing.values()}
        if len(shapes) > 1:
            raise ValueError(
                f"Feature cache at {path} has mixed shapes {shapes}; "
                "detect_features must return uniform-shape vectors per watermark."
            )

        indices = np.array(sorted(existing.keys()), dtype=np.int32)
        features = np.stack([existing[int(i)] for i in indices]).astype(np.float32)

        path.parent.mkdir(parents=True, exist_ok=True)
        # np.savez appends .npz if missing; give the tempfile a .npz suffix so the
        # written-to path matches the path we then os.replace.
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".npz") as f:
            tmp = f.name
        np.savez(tmp, indices=indices, features=features)
        os.replace(tmp, path)
        return
    if mode == PER_IMAGE_PT:
        raise NotImplementedError(
            f"save_feature_map(mode={PER_IMAGE_PT!r}) is not implemented yet; "
            "see pipeline/feature_cache.py."
        )
    raise AssertionError("unreachable")


def feature_cache_indices(path: Path, mode: str) -> set[int]:
    """Return the set of image indices covered by the cache at ``path``."""
    _check_mode(mode)
    if mode == NPZ_STACKED:
        if not path.exists():
            return set()
        with np.load(path) as data:
            return {int(i) for i in data["indices"]}
    if mode == PER_IMAGE_PT:
        raise NotImplementedError(
            f"feature_cache_indices(mode={PER_IMAGE_PT!r}) is not implemented yet; "
            "see pipeline/feature_cache.py."
        )
    raise AssertionError("unreachable")
