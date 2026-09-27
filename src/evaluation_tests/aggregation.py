"""Aggregation evaluation: pool per-image features into one decoded message and
test bit-accuracy against two nulls in parallel:

  (a) Binomial(K, 0.5) — parametric null assuming an unbiased per-bit prior.
  (b) Bootstrap-from-reference — empirical null built from the same-mode clean
      reference features. The reference source is decided upstream by the
      evaluation stage's ``clean_reference_source`` flag (``mi_0pct`` /
      ``m1_clean`` / ``mi_arch_clean``); this test just consumes whichever
      feature cache that resolves to. Skipped when the reference cache is absent.

Detection always stores raw features in the watermark's native space
(probabilities for StegaStamp, logits for TrustMark). The test runs the
aggregation in *both* native space and (when ``feature_normalisation`` is
non-identity) the normalised probability space.

Config (eval_params):
    aggregation_n:            # samples to aggregate per seed (0 → use all available).
    aggregation_bootstrap_B:  # bootstrap draws for the reference null. Default 10000.
    alpha:                    significance level for the per-seed reject decisions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import binom

from .base import BaseEvaluationTest

logger = logging.getLogger(__name__)


def _features_path_for(scores_path: Path, wm_name: str, mode: str) -> Path:
    """Resolve ``<...>/scores/<wm>.json`` → feature cache handle for the given mode."""
    from pipeline.feature_cache import feature_cache_path
    return feature_cache_path(scores_path.parent.parent / "features", wm_name, mode=mode)


def _load_features_array(features_path: Path, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, features) arrays from a feature cache.

    Aggregation requires fixed-shape features stackable into (N, K). Modes
    that don't satisfy that contract (e.g. PER_IMAGE_PT for variable-shape
    tensors) raise NotImplementedError; callers should aggregate them via a
    watermark-specific path instead.
    """
    from pipeline.feature_cache import NPZ_STACKED
    if mode != NPZ_STACKED:
        raise NotImplementedError(
            f"AggregationTest currently only supports feature_cache_mode={NPZ_STACKED!r}; "
            f"got {mode!r}."
        )
    with np.load(features_path) as data:
        return np.array(data["indices"], dtype=np.int64), np.array(data["features"], dtype=np.float32)


def _required_feature_mode(wm_cfg: Dict[str, Any]) -> str:
    mode = wm_cfg.get("feature_cache_mode") if hasattr(wm_cfg, "get") else None
    if not mode:
        raise ValueError(
            "AggregationTest requires watermark.feature_cache_mode to be set in the "
            "watermark config (e.g. 'npz_stacked')."
        )
    return mode


def _draw_pair(
    seed: int,
    n: int,
    w_idx: np.ndarray,
    w_feat: np.ndarray,
    c_idx: np.ndarray | None,
    c_feat: np.ndarray | None,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Sample n suspect (Mi) features and matched clean-reference features for one seed."""
    rng = np.random.default_rng(seed)
    if mode in ("supervised", "baseline"):
        if c_idx is None:
            raise ValueError(f"Mode '{mode}' requires the clean-reference feature cache to be present.")
        common = np.intersect1d(w_idx, c_idx)
        if len(common) < n:
            raise ValueError(
                f"Need {n} paired indices for mode '{mode}', only {len(common)} available."
            )
        chosen = rng.choice(common, size=n, replace=False)
        w_pos = np.searchsorted(w_idx, chosen)
        c_pos = np.searchsorted(c_idx, chosen)
        return w_feat[w_pos], c_feat[c_pos]
    if mode == "unsupervised":
        if len(w_idx) < n:
            raise ValueError(f"M2 cache has {len(w_idx)} images; need {n}.")
        chosen_w = rng.choice(len(w_idx), size=n, replace=False)
        W = w_feat[chosen_w]
        if c_feat is None:
            return W, None
        rng2 = np.random.default_rng(seed + 1_000_003)
        if len(c_idx) < n:
            raise ValueError(f"Clean-reference cache has {len(c_idx)} images; need {n}.")
        chosen_c = rng2.choice(len(c_idx), size=n, replace=False)
        return W, c_feat[chosen_c]
    raise ValueError(f"Unknown evaluation mode '{mode}'")


_NORMALISERS = {
    "identity": lambda arr: arr,
    "sigmoid": lambda arr: 1.0 / (1.0 + np.exp(-arr)),
}


def _planted_bits(wm_cfg: Dict[str, Any]) -> np.ndarray:
    """Parse the planted bit message from a watermark config block."""
    import re

    params = (wm_cfg.get("params", {}) if hasattr(wm_cfg, "get") else {}) or {}
    msg = params.get("bitmessage", params.get("messages"))
    if msg is None:
        raise ValueError(
            "Aggregation evaluation requires a planted bit message under "
            "wm_cfg.params.bitmessage or wm_cfg.params.messages"
        )
    if isinstance(msg, str):
        return np.array([int(c) for c in re.findall(r"[01]", msg)], dtype=np.int32)
    return np.array(np.asarray(msg).flatten().astype(int), dtype=np.int32)


def _aggregation_metadata(wm_cfg: Dict[str, Any]):
    """Return (planted_bits, normalisation_kind, native_threshold) from config alone."""
    if not hasattr(wm_cfg, "get"):
        raise ValueError("wm_cfg must be a dict-like config block")
    threshold = float(wm_cfg.get("feature_threshold"))
    norm_kind = str(wm_cfg.get("feature_normalisation", "identity"))
    if norm_kind not in _NORMALISERS:
        raise ValueError(
            f"feature_normalisation={norm_kind!r} is not supported; "
            f"expected one of {list(_NORMALISERS)}"
        )
    return _planted_bits(wm_cfg), norm_kind, threshold


def _bootstrap_null(
    rng: np.random.Generator,
    C: np.ndarray,
    n: int,
    B: int,
    threshold: float,
    planted: np.ndarray,
    chunk: int = 200,
) -> np.ndarray:
    """B bootstrap aggregations of n clean-reference samples; returns (B,) bit-accuracies."""
    K = len(planted)
    null = np.empty(B, dtype=np.float32)
    for s in range(0, B, chunk):
        end = min(s + chunk, B)
        idx = rng.integers(0, len(C), size=(end - s, n))
        agg = C[idx].mean(axis=1)
        decoded = (agg[:, :K] > threshold).astype(np.int32)
        null[s:end] = (decoded == planted[None, :]).mean(axis=1)
    return null


def _run_one_space(
    space_name: str,
    W_use: np.ndarray,
    C_use: np.ndarray | None,
    w_idx: np.ndarray,
    c_idx: np.ndarray | None,
    threshold: float,
    planted: np.ndarray,
    n: int,
    seeds: List[int],
    mode: str,
    B: int,
    alpha: float,
) -> Dict[str, Any]:
    """One aggregation pass in a fixed feature space.

    For each seed: compute aggregated bit-accuracy, then both p-values:
      - p_binomial: Binomial(K, 0.5) — parametric null.
      - p_clean_ref:   bootstrap from same-mode clean-reference features (when available).
    """
    K = len(planted)
    has_clean_ref = C_use is not None
    per_seed = []
    for seed in seeds:
        W_s, C_s = _draw_pair(seed, n, w_idx, W_use, c_idx, C_use, mode)
        agg_w = W_s.mean(axis=0)
        decoded_w = (agg_w[:K] > threshold).astype(np.int32)
        n_correct = int((decoded_w == planted).sum())
        bit_acc = n_correct / K

        # Binomial null: always available.
        p_binomial = float(binom.sf(n_correct - 1, K, 0.5))

        # Clean-reference null: bootstrap from same-mode reference features.
        p_clean_ref: float | None = None
        null_mean: float | None = None
        if has_clean_ref:
            rng_b = np.random.default_rng(seed + 7919)
            ref = C_s if mode in ("supervised", "baseline") else C_use
            null = _bootstrap_null(rng_b, ref, n, B, threshold, planted)
            p_clean_ref = float((null >= bit_acc).mean())
            null_mean = float(null.mean())

        per_seed.append({
            "seed": seed,
            "bit_acc": bit_acc,
            "n_correct": n_correct,
            "K": K,
            "n_aggregated": n,
            "p_binomial": p_binomial,
            "p_clean_ref": p_clean_ref,
            "null_mean_bit_acc_clean_ref": null_mean,
            "rejected_binomial": bool(p_binomial < alpha),
            "rejected_clean_ref": bool(p_clean_ref < alpha) if p_clean_ref is not None else None,
        })

    summary: Dict[str, Any] = {
        "space": space_name,
        "threshold": threshold,
        "per_seed": per_seed,
        "mean_bit_acc": float(np.mean([r["bit_acc"] for r in per_seed])),
        "mean_p_binomial": float(np.mean([r["p_binomial"] for r in per_seed])),
        "rejection_rate_binomial": float(np.mean([r["rejected_binomial"] for r in per_seed])),
    }
    if has_clean_ref:
        summary["mean_p_clean_ref"] = float(np.mean([r["p_clean_ref"] for r in per_seed]))
        summary["rejection_rate_clean_ref"] = float(
            np.mean([r["rejected_clean_ref"] for r in per_seed])
        )
    return summary


class AggregationTest(BaseEvaluationTest):
    """One-shot bit-accuracy test on aggregated per-image features.

    Always reports the native-space aggregation. Additionally reports a
    normalised-space (e.g. sigmoid → probability) aggregation when the
    watermark's ``feature_normalisation`` is anything other than ``identity``.
    """

    def run_scenario(
        self,
        wm_cfg: Dict[str, Any],
        percentage: int,
        mode: str,
        eval_params: Dict[str, Any],
        seeds: List[int],
        w_scores_path: Path,
        clean_scores_path: Path,
        output_dir: str | None,
    ) -> Dict[str, Any]:
        wm_name = wm_cfg.get("method", "unknown")
        n_cfg = int(eval_params.get("aggregation_n", 0))
        B = int(eval_params.get("aggregation_bootstrap_B", 10_000))
        alpha = float(eval_params.get("alpha", 0.05))
        feature_mode = _required_feature_mode(wm_cfg)

        w_feat_path = _features_path_for(w_scores_path, wm_name, feature_mode)
        c_feat_path = _features_path_for(clean_scores_path, wm_name, feature_mode)

        if not w_feat_path.exists():
            return {
                "status": "failed",
                "watermark_name": wm_name,
                "percentage": percentage,
                "mode": mode,
                "error": f"Missing M2 feature cache: {w_feat_path}",
            }

        w_idx, W_raw = _load_features_array(w_feat_path, feature_mode)
        if c_feat_path.exists():
            c_idx, C_raw = _load_features_array(c_feat_path, feature_mode)
        else:
            c_idx, C_raw = None, None
            logger.warning(
                "[aggregation] %s %s%% %s — clean-reference feature cache missing (%s); "
                "skipping bootstrap-vs-clean-ref null, reporting Binomial only.",
                wm_name, percentage, mode, c_feat_path,
            )

        planted, norm_kind, native_threshold = _aggregation_metadata(wm_cfg)
        K = len(planted)
        if n_cfg > 0:
            n = n_cfg
        elif c_idx is None:
            n = len(w_idx)
        elif mode in ("supervised", "baseline"):
            # Paired modes need shared indices; auto-pick from the intersection
            # so a transient cache mismatch can't trigger a "need N, only M
            # available" crash.
            n = int(len(np.intersect1d(w_idx, c_idx)))
        else:
            n = min(len(w_idx), len(c_idx))

        spaces: list[Dict[str, Any]] = []
        spaces.append(
            _run_one_space(
                "native", W_raw, C_raw, w_idx, c_idx, native_threshold,
                planted, n, seeds, mode, B, alpha,
            )
        )
        if norm_kind != "identity":
            normalise_fn = _NORMALISERS[norm_kind]
            W_norm = normalise_fn(W_raw)
            C_norm = normalise_fn(C_raw) if C_raw is not None else None
            spaces.append(
                _run_one_space(
                    norm_kind, W_norm, C_norm, w_idx, c_idx, 0.5,
                    planted, n, seeds, mode, B, alpha,
                )
            )

        logger.info(
            "[aggregation] %s %s%% %s — n=%d B=%d K=%d spaces=%s clean_ref=%s",
            wm_name, percentage, mode, n, B, K,
            [s["space"] for s in spaces], c_feat_path.exists(),
        )

        return {
            "status": "success",
            "watermark_name": wm_name,
            "percentage": percentage,
            "mode": mode,
            "aggregation": {
                "n_aggregated": n,
                "K": K,
                "alpha": alpha,
                "bootstrap_B": B,
                "has_clean_reference": c_feat_path.exists(),
                "feature_normalisation": norm_kind,
                "spaces": spaces,
            },
        }
