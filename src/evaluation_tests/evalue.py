from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.evaluation_tests.metrics import tpr_at_x_fpr

from .base import BaseEvaluationTest

logger = logging.getLogger(__name__)


def evalue_test(w_scores, no_w_scores, alpha=0.01, lambda_max=0.5, seed=42,
                num_trials=20, step_n=None, continue_threshold=None):
    """One-sided e-value test via KS-prediction + ONS betting.

    H0: w_scores and no_w_scores come from the same distribution.
    H1: w_scores is stochastically greater than no_w_scores.

    Adaptive-budget mode (when ``step_n`` is set): scores are shuffled in
    independent ``step_n``-sized batches and concatenated in batch order. After
    each batch, the wealth process is checked: if it has crossed 1/alpha →
    reject; if ``log(W) < continue_threshold * log(1/alpha)`` → stop early; else
    consume another batch. 

    Returns ``rejected``, ``stopping_time``, ``e_value``, ``alpha``, ``wealth``;
    plus ``n_used`` and ``n_checkpoints`` when adaptive.
    """
    _e_val_dir = str(Path(__file__).resolve().parent / "e_value_test")
    if _e_val_dir not in sys.path:
        sys.path.insert(0, _e_val_dir)
    from SeqTestsUtils import ONSstrategy, get_stopping_time_from_wealth
    from KStest import KSprediction

    scores_A = np.array(w_scores, dtype=float).flatten()
    scores_B = np.array(no_w_scores, dtype=float).flatten()
    n = min(len(scores_A), len(scores_B))
    if n < 3:
        raise ValueError(
            f"Need at least 3 paired observations, got n={n} "
            f"(|A|={len(scores_A)}, |B|={len(scores_B)})"
        )
    adaptive = step_n is not None
    if adaptive:
        if step_n > n:
            raise ValueError(
                f"step_n ({step_n}) exceeds available samples (n={n}); "
                f"reduce num_evalue_images or provide more scores"
            )
        n = (n // step_n) * step_n  # truncate to whole batches
        n_batches = n // step_n

    threshold = 1.0 / alpha
    wealth_processes = np.zeros((num_trials, n))

    if adaptive:
        # KSprediction & ONSstrategy are causal (F[i]/Lambda[i] depend only on
        # entries 0..i), so KSprediction(X[:cp], Y[:cp]) is bit-identical to
        # KSprediction(X, Y)[:cp]. Build wealth lazily per batch and break out
        # on rejection / continue-threshold — we only pay for the prefix we
        # actually inspect. KSprediction is O(n³), so a stop at batch K_typ
        # costs ~K_typ⁴·step_n³/4 per trial vs (K_max·step_n)³ before
        # (≈3000× cut at K_typ=2, K_max=25; break-even near K_typ≈16).
        trial_perms = [
            np.concatenate([
                k * step_n + np.random.RandomState(seed + trial * 100003 + k).permutation(step_n)
                for k in range(n_batches)
            ])
            for trial in range(num_trials)
        ]
        rejected, stopping_time = False, n
        n_used, n_checkpoints = 0, 0
        for k in range(n_batches):
            cp = (k + 1) * step_n
            n_checkpoints, n_used = k + 1, cp
            for trial in range(num_trials):
                perm_cp = trial_perms[trial][:cp]
                X, Y = scores_B[perm_cp], scores_A[perm_cp]
                F = KSprediction(X, Y, direction=0)
                Lambda = ONSstrategy(F, lambda_max=lambda_max)
                wealth_processes[trial, :cp] = np.cumprod(1 + Lambda * F)
            W_partial = wealth_processes[:, :cp].mean(axis=0)
            crossed = np.where(W_partial >= threshold)[0]
            if len(crossed) > 0:
                rejected, stopping_time = True, int(crossed[0]) + 1
                break
            # Stop early when not promising; final batch always exits the loop.
            if cp == n or np.log(max(W_partial[cp - 1], 1e-300)) < continue_threshold * np.log(threshold):
                stopping_time = cp
                break
        W = wealth_processes[:, :n_used].mean(axis=0)
        e_value = float(np.mean(wealth_processes[:, stopping_time - 1]))
    else:
        for trial in range(num_trials):
            perm = np.random.RandomState(seed + trial).permutation(n)
            X, Y = scores_B[perm], scores_A[perm]
            F = KSprediction(X, Y, direction=0)
            Lambda = ONSstrategy(F, lambda_max=lambda_max)
            wealth_processes[trial] = np.cumprod(1 + Lambda * F)
        W = wealth_processes.mean(axis=0)
        rejected, stopping_time = get_stopping_time_from_wealth(W, alpha)
        e_value = float(np.mean(wealth_processes[:, stopping_time - 1]))

    out = {
        "rejected": bool(rejected),
        "stopping_time": int(stopping_time),
        "e_value": e_value,
        "alpha": alpha,
        "wealth": W,
    }
    if adaptive:
        out["n_used"] = int(n_used)
        out["n_checkpoints"] = int(n_checkpoints)
        out["batch_e_values"] = [
            float(np.mean(wealth_processes[:, (k + 1) * step_n - 1]))
            for k in range(n_checkpoints)
        ]
    return out


def _save_multiseed_wealth_plot(
    wealth_per_seed,
    per_seed,
    *,
    alpha: float,
    output_dir: str,
    wm_name: str,
    percentage,
    mode: str,
    output_addition=None,
    step_n: int | None = None,
    continue_threshold: float | None = None,
) -> None:
    """Overlay one wealth-process line per outer seed onto a single PNG.

    Highlights the rejection threshold 1/α and any seed-specific stopping
    times. When ``step_n``/``continue_threshold`` are passed (adaptive mode),
    also draws batch checkpoint markers and the continue threshold.
    Saved as evalue_plot_multi_{wm}_{pct}pct_{mode}.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rej_by_seed = {r["seed"]: (r.get("rejected"), r.get("stopping_time"))
                   for r in per_seed}
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    cmap = plt.get_cmap("tab10")
    max_len = 0
    for i, (seed, W) in enumerate(wealth_per_seed):
        rej, stop = rej_by_seed.get(seed, (False, len(W)))
        plot_end = stop if rej else len(W)
        max_len = max(max_len, plot_end)
        ax.plot(W[:plot_end], color=cmap(i % 10), linewidth=1.0,
                alpha=0.85, label=f"seed {seed}")
        if rej and stop is not None and stop < len(W):
            ax.scatter([stop], [W[stop - 1]], color=cmap(i % 10),
                       s=20, zorder=5)
    ax.axhline(y=1 / alpha, color="red", linestyle="--",
               label=f"Rejection threshold 1/α = {1/alpha:.0f}")
    ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5,
               label="Initial wealth")
    if continue_threshold is not None:
        ax.axhline(y=(1 / alpha) ** continue_threshold, color="orange", linestyle=":",
                   label=f"Continue threshold = (1/α)^{continue_threshold:g}")
    if step_n is not None and max_len > step_n:
        for k in range(1, max_len // step_n + 1):
            ax.axvline(x=k * step_n, color="gray", linestyle="--",
                       alpha=0.3, linewidth=0.7)
    ax.set_xlabel("Number of paired observations")
    ax.set_ylabel("Wealth")
    ax.set_yscale("log")
    ax.set_title(f"Wealth processes — {wm_name} {percentage}pct {mode} ({len(wealth_per_seed)} seeds)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)
    plt.tight_layout()

    parts = ["evalue_plot_multi", wm_name, f"{percentage}pct", mode]
    if output_addition:
        safe = str(output_addition).strip("/\\")
        if safe:
            parts.append(safe)
    out_path = os.path.join(output_dir, "_".join(parts) + ".png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved multi-seed wealth plot: {out_path}")


class EvalueTest(BaseEvaluationTest):
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

        num_evalue_images = int(eval_params.get("num_evalue_images", 200))
        max_evalue_images = int(eval_params.get("max_evalue_images", num_evalue_images))
        continue_threshold = float(eval_params.get("continue_threshold", 0.5))
        alpha = eval_params.get("alpha", 0.01)
        lambda_max = eval_params.get("lambda_max", 0.5)
        num_evalue_trials = int(eval_params.get("num_evalue_trials", 20))
        x_fpr = eval_params.get("x_fpr", 0.01)
        if max_evalue_images < num_evalue_images:
            raise ValueError(
                f"max_evalue_images ({max_evalue_images}) must be >= num_evalue_images ({num_evalue_images})"
            )
        adaptive = max_evalue_images > num_evalue_images
        sample_n = max_evalue_images if adaptive else num_evalue_images
        # evalue_test handles both modes; pass step/threshold only when adaptive.
        adaptive_kwargs = (
            {"step_n": num_evalue_images, "continue_threshold": continue_threshold}
            if adaptive else {}
        )

        logger.info(
            "[evalue] %s %spct %s — w: %s | clean: %s",
            wm_name, percentage, mode, w_scores_path, clean_scores_path,
        )

        results: Dict[str, Any] = {
            "watermark_name": wm_name,
            "percentage": percentage,
            "mode": mode,
            "status": "success",
        }

        per_seed = []
        wealth_per_seed: List[Any] = []
        for seed in seeds:
            w_s, no_w_s = self._sample_scores_for_mode(
                mode, w_scores_path, clean_scores_path,
                seed=seed, n=sample_n,
            )
            logger.info(f"  seed {seed}: sampled {len(w_s)} watermarked, {len(no_w_s)} clean scores ({mode})")
            r = evalue_test(w_s, no_w_s, alpha=alpha, lambda_max=lambda_max,
                            seed=0, num_trials=num_evalue_trials,
                            **adaptive_kwargs)
            wealth_per_seed.append((seed, r.pop("wealth")))
            r["seed"] = seed
            r["n_clean"] = len(no_w_s)
            r["n_watermarked"] = len(w_s)
            auc, acc, tpr_fpr = tpr_at_x_fpr(no_w_s, w_s, x=x_fpr)
            r["auc"] = float(auc)
            r["acc"] = float(acc)
            r[f"tpr_at_{x_fpr*100:.0f}pct_fpr"] = float(tpr_fpr)
            per_seed.append(r)

        if output_dir and wealth_per_seed:
            _save_multiseed_wealth_plot(
                wealth_per_seed, per_seed, alpha=alpha,
                output_dir=output_dir, wm_name=wm_name,
                percentage=percentage, mode=mode,
                output_addition=eval_params.get("output_addition"),
                step_n=num_evalue_images if adaptive else None,
                continue_threshold=continue_threshold if adaptive else None,
            )

        e_values = [r["e_value"] for r in per_seed]
        stopping_times = [r["stopping_time"] for r in per_seed]
        avg_e_value = float(np.mean(e_values))
        tpr = float(np.mean([int(r["rejected"]) for r in per_seed]))
        multiseed: Dict[str, Any] = {
            "per_seed": per_seed,
            "e_value": avg_e_value,
            "rejected": bool(avg_e_value > (1.0 / alpha)),
            "tpr": tpr,
            "fpr": alpha,
            "stopping_time": int(round(float(np.mean(stopping_times)))),
            "num_seeds": len(seeds),
            "alpha": alpha,
        }
        if adaptive:
            n_used_list = [int(r.get("n_used", num_evalue_images)) for r in per_seed]
            multiseed["n_used_avg"] = float(np.mean(n_used_list))
            multiseed["n_used_max"] = int(np.max(n_used_list))
            multiseed["n_used_total"] = int(np.sum(n_used_list))
            multiseed["max_evalue_images"] = max_evalue_images
            multiseed["step_evalue_images"] = num_evalue_images
            multiseed["continue_threshold"] = continue_threshold
            batch_lists = [r.get("batch_e_values", []) for r in per_seed]
            min_k = min((len(b) for b in batch_lists), default=0)
            multiseed["batch_e_values_avg"] = [
                float(np.mean([b[k] for b in batch_lists])) for k in range(min_k)
            ]
        results["evalue_multiseed"] = multiseed
        logger.info(
            f"  multi-seed result ({len(seeds)} seeds): avg e-value={avg_e_value:.3g}, "
            f"TPR={tpr:.3f}, alpha={alpha}"
            + (f", n_used avg={multiseed['n_used_avg']:.0f}/{max_evalue_images}" if adaptive else "")
        )

        return results
