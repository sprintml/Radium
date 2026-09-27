"""Supervision-degree ablation for watermark radioactivity.

Fixes M2 at 100% watermarked and varies the supervision degree d ∈ [0, 1].
At degree d:
  - d·N images come from M2 supervised folder  (training prompts Alice knows)
  - (1-d)·N images come from M2 unsupervised folder (fresh prompts)
  - Clean references are matched: d·N from M1 supervised clean, (1-d)·N from M1 unsupervised clean

d=1.0 → fully supervised   (should match existing supervised result)
d=0.0 → fully unsupervised (should match existing unsupervised result)

Usage:
    python pipeline/evaluate_mode_supervision.py \
        --config_path configs/supervision/supervision_degree.yaml \
        --watermark_config configs/watermarks/bitmark.yaml \
        --output_dir results/
"""

from __future__ import annotations

from omegaconf import OmegaConf
import argparse
import json
import os
import sys
from typing import List, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_config
from src.utils.commonargs import add_watermark_config_arg
from src.utils.io_utils import load_paired_images
from src.evaluation_tests.metrics import tpr_at_x_fpr
from src.evaluation_tests.welch import dataset_inference
from src.paths import (
    ProjectPaths,
    build_run_id,
    prompt_set_name,
    resolve_m1_model_type,
    resolve_mi_model_type,
    resolve_wm_params,
)
from pipeline.eval_common import (
    evaluate_images,
    fnames_to_indices,
    get_watermark_from_config,
    json_safe,
    set_seed,
)

FIXED_PERCENTAGE: int = 100
DEFAULT_DEGREES: List[float] = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]



def run_degree_evaluation(
    config_base: Dict[str, Any],
    watermark,
    wm_cfg,
    m2_sup_dir: str,
    m2_uns_dir: str,
    m1_sup_dir: str,
    m1_uns_dir: str,
    eval_params: Dict[str, Any],
    seeds: List[int],
    degrees: List[float],
) -> List[Dict[str, Any]]:

    max_dataset_inference_images = int(eval_params.get("max_dataset_inference_images", 100))
    x_fpr = eval_params.get("x_fpr", 0.01)
    alpha = eval_params.get("alpha", 0.01)
    wm_name = wm_cfg.get("method", "unknown")

    all_results = []

    for d in degrees:
        n_sup = round(d * max_dataset_inference_images)
        n_uns = max_dataset_inference_images - n_sup

        print(f"\n{'='*70}")
        print(f"Supervision degree d={d:.2f}  →  {n_sup} supervised + {n_uns} unsupervised")
        print(f"{'='*70}")

        p_values = []
        tpr_values = []
        auc_values = []
        acc_values = []
        per_seed_results = []

        n_clean = 0
        n_watermarked = 0
        n_dataset_inference = 0

        for seed in seeds:
            set_seed(seed)

            # supervised pair
            if n_sup > 0:
                sup_clean, sup_wm, sup_fnames = load_paired_images(
                    m1_sup_dir, m2_sup_dir, config_base, finetuning_config=None, limit=n_sup,
                )
            else:
                sup_clean, sup_wm, sup_fnames = [], [], []

            # unsupervised pair
            if n_uns > 0:
                uns_clean, uns_wm, uns_fnames = load_paired_images(
                    m1_uns_dir, m2_uns_dir, config_base, finetuning_config=None, limit=n_uns,
                )
            else:
                uns_clean, uns_wm, uns_fnames = [], [], []

            clean_images = list(sup_clean) + list(uns_clean)
            watermarked_images = list(sup_wm) + list(uns_wm)

            no_w_scores = evaluate_images(watermark, clean_images)
            w_scores = evaluate_images(watermark, watermarked_images)

            n_dataset_inference = min(max_dataset_inference_images, len(no_w_scores), len(w_scores))
            inference = dataset_inference(
                w_scores[:n_dataset_inference],
                no_w_scores[:n_dataset_inference],
                alpha=alpha,
            )
            auc, acc, tpr = tpr_at_x_fpr(no_w_scores, w_scores, x=x_fpr)

            p_val = float(inference.get("p_value", float("nan")))
            log_p = float(inference.get("-log(p_value)", float("nan")))

            p_values.append(p_val)
            tpr_values.append(float(tpr))
            auc_values.append(float(auc))
            acc_values.append(float(acc))

            per_seed_results.append({
                "seed": int(seed),
                "p_value": p_val,
                "-log(p_value)": log_p,
                "tpr_at_x_fpr": float(tpr),
                "auc": float(auc),
                "acc": float(acc),
                "n_clean": len(no_w_scores),
                "n_watermarked": len(w_scores),
                "n_dataset_inference": n_dataset_inference,
                "n_sup": n_sup,
                "n_uns": n_uns,
                "sampled_indices": {
                    "supervised": fnames_to_indices(sup_fnames),
                    "unsupervised": fnames_to_indices(uns_fnames),
                },
            })

            n_clean = len(no_w_scores)
            n_watermarked = len(w_scores)

            print(
                f"Seed {seed}: AUC={auc:.4f}, ACC={acc:.4f}, "
                f"TPR@{x_fpr*100:.1f}%FPR={tpr:.4f}, p-value={p_val:.6e}"
            )

        result = {
            "watermark_name": wm_name,
            "supervision_degree": d,
            "n_sup": n_sup,
            "n_uns": n_uns,
            "n_clean": n_clean,
            "n_watermarked": n_watermarked,
            "n_dataset_inference": n_dataset_inference,
            "seed_values": {
                "seeds": [int(s) for s in seeds],
                "p_values": p_values,
                "tprs_at_x_fpr": tpr_values,
                "aucs": auc_values,
                "accs": acc_values,
            },
            "per_seed": per_seed_results,
            "p_value_mean": float(np.nanmean(p_values)),
            "p_value_std": float(np.nanstd(p_values)),
            "tpr_at_x_fpr_mean": float(np.nanmean(tpr_values)),
            "tpr_at_x_fpr_std": float(np.nanstd(tpr_values)),
            "auc_mean": float(np.nanmean(auc_values)),
            "acc_mean": float(np.nanmean(acc_values)),
            "status": "success",
        }
        all_results.append(result)

        print(
            f"Summary ({len(seeds)} seeds): "
            f"AUC={result['auc_mean']:.4f}, ACC={result['acc_mean']:.4f}, "
            f"TPR@{x_fpr*100:.1f}%FPR={result['tpr_at_x_fpr_mean']:.4f} (+/- {result['tpr_at_x_fpr_std']:.4f}), "
            f"p-value={result['p_value_mean']:.6e} (+/- {result['p_value_std']:.6e})"
        )

    return all_results


def _resolve_m2_dirs(config) -> tuple[str, str]:
    """Derive M2 supervised / unsupervised generated-image dirs from ProjectPaths.

    supervised   -> prompt_set_1 (model trained against these prompts)
    unsupervised -> prompt_set_2 (fresh prompts, same finetuned model)
    """
    layout = ProjectPaths.from_config(config)
    m1 = resolve_m1_model_type(config)
    mi = resolve_mi_model_type(config) or m1
    wm_cfg = config.get("watermark", {}) or {}
    method = wm_cfg.get("method")
    params = resolve_wm_params(config)
    if not (m1 and mi and method and params):
        raise ValueError(
            "supervision config needs m1_model_type, mi_model_type, and a "
            "watermark config with method+slug_keys to derive M2 paths."
        )

    run_id = config.get("run_id")
    if not run_id:
        ft = config.get("finetuning", {}) or {}
        missing = [k for k in ("watermark_fraction", "num_train_epochs", "learning_rate", "effective_batch_size") if ft.get(k) is None]
        if missing:
            raise ValueError(
                f"supervision config: set run_id explicitly or fill finetuning.* "
                f"(missing {missing}) so build_run_id can compute it."
            )
        run_id = build_run_id(
            watermark_fraction=float(ft["watermark_fraction"]),
            epochs=int(ft["num_train_epochs"]),
            learning_rate=float(ft["learning_rate"]),
            effective_batch_size=int(ft["effective_batch_size"]),
            clean_source=str(ft.get("clean_source", "generated")),
        )

    dataset = (config.get("datasets", {}) or {}).get("name") or "coco"
    sup = layout.mi_generated(m1, method, params, run_id, 2, mi, dataset, prompt_set_name(1))
    uns = layout.mi_generated(m1, method, params, run_id, 2, mi, dataset, prompt_set_name(2))
    return str(sup), str(uns)


def main(args):
    config = load_config(args.config_path, watermark_config_path=args.watermark_config)

    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    config_base = dict(config)

    wm_cfg_raw = config_base.get("watermark", {})
    datasets_cfg = config_base.get("datasets", {})
    eval_params = config_base.get("eval_params", {})
    eval_params.setdefault("max_dataset_inference_images", 100)

    num_seeds = int(eval_params.get("num_seeds", args.num_seeds))
    start_seed = int(eval_params.get("start_seed", args.seed_start))
    seeds = [start_seed + i for i in range(num_seeds)]
    degrees = args.degrees if args.degrees else eval_params.get("degrees", DEFAULT_DEGREES)

    print(f"Seeds:   {seeds}")
    print(f"Degrees: {degrees}")

    full_config = OmegaConf.create({
        "watermark": wm_cfg_raw,
        "datasets": datasets_cfg
    })
    watermark, wm_cfg = get_watermark_from_config(full_config)
    wm_name = wm_cfg.get("method")

    m1_sup_dir = datasets_cfg.get("clean_dir")
    m1_uns_dir = datasets_cfg.get("clean_unsupervised_dir")

    if not m1_sup_dir or not m1_uns_dir:
        raise ValueError("Both clean_dir and clean_unsupervised_dir must be set in datasets")

    m2_sup_dir, m2_uns_dir = _resolve_m2_dirs(config)

    print(f"\nDirectories:")
    print(f"  M2 supervised   : {m2_sup_dir}")
    print(f"  M2 unsupervised : {m2_uns_dir}")
    print(f"  M1 supervised   : {m1_sup_dir}")
    print(f"  M1 unsupervised : {m1_uns_dir}")

    all_results = run_degree_evaluation(
        config_base=config_base,
        watermark=watermark,
        wm_cfg=wm_cfg,
        m2_sup_dir=m2_sup_dir,
        m2_uns_dir=m2_uns_dir,
        m1_sup_dir=m1_sup_dir,
        m1_uns_dir=m1_uns_dir,
        eval_params=eval_params,
        seeds=seeds,
        degrees=degrees,
    )

    print(f"\n{'='*70}")
    print("SUPERVISION DEGREE ABLATION — SUMMARY")
    print(f"{'='*70}")
    print(f"{'d':<8} {'n_sup':<8} {'n_uns':<8} {'AUC':<10} {'ACC':<10} {'TPR(mean)':<12} {'TPR(std)':<12} {'p(mean)':<14} {'p(std)':<14}")
    print("-" * 100)
    for r in all_results:
        print(
            f"{r['supervision_degree']:<8.2f} "
            f"{r['n_sup']:<8} "
            f"{r['n_uns']:<8} "
            f"{r['auc_mean']:<10.4f} "
            f"{r['acc_mean']:<10.4f} "
            f"{r['tpr_at_x_fpr_mean']:<12.4f} "
            f"{r['tpr_at_x_fpr_std']:<12.4f} "
            f"{r['p_value_mean']:<14.6e} "
            f"{r['p_value_std']:<14.6e}"
        )

    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_eval = int(eval_params.get("max_dataset_inference_images", 100))
        parts = [timestamp, wm_name, f"p{n_eval}"]
        if args.output_addition:
            safe_add = str(args.output_addition).strip('/\\')
            if safe_add:
                parts.append(safe_add)
        fname = "_".join(parts) + ".json"
        output_file = os.path.join(output_dir, fname)

        summary = {
            "config_path": os.path.abspath(args.config_path),
            "timestamp": timestamp,
            "watermark_name": wm_name,
            "fixed_percentage": FIXED_PERCENTAGE,
            "supervision_degrees": list(degrees),
            "seeds": list(seeds),
            "results": json_safe(all_results),
        }

        with open(output_file, "w") as fp:
            json.dump(summary, fp, indent=2)
        print(f"\nFull results saved to: {output_file}")

    print("\n" + json.dumps(json_safe(all_results), indent=2))
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervision-degree ablation for watermark radioactivity")
    parser.add_argument("--config_path", required=True)
    add_watermark_config_arg(parser)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_addition", default=None)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--degrees", type=float, nargs="+", default=None)
    args = parser.parse_args()
    main(args)
