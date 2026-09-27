"""Statistical evaluation: reads cached detection scores and runs e-value / Welch tests.

Expects score files written by pipeline/stages/detection.py under
``<output_dir>/scores/``.  No watermark model or GPU required.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from pipeline.eval_common import json_safe, score_cache_path, summarize_score_cache
from pathlib import Path as _Path
from pipeline.stage import save_stage_config_snapshot
from src.evaluation_tests import REGISTRY
from src.utils import load_config

logger = logging.getLogger(__name__)

def resolve_evaluation_paths(config) -> dict[str, str | None]:
    """Derive evaluation output directory from ProjectPaths when not explicitly set.

    Uses the same layout as detection (mi_evaluations), keyed by M1 root
    model plus Mi target model, so detection and evaluation share one
    output directory.
    """
    from src.paths import (
        ProjectPaths,
        prompt_set_name,
        resolve_m1_model_type,
        resolve_mi_model_type,
        resolve_wm_params,
    )

    wm = (config.get("watermark", {}) or {}) if hasattr(config, "get") else {}
    ds = (config.get("datasets", {}) or {}) if hasattr(config, "get") else {}

    method = wm.get("method")
    params = resolve_wm_params(config)
    m1_model_type = resolve_m1_model_type(config)
    mi_model_type = resolve_mi_model_type(config)
    run_id = wm.get("run_id")
    iteration = int(wm.get("iteration", 1))
    model_iteration = iteration + 1
    dataset = ds.get("name") or "coco"
    set_name = ds.get("set_name") or prompt_set_name(iteration)

    if not all([method, params, m1_model_type, mi_model_type, run_id]):
        return {"output_dir": None}

    try:
        layout = ProjectPaths.from_config(config)
        return {"output_dir": str(layout.mi_evaluations(
            m1_model_type, method, params, run_id, model_iteration, mi_model_type, dataset, set_name
        ))}
    except ValueError:
        return {"output_dir": None}


def is_evaluation_complete(config, output_dir: str) -> bool:
    """Return True if a result file exists that matches the current config's
    detection_mode, covers every configured (percentage, mode) scenario, and
    was produced with at least the configured seed range and the mode-specific
    eval_params that shape the output.

    Rationale: result files for different detection_modes (evalue vs welch),
    seed counts, and eval_params live side-by-side in the same directory.
    Matching on scenarios alone would let an evalue result satisfy a welch run,
    or let a 10-seed result satisfy a 20-seed run. We additionally require:
      * top-level ``mode`` == config ``detection_mode``;
      * ``seeds`` in the result is a superset of the seeds the current config
        would generate (seed_start..seed_start+num_seeds);
      * the mode-specific shaping param embedded in the filename matches
        (``e{num_evalue_images}`` or ``e{n_init}-{n_max}`` for evalue,
        ``p{max_dataset_inference_images}`` for welch) — checked as exact
        underscore-delimited tokens to avoid e.g. ``e500`` matching ``e5000``;
      * for evalue, every successful per-scenario result carries an
        ``evalue_multiseed`` block (rejects pre-refactor files that only
        wrote ``evalue_single_seed``).
    """
    if not output_dir:
        return False
    od = Path(output_dir)
    if not od.exists():
        return False

    config_base = dict(config) if hasattr(config, "items") else {}
    scenarios = config_base.get("evaluation_scenarios", [])
    if not scenarios:
        return False

    expected_detection_mode = config_base.get("detection_mode")
    eval_params = dict(config_base.get("eval_params", {}) or {})
    num_seeds = int(eval_params.get("num_seeds", 10))
    seed_start = int(eval_params.get("start_seed", 42))
    expected_seeds = set(range(seed_start, seed_start + num_seeds))

    extra_tokens: list[str] = []
    skip_token_checks = False
    if expected_detection_mode == "evalue":
        # Evalue filenames are `evalue_ref_<src>_n<n_init>-<n_max>.json` —
        # one canonical file per (run_dir, clean_reference_source, batch/nmax
        # pair); re-runs with the same n_init/n_max overwrite, while runs with
        # different sampling settings coexist.
        ref_src = config_base.get("clean_reference_source") or "mi_0pct"
        n_init = int(eval_params.get("num_evalue_images", 200))
        n_max = int(eval_params.get("max_evalue_images", n_init))
        shaping_token = ""
        glob_pattern = f"evalue_ref_{ref_src}_n{n_init}-{n_max}.json"
        required_result_key = "evalue_multiseed"
        skip_token_checks = True
    elif expected_detection_mode == "welch":
        shaping_token = f"p{int(eval_params.get('max_dataset_inference_images', 100))}"
        glob_pattern = "welch_results_*.json"
        required_result_key = None
    elif expected_detection_mode == "aggregation":
        ref_src = config_base.get("clean_reference_source") or "mi_0pct"
        shaping_token = ""
        glob_pattern = f"aggregation_ref_{ref_src}.json"
        required_result_key = "aggregation"
        skip_token_checks = True
    else:
        return False

    result_files = sorted(od.glob(glob_pattern),
                          key=lambda p: p.stat().st_mtime, reverse=True)
    if not result_files:
        return False

    wm_cfg = config_base.get("watermark", {}) or {}
    iteration_now = int((wm_cfg.get("iteration") if hasattr(wm_cfg, "get") else 1) or 1)
    required: set[tuple] = set()
    for sc in scenarios:
        pct = sc.get("percentage")
        for mode in sc.get("modes", []):
            if mode == "baseline" and iteration_now != 1:
                continue
            required.add((pct, mode))

    seeds_token = f"seeds{num_seeds}"
    for result_file in result_files:
        if not skip_token_checks:
            # Filename token check (exact underscore-delimited match) — avoids
            # "e1000" silently matching "e10000" or "seeds5" matching "seeds50".
            name_tokens = result_file.stem.split("_")
            if seeds_token not in name_tokens:
                continue
            if shaping_token and shaping_token not in name_tokens:
                continue
            if any(t not in name_tokens for t in extra_tokens):
                continue
        try:
            data = json.loads(result_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("mode") != expected_detection_mode:
            continue
        actual_seeds = set(data.get("seeds") or [])
        if not expected_seeds.issubset(actual_seeds):
            continue
        covered = set()
        for r in data.get("results", []):
            if r.get("status") != "success":
                continue
            if required_result_key and required_result_key not in r:
                continue
            covered.add((r.get("percentage"), r.get("mode")))
        if required.issubset(covered):
            return True

    return False


def _resolve_score_paths_for_mode(config, mode: str) -> tuple[_Path, _Path]:
    """Return (w_scores_path, clean_scores_path) for the given evaluation mode.

    - supervised:   Mi[prompt_set_i]   vs M1_clean[prompt_set_i]
    - unsupervised: Mi[prompt_set_i+1] vs M1_clean[prompt_set_i+1]
    - baseline:     M1_watermarked[prompt_set_i] vs M1_clean[prompt_set_i]
                    (sanity check: confirms the watermark is detectable in M1's
                    own outputs before asking whether it survives finetuning).
    """
    from src.paths import (
        ProjectPaths,
        prompt_set_name,
        resolve_m1_model_type,
        resolve_mi_model_type,
        resolve_wm_params,
    )

    wm = (config.get("watermark", {}) or {}) if hasattr(config, "get") else {}
    ds = (config.get("datasets", {}) or {}) if hasattr(config, "get") else {}

    method = wm.get("method")
    params = resolve_wm_params(config)
    m1_model_type = resolve_m1_model_type(config)
    mi_model_type = resolve_mi_model_type(config)
    run_id = wm.get("run_id")
    iteration = int(wm.get("iteration", 1))
    model_iteration = iteration + 1
    dataset = ds.get("name") or "coco"
    wm_name = method or "unknown"

    if mode == "supervised":
        set_name = prompt_set_name(iteration)
    elif mode == "unsupervised":
        set_name = prompt_set_name(iteration + 1)
    elif mode == "baseline":
        set_name = prompt_set_name(iteration)
    else:
        raise ValueError(f"Unknown evaluation mode '{mode}'")

    layout = ProjectPaths.from_config(config)

    if mode == "baseline":
        m1_wm_eval_dir = layout.m1_evaluations(m1_model_type, method, params, dataset, set_name)
        w_scores_path = score_cache_path(str(m1_wm_eval_dir / "scores"), wm_name)
    else:
        mi_eval_dir = layout.mi_evaluations(
            m1_model_type, method, params, run_id, model_iteration, mi_model_type, dataset, set_name
        )
        w_scores_path = score_cache_path(str(mi_eval_dir / "scores"), wm_name)

    clean_ref_flag = (
        (config.get("clean_reference_source") if hasattr(config, "get") else None) or "mi_0pct"
    )
    if mode == "baseline":
        # Baseline is M1-watermarked vs M1-clean by definition; ignore the flag.
        clean_eval_dir = layout.clean_evaluations(m1_model_type, dataset, set_name)
    elif clean_ref_flag == "m1_clean":
        clean_eval_dir = layout.clean_evaluations(m1_model_type, dataset, set_name)
    elif clean_ref_flag == "mi_arch_clean":
        if not mi_model_type:
            raise ValueError(
                "clean_reference_source='mi_arch_clean' requires mi_model_type to be set in the config."
            )
        clean_eval_dir = layout.clean_evaluations(mi_model_type, dataset, set_name)
    elif clean_ref_flag == "mi_0pct":
        if not all([mi_model_type, run_id, method, params]):
            raise ValueError(
                "clean_reference_source='mi_0pct' requires mi_model_type, watermark.run_id, "
                "watermark.method, and watermark.params to be set in the config."
            )
        run_id_0pct, n_subs = re.subn(r"^\d+pct", "0pct", run_id)
        if n_subs == 0:
            raise ValueError(
                f"clean_reference_source='mi_0pct': run_id {run_id!r} does not start with "
                f"'{{N}}pct_' — cannot derive 0% sibling run_id."
            )
        if run_id_0pct == run_id:
            raise ValueError(
                f"clean_reference_source='mi_0pct': run_id is already at 0pct ({run_id!r}); "
                f"watermarked and clean-reference scores would point at the same path."
            )
        clean_eval_dir = layout.mi_evaluations(
            m1_model_type, method, params, run_id_0pct, model_iteration, mi_model_type, dataset, set_name
        )
    else:
        raise ValueError(
            f"clean_reference_source must be one of 'm1_clean', 'mi_arch_clean', 'mi_0pct'; "
            f"got {clean_ref_flag!r}"
        )
    clean_scores_path = score_cache_path(str(clean_eval_dir / "scores"), wm_name)

    return w_scores_path, clean_scores_path


def _log_score_cache_summary(cache_path: Path, label: str) -> None:
    summary = summarize_score_cache(cache_path)
    logger.info(
        "%s - score cache: %s (%d scored images)",
        label,
        summary["path"],
        summary["num_scores"],
    )


def _log_result_summary(detection_mode: str, percentage: int, mode: str, result: dict, eval_params: dict) -> None:
    if result.get("status") != "success":
        logger.warning(
            "Scenario %s%% %s finished with status=%s",
            percentage,
            mode,
            result.get("status"),
        )
        return

    if detection_mode == "evalue":
        multi = result.get("evalue_multiseed", {})
        logger.info(
            "Scenario %s%% %s result - e-value=%s, TPR=%.3f, stopping_time=%s",
            percentage,
            mode,
            f"{float(multi.get('e_value', float('nan'))):.3g}",
            float(multi.get("tpr", 0.0)),
            multi.get("stopping_time"),
        )
    elif detection_mode == "aggregation":
        agg = result.get("aggregation", {})
        spaces = agg.get("spaces", []) or []
        has_ref = bool(agg.get("has_clean_reference"))
        for s in spaces:
            p_ref = s.get("mean_p_clean_ref") if has_ref else None
            p_ref_str = f", p_clean_ref={p_ref:.3g}" if p_ref is not None else ""
            logger.info(
                "Scenario %s%% %s [%s] - bit_acc=%.4f, p_binomial=%.3g%s (n=%d, K=%d)",
                percentage,
                mode,
                s.get("space"),
                float(s.get("mean_bit_acc", float("nan"))),
                float(s.get("mean_p_binomial", float("nan"))),
                p_ref_str,
                int(agg.get("n_aggregated", 0)),
                int(agg.get("K", 0)),
            )
    else:
        x_fpr = float(eval_params.get("x_fpr", 0.01)) * 100.0
        logger.info(
            "Scenario %s%% %s result - p-value=%s, TPR@%.0f%%FPR=%.3f",
            percentage,
            mode,
            f"{float(result.get('p_value_mean', float('nan'))):.3g}",
            x_fpr,
            float(result.get("tpr_at_x_fpr_mean", 0.0)),
        )


def run_evaluation(args, detection_mode: str) -> list:
    if detection_mode not in REGISTRY:
        raise ValueError(f"Unknown detection_mode '{detection_mode}'. Known: {list(REGISTRY)}")

    config = load_config(
        args.config_path,
        watermark_config_path=getattr(args, "watermark_config", None),
        model_type=getattr(args, "model_type", None),
    )
    logger.info("Config loaded from %s:\n%s", args.config_path, OmegaConf.to_yaml(config))

    output_dir = args.output_dir or resolve_evaluation_paths(config).get("output_dir")
    logger.info("Output directory: %s", output_dir)
    saved_config = save_stage_config_snapshot(
        config,
        output_dir,
        filename="evaluation_config.yaml",
        overrides={
            "runtime.output_dir": output_dir,
            "runtime.config_path": str(Path(args.config_path).resolve()),
            "runtime.detection_mode": detection_mode,
        },
    )
    if saved_config:
        logger.info("Saved config snapshot to %s", saved_config)

    force = bool(getattr(args, "force", False))

    if force and output_dir and Path(output_dir).exists():
        force_patterns = {
            "evalue": ("evalue_results_*.json", "evalue_ref_*.json", "evalue_ref_*_n*-*.json"),
            "welch": ("welch_results_*.json",),
            "aggregation": ("aggregation_results_*.json", "aggregation_ref_*.json"),
        }.get(detection_mode, ())
        removed = 0
        for pattern in force_patterns:
            for stale in Path(output_dir).glob(pattern):
                stale.unlink()
                removed += 1
        if removed:
            logger.info("--force: deleted %d existing %s result file(s) in %s",
                        removed, detection_mode, output_dir)

    if not force and is_evaluation_complete(config, output_dir):
        logger.info("Evaluation already complete - skipping.")
        return []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    config_base = dict(config)
    wm_base_cfg = config_base.get("watermark", {})
    wm_name = (wm_base_cfg.get("method") or wm_base_cfg.get("name") or "unknown") if hasattr(wm_base_cfg, "get") else "unknown"
    eval_params = dict(config_base.get("eval_params", {}) or {})

    if detection_mode == "evalue":
        eval_params.setdefault("num_evalue_images", 200)
        # Adaptive sampling: max_evalue_images > num_evalue_images enables it,
        # equality (the default) keeps the original fixed-n behavior.
        eval_params.setdefault("max_evalue_images", eval_params["num_evalue_images"])
        eval_params.setdefault("continue_threshold", 0.5)
    elif detection_mode == "aggregation":
        eval_params.setdefault("aggregation_n", 0)        # 0 = use all available
        eval_params.setdefault("aggregation_bootstrap_B", 10_000)
        eval_params.setdefault("alpha", 0.05)
    else:
        eval_params.setdefault("max_dataset_inference_images", 100)

    eval_params.setdefault("num_seeds", 10)
    eval_params.setdefault("start_seed", 42)

    seed_start = args.seed_start if args.seed_start is not None else int(eval_params["start_seed"])
    num_seeds = args.num_seeds if args.num_seeds is not None else int(eval_params["num_seeds"])
    seeds = [seed_start + i for i in range(num_seeds)]

    if output_dir:
        eval_params["output_dir"] = output_dir
        eval_params["output_addition"] = getattr(args, "output_addition", None)

    test = REGISTRY[detection_mode]()
    all_results = []
    iteration_now = int((wm_base_cfg.get("iteration") if hasattr(wm_base_cfg, "get") else 1) or 1)
    for scenario in config_base.get("evaluation_scenarios", []):
        percentage = scenario.get("percentage")
        for mode in scenario.get("modes", []):
            if mode == "baseline" and iteration_now != 1:
                logger.info(
                    "Skipping baseline scenario at iteration=%d (only meaningful at i=1).",
                    iteration_now,
                )
                continue
            w_scores_path, clean_scores_path = _resolve_score_paths_for_mode(config, mode)
            _log_score_cache_summary(w_scores_path, f"Scenario {percentage}% {mode} [Mi]")
            _log_score_cache_summary(clean_scores_path, f"Scenario {percentage}% {mode} [M1]")
            result = test.run_scenario(
                wm_cfg=wm_base_cfg,
                percentage=percentage,
                mode=mode,
                eval_params=eval_params,
                seeds=seeds,
                w_scores_path=w_scores_path,
                clean_scores_path=clean_scores_path,
                output_dir=output_dir,
            )
            _log_result_summary(detection_mode, percentage, mode, result, eval_params)
            all_results.append(result)

    successful = [r for r in all_results if r.get("status") == "success"]
    failed = [r for r in all_results if r.get("status") == "failed"]
    logger.info(
        f"Evaluation complete: {len(successful)} successful, {len(failed)} failed "
        f"across {len(all_results)} scenario(s)."
    )

    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if detection_mode == "evalue":
            ref_src = config_base.get("clean_reference_source") or "mi_0pct"
            n_init = int(eval_params.get("num_evalue_images", 200))
            n_max = int(eval_params.get("max_evalue_images", n_init))
            # Canonical name encodes the (ref_src, n_init, n_max) tuple so
            # different sampling settings coexist; same-tuple re-runs overwrite.
            output_file = os.path.join(
                output_dir, f"evalue_ref_{ref_src}_n{n_init}-{n_max}.json"
            )
            parts = None
        elif detection_mode == "aggregation":
            ref_src = config_base.get("clean_reference_source") or "mi_0pct"
            output_file = os.path.join(output_dir, f"aggregation_ref_{ref_src}.json")
            parts = None
        else:
            parts = ["welch_results", wm_name, f"seeds{num_seeds}",
                     f"p{eval_params.get('max_dataset_inference_images', 100)}", timestamp]
        if parts is not None:
            add = getattr(args, "output_addition", None)
            if add:
                safe = str(add).strip("/\\")
                if safe:
                    parts.append(safe)
            output_file = os.path.join(output_dir, "_".join(parts) + ".json")
        summary = {
            "config_path": os.path.abspath(args.config_path),
            "timestamp": timestamp,
            "watermark_name": wm_name,
            "mode": detection_mode,
            "seeds": seeds,
            "total_evaluations": len(all_results),
            "successful": len(successful),
            "failed": len(failed),
            "results": json_safe(all_results),
        }
        if detection_mode == "evalue":
            summary["reference"] = (
                "Arithmetic mean e-value merging: Vovk & Wang (2021), Ann. Statist. 49(3), Prop 3.1"
            )
        with open(output_file, "w") as fp:
            json.dump(summary, fp, indent=2)
        logger.info("Results saved to: %s", output_file)

    return all_results


class EvaluationStage:
    name = "evaluation"

    def load_config(self, config_path: str, watermark_config_path: str | None = None):
        return load_config(config_path, watermark_config_path=watermark_config_path)

    def is_complete(self, config, command: list[str]) -> bool:
        from pipeline.stage import extract_arg
        output_dir = extract_arg(command, "--output_dir") or resolve_evaluation_paths(config).get("output_dir")
        if not output_dir:
            return False
        return is_evaluation_complete(config, output_dir)


STAGE = EvaluationStage()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from src.utils.commonargs import add_watermark_config_arg, add_model_type_arg
    parser = argparse.ArgumentParser(description="Statistical evaluation from cached scores")
    parser.add_argument("--detection_mode", choices=("evalue", "welch", "aggregation"), default=None)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir", default=None, help="Omit to auto-resolve from config via ProjectPaths")
    parser.add_argument("--output_addition", default=None)
    parser.add_argument("--num_seeds", type=int, default=None)
    parser.add_argument("--seed_start", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing *_results_*.json in the output dir and re-run, "
        "ignoring is_evaluation_complete.",
    )
    add_watermark_config_arg(parser)
    add_model_type_arg(parser)
    args = parser.parse_args()
    config = load_config(
        args.config_path,
        watermark_config_path=args.watermark_config,
        model_type=args.model_type,
    )
    detection_mode = args.detection_mode or config.get("detection_mode")
    if not detection_mode:
        parser.error("detection_mode must be set via --detection_mode or config key 'detection_mode'")
    run_evaluation(args, detection_mode=detection_mode)
