from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pipeline.eval_common import (
    load_score_map,
    sample_paired_scores,
    sample_scores,
)


class BaseEvaluationTest(ABC):
    @abstractmethod
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
        """Run the test for one scenario (percentage × mode) over all seeds.

        w_scores_path:   score cache for the Mi (watermarked model) image set.
        clean_scores_path: score cache for the M1 clean image set.
        Returns a result dict with at least {"status": "success", ...}.
        """
        ...

    @staticmethod
    def _sample_scores_for_mode(
        mode: str,
        w_scores_path: Path,
        clean_scores_path: Path,
        seed: int,
        n: int,
    ) -> Tuple[List[float], List[float]]:
        """Draw (w_scores, clean_scores) for one seed.

        - supervised:   paired by image index (Mi and M1 share prompts).
                        Aborts if the two caches don't cover identical indices.
        - unsupervised: independent random samples (distinct RNG streams).
        - baseline:     paired by image index on prompt_set_i; the "watermarked"
                        side here is M1_watermarked (not Mi) vs M1_clean.
        """
        for label, path in (("Mi", w_scores_path), ("M1 clean", clean_scores_path)):
            if not path.exists():
                raise FileNotFoundError(
                    f"{label} score cache missing: {path}. Run detection first."
                )
        w_map = load_score_map(w_scores_path)
        c_map = load_score_map(clean_scores_path)

        if mode in ("supervised", "baseline"):
            return sample_paired_scores(w_map, c_map, seed=seed, n=n)
        if mode == "unsupervised":
            w = sample_scores(w_map, seed=seed, n=n)
            c = sample_scores(c_map, seed=seed + 1_000_003, n=n)
            return w, c
        raise ValueError(f"Unknown evaluation mode '{mode}'")
