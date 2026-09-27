"""Project-wide artifact layout derived from a single output directory.

Set the root via config key `paths.out_base` or env var `RADIOACTIVITY_OUT`.

Layout:
    {out_base}/
      M1_{m1_model_type}/
        clean/{dataset}/{set_name}/
          generated/
          evaluations/
        {watermark_method}/{watermark_params}/
          metadata.jsonl
          {dataset}/{set_name}/
            generated/
            evaluations/
          {mi_model_type}/{run_id}/
            M{i}/
              metadata.jsonl
              final_pipeline/     <- canonical trained model (diffusers pipeline format)
              {dataset}/{set_name}/
                generated/
                evaluations/
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def build_run_id(
    *,
    watermark_fraction: float,
    epochs: int,
    learning_rate: float,
    effective_batch_size: int,
    clean_source: str = "generated",
    suffix: str | None = None,
) -> str:
    """Deterministic run identifier from experiment parameters.

    `effective_batch_size` is `train_batch_size * gradient_accumulation_steps`
    — what actually drives optimization, independent of per-step memory.

    ``suffix`` is appended verbatim (after ``_``) when non-empty. It exists
    so chain hops at the same ``watermark_fraction`` but seeded from
    different source models — e.g. cleani2 chains starting from a 5% vs
    100% M2 — land in disjoint run dirs instead of clobbering each other.

    Example: 0.05, 5, 5e-6, 8, "generated"           -> "5pct_5ep_lr5e-6_eff_bs8"
    Example: 0.05, 5, 5e-6, 8, "coco_val"            -> "5pct_5ep_lr5e-6_eff_bs8_coco_val"
    Example: 1.00, 5, 5e-6, 8, "coco_val"            -> "100pct_5ep_lr5e-6_eff_bs8"
        (clean_source is irrelevant at 100% watermarked, so the suffix is
        omitted to keep both modes pointing at the same artifacts.)
    Example: 0.00, 2, 6e-4, 4, "generated", "from100pct"
                                                     -> "0pct_2ep_lr6e-4_eff_bs4_from100pct"
    """
    pct_value = watermark_fraction * 100
    if 0 < pct_value < 1:
        # Sub-1% fractions can't round-trip through integer percent without
        # collapsing to 0pct_ and colliding with the zero-fraction / cleani2
        # output dir. Render decimals with 'p' instead (0.001 -> 0p1pct).
        pct_token = f"{pct_value:.4f}".rstrip("0").rstrip(".").replace(".", "p") + "pct"
    else:
        pct_token = f"{int(round(pct_value))}pct"
    lr_str = re.sub(r"e([+-])0+(\d)", r"e\1\2", f"{learning_rate:.0e}")
    base = f"{pct_token}_{epochs}ep_lr{lr_str}_eff_bs{effective_batch_size}"
    if clean_source != "generated" and watermark_fraction < 1.0:
        base += f"_{clean_source}"
    if suffix:
        base += f"_{suffix}"
    return base


def prompt_set_name(iteration: int) -> str:
    """Canonical prompt-set name for a given prompt slice index."""
    return f"prompt_set_{int(iteration)}"


def resolve_m1_model_type(config) -> str | None:
    """Return the M1 model type label (e.g. 'sd21') used to root the M1 path tree."""
    return config.get("m1_model_type") if hasattr(config, "get") else None


def resolve_mi_model_type(config) -> str | None:
    """Return the Mi model type label (e.g. 'sd14') used for the Mi path sub-tree.

    Mi can differ from M1 — e.g. training sd14 on sd21-generated watermarked data.
    """
    return config.get("mi_model_type") if hasattr(config, "get") else None


def _slugify_path_part(value) -> str:
    text = str(value).strip().replace(" ", "-")
    text = re.sub(r"[^A-Za-z0-9._=-]+", "-", text)
    return text


def resolve_wm_params(config) -> str | None:
    """Return the watermark params slug, restricted to ``watermark.slug_keys``.

    Each watermark config declares ``watermark.slug_keys: [...]`` — the
    allow-list of param names that meaningfully identify a run on disk.
    Everything else (architecture knobs, model paths, model-config defaults
    that leak in via merge, bit-string payloads) is excluded so paths stay
    short and stable.

    Missing/empty ``slug_keys`` is fatal by design — silently falling back
    to ``"default"`` causes writer/reader stages to land on different
    directories (one with the proper slug, one at ``default/``) and
    downstream stages then crash on missing inputs far from the cause.
    """
    watermark_cfg = config.get("watermark", {}) or {}
    params = watermark_cfg.get("params")
    if not params:
        return None
    try:
        params = dict(params)
    except Exception:
        return _slugify_path_part(params) or None

    slug_keys = watermark_cfg.get("slug_keys")
    if not slug_keys:
        method = watermark_cfg.get("method")
        raise ValueError(
            f"watermark.slug_keys missing/empty for method={method!r}. "
            f"Declare it in configs/watermarks/<method>.yaml as the allow-list "
            f"of params that identify a run on disk."
        )

    parts: list[str] = []
    for key in slug_keys:
        value = params.get(key)
        if value is None or isinstance(value, (dict, list, tuple, set)):
            continue
        key_slug = _slugify_path_part(key)
        value_slug = _slugify_path_part(value)
        if key_slug and value_slug:
            parts.append(f"{key_slug}-{value_slug}")
    if not parts:
        method = watermark_cfg.get("method")
        raise ValueError(
            f"watermark.slug_keys for method={method!r} produced an empty slug — "
            f"none of {list(slug_keys)} resolved to a usable scalar value in "
            f"watermark.params. Check the merged config."
        )
    return "_".join(parts)


class ProjectPaths:
    def __init__(self, base: str | Path):
        self.base = Path(base)

    # --- M1 root ---

    def m1_root(self, m1_model_type: str) -> Path:
        return self.base / f"M1_{m1_model_type}"

    # --- clean (no watermark, M1 model) ---

    def clean_set(self, m1_model_type: str, dataset: str, set_name: str) -> Path:
        return self.m1_root(m1_model_type) / "clean" / dataset / set_name

    def clean_generated(self, m1_model_type: str, dataset: str, set_name: str) -> Path:
        return self.clean_set(m1_model_type, dataset, set_name) / "generated"

    def clean_evaluations(self, m1_model_type: str, dataset: str, set_name: str) -> Path:
        return self.clean_set(m1_model_type, dataset, set_name) / "evaluations"

    # --- M1 watermarked data ---

    def m1_watermark_root(
        self, m1_model_type: str, watermark_method: str, watermark_params: str
    ) -> Path:
        return self.m1_root(m1_model_type) / watermark_method / watermark_params

    def m1_set(
        self,
        m1_model_type: str,
        watermark_method: str,
        watermark_params: str,
        dataset: str,
        set_name: str,
    ) -> Path:
        return self.m1_watermark_root(m1_model_type, watermark_method, watermark_params) / dataset / set_name

    def m1_generated(
        self,
        m1_model_type: str,
        watermark_method: str,
        watermark_params: str,
        dataset: str,
        set_name: str,
    ) -> Path:
        return self.m1_set(m1_model_type, watermark_method, watermark_params, dataset, set_name) / "generated"

    def m1_evaluations(
        self,
        m1_model_type: str,
        watermark_method: str,
        watermark_params: str,
        dataset: str,
        set_name: str,
    ) -> Path:
        return self.m1_set(m1_model_type, watermark_method, watermark_params, dataset, set_name) / "evaluations"

    # --- Mi finetuned models ---

    def mi_root(
        self,
        m1_model_type: str,
        watermark_method: str,
        watermark_params: str,
        run_id: str,
        i: int,
        mi_model_type: str,
    ) -> Path:
        return (
            self.m1_watermark_root(m1_model_type, watermark_method, watermark_params)
            / mi_model_type
            / run_id
            / f"M{i}"
        )

    def mi_model_dir(
        self,
        m1_model_type: str,
        watermark_method: str,
        watermark_params: str,
        run_id: str,
        i: int,
        mi_model_type: str,
    ) -> Path:
        return self.mi_root(m1_model_type, watermark_method, watermark_params, run_id, i, mi_model_type) / "model"

    def mi_set(
        self,
        m1_model_type: str,
        watermark_method: str,
        watermark_params: str,
        run_id: str,
        i: int,
        mi_model_type: str,
        dataset: str,
        set_name: str,
    ) -> Path:
        return (
            self.mi_root(m1_model_type, watermark_method, watermark_params, run_id, i, mi_model_type)
            / dataset
            / set_name
        )

    def mi_generated(
        self,
        m1_model_type: str,
        watermark_method: str,
        watermark_params: str,
        run_id: str,
        i: int,
        mi_model_type: str,
        dataset: str,
        set_name: str,
    ) -> Path:
        return self.mi_set(
            m1_model_type, watermark_method, watermark_params, run_id, i, mi_model_type, dataset, set_name
        ) / "generated"

    def mi_evaluations(
        self,
        m1_model_type: str,
        watermark_method: str,
        watermark_params: str,
        run_id: str,
        i: int,
        mi_model_type: str,
        dataset: str,
        set_name: str,
    ) -> Path:
        return self.mi_set(
            m1_model_type, watermark_method, watermark_params, run_id, i, mi_model_type, dataset, set_name
        ) / "evaluations"

    # --- detection cache helpers ---

    def detection_cache(self, generated_dir: Path, detector_id: str) -> Path:
        return generated_dir / f"{detector_id}.jsonl"

    def has_detection_cache(self, generated_dir: Path, detector_id: str) -> bool:
        return self.detection_cache(generated_dir, detector_id).exists()

    # --- existence checks ---

    def has_m1_data(
        self, m1_model_type: str, watermark_method: str, watermark_params: str
    ) -> bool:
        return self.m1_watermark_root(m1_model_type, watermark_method, watermark_params).exists()

    def has_mi_model_dir(
        self,
        m1_model_type: str,
        watermark_method: str,
        watermark_params: str,
        run_id: str,
        i: int,
        mi_model_type: str,
    ) -> bool:
        return self.mi_model_dir(
            m1_model_type, watermark_method, watermark_params, run_id, i, mi_model_type
        ).exists()

    # --- factory ---

    @classmethod
    def from_config(cls, config) -> "ProjectPaths":
        out_base = None
        try:
            out_base = (config.get("paths", {}) or {}).get("out_base") or None
        except Exception:
            pass
        if not out_base:
            out_base = os.environ.get("RADIOACTIVITY_OUT")
        if not out_base:
            raise ValueError(
                "Project output path not configured. "
                "Set 'paths.out_base' in your config or the RADIOACTIVITY_OUT environment variable."
            )
        return cls(out_base)


def resolve_auto_paths(config) -> dict[str, str | None]:
    """Derive data and output paths from ProjectPaths when not explicitly set in config.

    Reads explicit path labels plus finetuning params to build deterministic paths.
    Only fills keys not already present in the config — explicit values always win.
    """
    try:
        out_layout = ProjectPaths.from_config(config)
    except ValueError:
        return {}

    watermark_cfg = config.get("watermark", {}) or {}
    method = watermark_cfg.get("method")
    if not method:
        return {}

    m1_model_type = resolve_m1_model_type(config)
    if not m1_model_type:
        print("[paths] WARNING: could not resolve m1 model type — cannot derive auto paths.")
        return {}

    mi_model_type = resolve_mi_model_type(config)
    if not mi_model_type:
        print("[paths] WARNING: could not resolve mi model type — cannot derive auto paths.")
        return {}

    watermark_params = resolve_wm_params(config)
    if not watermark_params:
        print("[paths] WARNING: watermark.params is missing — cannot derive auto paths.")
        return {}

    training = config.get("training", {}) or {}
    missing = [k for k in ("watermark_fraction", "num_train_epochs", "learning_rate", "train_batch_size") if training.get(k) is None]
    if missing:
        raise ValueError(
            f"resolve_auto_paths: training is missing required keys {missing}. "
            "Ensure the merged finetuning config (stablediffusion/common.yaml/infinity base + run config) is passed in."
        )
    watermark_fraction = float(training["watermark_fraction"])
    iteration = int(watermark_cfg.get("iteration", 2))

    epochs = int(training["num_train_epochs"])
    lr = float(training["learning_rate"])
    bs = int(training["train_batch_size"])
    ga = int(training.get("gradient_accumulation_steps", 1) or 1)
    eff_bs = bs * ga

    datasets_cfg = config.get("datasets", {}) or {}
    dataset = datasets_cfg.get("name", "coco")
    default_set_name = prompt_set_name(max(1, iteration - 1))
    train_set = datasets_cfg.get("set_name") or default_set_name
    val_set = train_set

    finetune_clean_source = datasets_cfg.get("finetune_clean_source", "generated")
    suffix = training.get("run_id_suffix")
    run_id = build_run_id(
        watermark_fraction=watermark_fraction,
        epochs=epochs,
        learning_rate=lr,
        effective_batch_size=eff_bs,
        clean_source=finetune_clean_source,
        suffix=str(suffix) if suffix else None,
    )
    output_cfg = config.get("output", {}) or {}
    derived: dict[str, str | None] = {}

    if not datasets_cfg.get("validation_data_dir"):
        if finetune_clean_source == "coco_val" and datasets_cfg.get("coco_val_val_dir"):
            derived["validation_data_dir"] = str(datasets_cfg.get("coco_val_val_dir"))
        else:
            derived["validation_data_dir"] = str(out_layout.clean_generated(m1_model_type, dataset, val_set))

    # Two training sources, distinguished by what they contain:
    #   train_data_dir_clean       — unwatermarked images (M1/clean/... or real COCO val)
    #   train_data_dir_watermarked — watermarked images (M1/{wm}/... for M2, or M{i-1}/... for i>=3)
    # The runner decides the mix from training.watermark_fraction alone.
    if iteration >= 3:
        # Mi (i>=3) trains on M(i-1)'s generated prompt_set_{i-1} images, mixed with
        # the M1 clean generation for the same prompt_set. watermark_fraction sets the mix.
        if finetune_clean_source == "generated" and not datasets_cfg.get("train_data_dir_clean"):
            derived["train_data_dir_clean"] = str(out_layout.clean_generated(m1_model_type, dataset, train_set))
        if not datasets_cfg.get("train_data_dir_watermarked"):
            derived["train_data_dir_watermarked"] = str(out_layout.mi_generated(
                m1_model_type, method, watermark_params, run_id, iteration - 1, mi_model_type, dataset, train_set
            ))
    else:
        # M2 trains on a clean source + M1 watermarked; watermark_fraction sets the mix.
        # When finetune_clean_source == "coco_val", the runner uses real COCO val images
        # directly (path comes from datasets.coco_val_image_dir) — don't auto-derive here.
        if finetune_clean_source == "generated" and not datasets_cfg.get("train_data_dir_clean"):
            derived["train_data_dir_clean"] = str(out_layout.clean_generated(m1_model_type, dataset, train_set))
        if not datasets_cfg.get("train_data_dir_watermarked"):
            derived["train_data_dir_watermarked"] = str(
                out_layout.m1_generated(m1_model_type, method, watermark_params, dataset, train_set)
            )

    if not output_cfg.get("output_dir"):
        out = out_layout.mi_root(m1_model_type, method, watermark_params, run_id, iteration, mi_model_type)
        derived["output_dir"] = str(out)

    return derived
