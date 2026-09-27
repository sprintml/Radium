import argparse
import glob
import json
import logging
import os
from pathlib import Path

from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from pipeline.stage import save_stage_config_snapshot
from src.utils import compose_generation_config, load_config, optim_utils
from src.utils.io_utils import save_images
from math import ceil

logger = logging.getLogger(__name__)

def resolve_generation_paths(config) -> str | None:
    """Resolve generation output directory, auto-deriving from ProjectPaths when possible.

    Resolution order:
    1. ``datasets.output_dir`` (explicit — always wins)
    2. ProjectPaths derivation from ``paths.out_base`` / ``RADIOACTIVITY_OUT``
       when explicit model and watermark identifiers are present in the config.

    Returns None if neither source can supply a path.
    """
    datasets_cfg = config.get("datasets", {}) or {}
    explicit = datasets_cfg.get("output_dir")
    if explicit:
        return str(explicit)

    try:
        from src.paths import ProjectPaths, prompt_set_name, resolve_m1_model_type, resolve_wm_params
        layout = ProjectPaths.from_config(config)
    except (ValueError, ImportError):
        return None

    wm_cfg = config.get("watermark", {}) or {}
    wm_method = wm_cfg.get("method")
    wm_params = resolve_wm_params(config)
    m1_model_type = resolve_m1_model_type(config)
    dataset = datasets_cfg.get("name") or "coco"
    iteration = int(wm_cfg.get("iteration", 0) or 0)
    set_name = datasets_cfg.get("set_name") or prompt_set_name(iteration)
    wm_mode = wm_cfg.get("mode", "generate")

    if not (wm_method and wm_params and m1_model_type):
        return None

    if wm_mode == "clean_generate":
        return str(layout.clean_generated(m1_model_type, dataset, set_name))
    return str(layout.m1_generated(m1_model_type, wm_method, wm_params, dataset, set_name))


def is_generation_complete(config) -> bool:
    """Return True if metadata.jsonl exists and contains the expected number of entries.

    This is a fast completion check suitable for pipeline planning — it does
    not verify that every image file is present on disk.  Use
    verify_generation_integrity() for a full file-by-file audit.
    """
    out_dir = resolve_generation_paths(config)
    if not out_dir:
        return False
    jsonl = Path(out_dir) / "metadata.jsonl"
    if not jsonl.exists():
        return False
    try:
        dataset_params = config.get("dataset_params", {}) or {}
        num_images = int(dataset_params.get("num_images"))
        line_count = sum(1 for l in jsonl.read_text().splitlines() if l.strip())
        return line_count >= num_images
    except Exception:
        return False




def post_process_watermark(config, watermark):
    """Apply a post-generation watermark to a directory of clean images.

    Reads images from ``datasets.input_dir`` (typically M1 clean prompt_set_1),
    writes watermarked copies to ``datasets.output_dir`` preserving the source
    filenames so the index-based pairing convention still holds. Source
    ``metadata.jsonl`` (if present) is propagated verbatim.
    """
    if is_generation_complete(config):
        logger.info("Generation already complete - skipping.")
        return

    input_dir = config.get("datasets", {}).get("input_dir")
    if input_dir is None:
        raise ValueError("config.datasets.input_dir must be set")

    patterns = ["*.png", "*.jpg", "*.jpeg"]
    img_files = []
    for p in patterns:
        img_files.extend(glob.glob(os.path.join(input_dir, p)))
    if not img_files:
        raise FileNotFoundError(f"No images found in {input_dir}")
    img_files = sorted(img_files)

    out_dir = resolve_generation_paths(config)
    if not out_dir:
        raise ValueError(
            "config.datasets.output_dir must be set (or provide paths.out_base / RADIOACTIVITY_OUT plus "
            "resolvable model/watermark params)"
        )
    os.makedirs(out_dir, exist_ok=True)
    logger.info(
        "Applying post-generation watermark to %d images from %s -> %s",
        len(img_files), input_dir, out_dir,
    )

    src_metadata_path = os.path.join(input_dir, "metadata.jsonl")
    prompt_by_fname: dict[str, str] = {}
    if os.path.exists(src_metadata_path):
        with open(src_metadata_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                prompt_by_fname[entry.get("file_name", "")] = entry.get("text", "")

    out_jsonl = os.path.join(out_dir, "metadata.jsonl")
    with open(out_jsonl, "w") as meta:
        for input_img_path in tqdm(img_files, "Processing images"):
            img = Image.open(input_img_path).convert("RGB")
            w_images = watermark.apply(img)
            fname = os.path.basename(input_img_path)
            out_fname = os.path.join(out_dir, fname)
            save_images(w_images, [out_fname])
            meta.write(json.dumps({"file_name": fname, "text": prompt_by_fname.get(fname, "")}) + "\n")
    logger.info("Post-generation wrote metadata to %s", out_jsonl)
    _assert_generation_output(out_dir, expected=len(img_files))


def _assert_generation_output(out_dir: str, *, expected: int) -> None:
    """Fail loudly if the generation stage exits with no/incomplete output.

    Without this, a script that finishes its loop but failed to actually
    write images would exit 0 and downstream slurm afterok dependents would
    happily launch onto a missing or partial directory. We require both
    metadata.jsonl line count and on-disk image count to match ``expected``.
    """
    od = Path(out_dir)
    image_count = sum(1 for p in od.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    jsonl = od / "metadata.jsonl"
    metadata_count = (
        sum(1 for line in jsonl.read_text().splitlines() if line.strip())
        if jsonl.exists() else 0
    )
    if image_count < expected or metadata_count < expected:
        raise RuntimeError(
            f"Generation finished but output is incomplete at {out_dir}: "
            f"{image_count} image(s), {metadata_count} metadata line(s); expected {expected}. "
            f"Refusing to exit 0 — downstream stages would crash on missing inputs."
        )

def generation_loop(config, generator):
    """Run the generation loop using any object exposing ``.generate(prompts)``.

    Both ``GenerationWatermark`` (for ``mode=generate``) and ``CleanGenerator``
    (for ``mode=clean_generate``) implement this contract — the loop itself is
    architecture- and watermark-agnostic.
    """

    if is_generation_complete(config):
        logger.info("Generation already complete - skipping.")
        return

    dataset, prompt_key = optim_utils.get_dataset(config)
    dataset_params = config.get("dataset_params", {}) or {}

    num_images = int(dataset_params["num_images"])
    # Use iteration from the watermark instance (set in main via config.watermark.iteration).
    # prompt_set_N contains the N-th 5k prompt slice, 1-indexed:
    #   iteration=1 → prompts [0, num_images), iteration=2 → [num_images, 2*num_images), ...
    iteration = int(getattr(config.watermark, "iteration", 1) or 1)
    if iteration < 1:
        raise ValueError(f"watermark.iteration must be >= 1, got {iteration}")
    start_idx = (iteration - 1) * num_images
    end_idx = iteration * num_images

    # Ensure we don't go out of bounds of the dataset
    dataset_len = len(dataset)
    if start_idx >= dataset_len:
        raise ValueError(f"iteration {iteration} out of range: start index {start_idx} >= dataset length {dataset_len}")
    end_idx = min(end_idx, dataset_len)

    prompts = [dataset[i][prompt_key] for i in range(start_idx, end_idx)]

    batch_size = int(dataset_params["batch_size"])
    # Start image indexing at the global start index so filenames align with dataset indices
    img_idx = start_idx

    out_dir = resolve_generation_paths(config)
    if not out_dir:
        raise ValueError(
            "config.datasets.output_dir must be set "
            "(or provide paths.out_base / RADIOACTIVITY_OUT plus resolvable model/watermark params)"
        )
    os.makedirs(out_dir, exist_ok=True)

    actual_count = end_idx - start_idx
    num_batches = ceil(len(prompts) / batch_size)
    jsonl_path = os.path.join(out_dir, "metadata.jsonl")
    logger.info("Using iteration: %d", iteration)
    logger.info("Output directory: %s", out_dir)
    logger.info(
        "Generating %d images with IDs %d-%d in %d batch(es)",
        actual_count,
        start_idx,
        end_idx - 1,
        num_batches,
    )

    with open(jsonl_path, "w") as f:
        for b in tqdm(range(num_batches), "Generation Progress"):
            batch_prompts = prompts[b * batch_size : (b + 1) * batch_size]

            images = generator.generate(batch_prompts)  # List[PIL.Image]

            for img, prompt in zip(images, batch_prompts):
                filename = f"img_{img_idx:06d}.png"
                img.save(os.path.join(out_dir, filename))

                metadata_entry = {
                    "file_name": filename,
                    "text": prompt,
                }
                f.write(json.dumps(metadata_entry) + '\n')

                img_idx += 1
    logger.info("Generation wrote metadata to %s", jsonl_path)
    _assert_generation_output(out_dir, expected=actual_count)



def main(config) -> None:
    config = compose_generation_config(config)
    wm_cfg = config.get("watermark", {}) or {}
    wm_type = wm_cfg.get("type")
    wm_name = wm_cfg.get("method")
    wm_mode = wm_cfg.get("mode", "generate")
    logger.info("Watermark backend: %s (%s) mode=%s", wm_name, wm_type, wm_mode)

    # clean_generate is architecture-driven, not watermark-driven: the
    # watermark plays no role and shouldn't be instantiated (avoids loading
    # an SD pipe when the active model is e.g. Infinity).
    if wm_mode == "clean_generate":
        from src.architectures import build_clean_generator
        generator = build_clean_generator(config)
        generation_loop(config, generator)
        return

    from src.watermarks import build_watermark
    watermark, _ = build_watermark(config)

    if wm_type == "post_generation":
        post_process_watermark(config, watermark)
    else:
        generation_loop(config, watermark)

# Utility function to save prompt splits into two JSONL files - P1 AND P2,  so on based on number of splits
def save_prompt_splits(config, split_size=5000, num_splits=8):
    dataset, prompt_key = optim_utils.get_dataset(config)

    out_dir = config.datasets.output_dir
    os.makedirs(out_dir, exist_ok=True)

    total_needed = split_size * num_splits
    if len(dataset) < total_needed:
        raise ValueError(
            f"Dataset too small: need {total_needed} samples, got {len(dataset)}"
        )

    for split_idx in range(num_splits):
        start = split_idx * split_size
        end = start + split_size - 1

        jsonl_path = os.path.join(
            out_dir,
            f"prompts_{start:06d}_{end:06d}.jsonl"
        )

        with open(jsonl_path, "w") as f:
            for idx in range(start, end + 1):
                prompt = dataset[idx][prompt_key]
                entry = {
                    "file_name": f"img_{idx:06d}.png",
                    "text": prompt,
                }
                f.write(json.dumps(entry) + "\n")

        logger.info("Wrote %d prompts -> %s", split_size, jsonl_path)


class GenerationStage:
    name = "generation"

    def load_config(self, config_path: str, watermark_config_path: str | None = None):
        return load_config(config_path, watermark_config_path=watermark_config_path)

    def is_complete(self, config, command: list[str]) -> bool:
        return is_generation_complete(config)


STAGE = GenerationStage()


if __name__ == "__main__":
    from src.utils.commonargs import add_watermark_config_arg, add_model_type_arg
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="diffusion watermark")

    parser.add_argument("--config_path")
    add_watermark_config_arg(parser)
    add_model_type_arg(parser)

    args = parser.parse_args()

    config = load_config(
        args.config_path,
        watermark_config_path=args.watermark_config,
        model_type=args.model_type,
    )
    output_dir = resolve_generation_paths(config)
    logger.info("Loaded config from %s:\n%s", args.config_path, OmegaConf.to_yaml(config))
    logger.info("Resolved output directory: %s", output_dir)
    saved_config = save_stage_config_snapshot(
        config,
        output_dir,
        filename="generation_config.yaml",
        overrides={
            "datasets.output_dir": output_dir,
            "runtime.output_dir": output_dir,
            "runtime.config_path": str(Path(args.config_path).resolve()),
        },
    )
    if saved_config:
        logger.info("Saved config snapshot to %s", saved_config)

    optim_utils.set_random_seed(config.seed)
    logger.info("Random seed: %s", config.seed)
    
    main(config)
    # save_prompt_splits(config)
