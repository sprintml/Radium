"""Disjoint image-id partitioning for clean/watermarked finetuning mixes.

Both clean (M1-generated) and watermarked stages produce filenames of the form
``img_{idx:06d}.png`` where ``idx`` is the dataset row index. Without
intervention the same idx appears in both folders, so a mixed manifest can show
the model clean+watermarked variants of the same prompt. This helper splits the
shared id space into two disjoint slices sized by ``watermark_fraction``.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_ID_RE = re.compile(r"^img_(\d+)")


def _index_by_id(directory: str | Path) -> dict[int, Path]:
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    out: dict[int, Path] = {}
    for f in sorted(path.iterdir()):
        if not (f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS):
            continue
        match = _ID_RE.match(f.name)
        if not match:
            raise ValueError(f"Cannot parse image id from filename: {f.name}")
        out.setdefault(int(match.group(1)), f)
    return out


def partition_by_image_id(
    clean_dir: str | Path,
    watermarked_dir: str | Path,
    *,
    total: int,
    watermark_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Return (clean_files, watermarked_files) with disjoint image ids.

    Watermarked ids are sampled first (sized by ``watermark_fraction * total``);
    the remaining clean budget is filled from clean ids that didn't land in the
    watermarked slice. Sampling is deterministic in ``seed``.
    """
    if total <= 0:
        return [], []

    watermarked_count = int(round(watermark_fraction * total))
    clean_count = total - watermarked_count

    clean_by_id = _index_by_id(clean_dir)
    watermarked_by_id = _index_by_id(watermarked_dir)

    rng = random.Random(seed)

    watermarked_ids = list(watermarked_by_id)
    rng.shuffle(watermarked_ids)
    if len(watermarked_ids) < watermarked_count:
        raise ValueError(
            f"Not enough watermarked images: need {watermarked_count}, have {len(watermarked_ids)}."
        )
    chosen_watermarked = watermarked_ids[:watermarked_count]

    used = set(chosen_watermarked)
    clean_ids = [i for i in clean_by_id if i not in used]
    rng.shuffle(clean_ids)
    if len(clean_ids) < clean_count:
        raise ValueError(
            f"Not enough clean images disjoint from watermarked ids: "
            f"need {clean_count}, have {len(clean_ids)}."
        )
    chosen_clean = clean_ids[:clean_count]

    return (
        [clean_by_id[i] for i in chosen_clean],
        [watermarked_by_id[i] for i in chosen_watermarked],
    )


__all__ = ["partition_by_image_id"]
