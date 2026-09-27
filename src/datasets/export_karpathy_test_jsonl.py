#!/usr/bin/env python3

from pathlib import Path
import json
import random
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Export COCO Karpathy test split to Infinity JSONL"
    )
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--num_images", type=int, default=5000)
    parser.add_argument("--h_div_w", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_path = Path(args.out)

    # --------------------------------------------------
    # 1. Load captions (YOUR JSON FORMAT)
    # --------------------------------------------------
    with open(args.metadata, "r", encoding="utf-8") as f:
        data = json.load(f)

    caption_map = {}
    for entry in data:
        img_path = entry.get("image")
        captions = entry.get("caption")

        if not img_path or not captions:
            continue

        fname = Path(img_path).name   # strip "val2014/"
        caption_map[fname] = captions[0].strip()

    print(f"Loaded captions for {len(caption_map)} images")

    # --------------------------------------------------
    # 2. Resolve image paths
    # --------------------------------------------------
    image_paths = []
    for fname in caption_map:
        p = images_dir / fname
        if p.exists():
            image_paths.append(p)

    print(f"Found {len(image_paths)} images with captions on disk")

    if len(image_paths) < args.num_images:
        raise RuntimeError(
            f"Not enough captioned images ({len(image_paths)}) "
            f"to sample {args.num_images}"
        )

    # --------------------------------------------------
    # 3. Deterministic sampling
    # --------------------------------------------------
    rng = random.Random(args.seed)
    rng.shuffle(image_paths)
    image_paths = image_paths[: args.num_images]

    # --------------------------------------------------
    # 4. Write Infinity JSONL
    # --------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for p in image_paths:
            caption = caption_map[p.name]

            record = {
                "image_path": str(p.resolve()),
                "h_div_w": args.h_div_w,
                "long_caption": caption,
                "long_caption_type": "MSCOCO",
                "text": caption,
                "short_caption_type": "user",
            }

            f.write(json.dumps(record) + "\n")

    print(f"  Export complete")
    print(f"  Output: {out_path}")
    print(f"  Samples written: {len(image_paths)}")


if __name__ == "__main__":
    main()

