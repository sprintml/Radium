"""Dataset utilities for loading images from two folders.

Provides DualFolderDataset: given two folders, sample p1 * num_images
from folder1 and p2 * num_images from folder2 and expose a standard
torch Dataset interface. Can return PIL.Image or torch.Tensor in [-1, 1].
"""

from pathlib import Path
import json
import random
from typing import Callable, List, Optional, Tuple, Union

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def _is_image_file(p: Path) -> bool:
	return p.suffix.lower() in {
		".jpg",
		".jpeg",
		".png",
	}


def load_coco_val_as_metadata(annotation_path: str, image_dir: str, seed: int = 0) -> dict:
	"""Randomly pair COCO val captions with images in image_dir.

	Precise image-ID matching is not needed here: these COCO val images are
	used alongside watermarked images generated from test prompts, so the two
	prompt sets are disjoint by construction. Any val caption paired with any
	val image is valid for finetuning.
	"""
	annotation_path = Path(annotation_path)
	image_dir = Path(image_dir)

	with annotation_path.open() as f:
		data = json.load(f)

	# One caption per image_id to avoid over-representing scenes with many captions.
	seen: set = set()
	captions: list = []
	for item in data:
		img_id = item.get("image_id")
		caption = item.get("caption")
		if caption and img_id not in seen:
			seen.add(img_id)
			captions.append(caption)

	image_files = sorted(p for p in image_dir.iterdir() if _is_image_file(p))

	rng = random.Random(seed)
	rng.shuffle(captions)

	metadata: dict = {}
	for img_file, caption in zip(image_files, captions):
		metadata[str(img_file.resolve())] = caption
		metadata[img_file.name] = caption
		metadata[img_file.stem] = caption

	return metadata


class DualFolderDataset(Dataset):
	"""Dataset combining images from two folders.

	Parameters
	- folder1, folder2: path-like to two directories containing images
	- p1, p2: proportions of images to take from folder1 and folder2.
	  They do not have to sum to 1; counts are computed as round(p1 * num_images)
	  and round(p2 * num_images). If the sum of counts is not equal to
	  num_images, we truncate/pad from folder1 to match `num_images`.
	- num_images: total number of images returned by the dataset
	- return_type: either 'pil' or 'tensor'. If 'tensor', output will be
	  a torch.FloatTensor with values in [-1, 1].
	- transform: optional torchvision-style transform applied to PIL image
	  before conversion to tensor (if any).
	- with_labels: whether to return (image, label) where label is 0 for
	  folder1 and 1 for folder2. If False, only the image is returned.
	- seed: optional random seed for deterministic sampling.

	Behavior notes:
	- If a requested count for a folder exceeds the available images, we
	  sample with replacement from that folder.
	"""

	def __init__(
		self,
		folder1: Union[str, Path],
		folder2: Optional[Union[str, Path]] = None,
		p1: float = 0.5,
		p2: float = 0.5,
		num_images: int = 5000,
		return_type: str = "tensor",
		transform: Optional[Callable] = None,
		with_labels: bool = False,
		return_prompt: bool = True,
		metadata_path: Optional[Union[str, Path]] = None,
		shuffle: bool = False,
		seed: Optional[int] = None,
		return_path: bool = False,
		metadata_override: Optional[dict] = None,
		files_override: Optional[Tuple[List[Path], List[Path]]] = None,
	):
		self.folder1 = Path(folder1)
		self.folder2 = Path(folder2) if folder2 is not None else None
		self.p1 = float(p1)
		self.p2 = float(p2)
		self.num_images = int(num_images)
		self.return_type = return_type.lower()
		assert self.return_type in {"pil", "tensor"}, "return_type must be 'pil' or 'tensor'"
		self.transform = transform
		self.with_labels = with_labels
		self.return_path = bool(return_path)


		# RNG for deterministic behaviour when requested
		self.shuffle = bool(shuffle)
		self._rng = random.Random(seed) if seed is not None else random.Random(0)

		if files_override is not None:
			# Caller has already chosen the exact files for each side (e.g. from a
			# disjoint image-id partition). Skip iterdir scanning and count slicing.
			files1_o, files2_o = files_override
			self._files1 = list(files1_o)
			self._files2 = list(files2_o)
			if len(self._files1) == 0:
				raise ValueError("files_override provided empty file list for folder1")
			self._samples: List[Tuple[Path, int]] = []
			self._samples.extend([(p, 0) for p in self._files1])
			self._samples.extend([(p, 1) for p in self._files2])
		else:
			# gather files
			self._files1: List[Path] = sorted([p for p in self.folder1.iterdir() if p.is_file() and _is_image_file(p)])
			if folder2 is None:
				self._files2 = []
			else:
				self._files2: List[Path] = sorted([p for p in Path(folder2).iterdir() if p.is_file() and _is_image_file(p)])

			if len(self._files1) == 0:
				raise ValueError(f"No image files found in folder1: {self.folder1}")
			if folder2 is not None and len(self._files2) == 0:
				raise ValueError(f"No image files found in folder2: {folder2}")

			# compute counts
			c1 = int(round(self.p1 * self.num_images))
			c2 = int(round(self.p2 * self.num_images))
			total = c1 + c2
			if total != self.num_images:
				diff = self.num_images - total
				c1 = max(0, c1 + diff)

			# optionally shuffle file ordering deterministically with seed
			files1 = list(self._files1)
			files2 = list(self._files2)
			if self.shuffle:
				self._rng.shuffle(files1)
				if files2:
					self._rng.shuffle(files2)

			def _take_deterministic(files: List[Path], count: int) -> List[Path]:
				if count <= 0:
					return []
				if not files:
					return []
				if count <= len(files):
					return files[:count]
				# repeat deterministically
				out: List[Path] = []
				idx = 0
				while len(out) < count:
					out.append(files[idx % len(files)])
					idx += 1
				return out

			self._samples: List[Tuple[Path, int]] = []  # list of (path, label)

			if not files2:
				# single-folder mode: sample exactly `num_images` items from folder1
				# and assign label 0 for all samples. p2 is ignored in this mode.
				total_count = self.num_images
				g = _take_deterministic(files1, total_count)
				self._samples.extend([(p, 0) for p in g])
			else:
				g1 = _take_deterministic(files1, c1)
				g2 = _take_deterministic(files2, c2)
				self._samples.extend([(p, 0) for p in g1])
				self._samples.extend([(p, 1) for p in g2])

		if self.shuffle:
			self._rng.shuffle(self._samples)


		# Prepare default tensor conversion if asked for tensors and no transform
		if self.return_type == "tensor":
			self._to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)])
		else:
			self._to_tensor = None

		# prompt handling
		self.return_prompt = bool(return_prompt)
		self.metadata: dict = {}
		if metadata_override is not None:
			self.metadata = dict(metadata_override)
		elif self.return_prompt:
			# If a metadata_path is provided, load it (if it's a dir, search for metadata file inside)
			if metadata_path is not None:
				mp = Path(metadata_path)
				if mp.is_dir():
					# prefer .jsonl then .json
					candidates = [mp / "metadata.jsonl", mp / "metadata.json"]
					for c in candidates:
						if c.exists():
							self.metadata.update(self._load_metadata(c, base_folder=mp))
							break
				else:
					self.metadata.update(self._load_metadata(mp, base_folder=mp.parent))
			# search for metadata files inside the provided folders (folder1 and folder2)
			# This allows disambiguation when files have the same name across folders
			for fld in [self.folder1, getattr(self, "folder2", None)]:
				if not fld:
					continue
				for name in ("metadata.jsonl", "metadata.json", "manifest.jsonl", "manifest.json"):
					candidate = fld / name
					if candidate.exists():
						self.metadata.update(self._load_metadata(candidate, base_folder=fld))
						break
  
  
  
  

	def __len__(self) -> int:
		return len(self._samples)

	def __getitem__(self, idx: int):
		path, label = self._samples[idx]
		img = Image.open(path).convert("RGB")

		# apply user transform first (operates on PIL.Image)
		if self.transform is not None:
			img = self.transform(img)

		if self.return_type == "pil":
			out = img
		else:
			# if the transform returned a tensor already, accept it
			if isinstance(img, torch.Tensor):
				tensor = img
			else:
				# assume PIL.Image
				tensor = self._to_tensor(img)
			out = tensor

		prompt = None
		if self.return_prompt:
			key_abs = str(path.resolve())
			prompt = self.metadata.get(key_abs)

			if prompt is None:
				prompt = self.metadata.get(path.name) or self.metadata.get(path.stem)
		


		# if self.with_labels and self.return_prompt:
		# 	return out, prompt, label
		# if self.with_labels:
		# 	return out, label
		# if self.return_prompt:
		# 	return out, prompt
		# return out

		path_str = str(path.resolve())

		if self.with_labels and self.return_prompt and self.return_path:
			return out, prompt, label, path_str

		if self.return_prompt and self.return_path:
			return out, prompt, path_str

		if self.with_labels and self.return_path:
			return out, label, path_str

		if self.return_path:
			return out, path_str

		if self.with_labels and self.return_prompt:
			return out, prompt, label

		if self.with_labels:
			return out, label

		if self.return_prompt:
			return out, prompt

		return out


	def _load_metadata(self, path: Path, base_folder: Optional[Path] = None) -> dict:
		"""Load metadata mapping filenames (or stems) to prompts.

		Supports:
		- JSONL (one JSON object per line)
		- JSON list of objects
		- JSON dict mapping filenames to prompts
		"""
		import json

		if not path.exists():
			raise ValueError(f"metadata_path does not exist: {path}")

		filename_keys = {"image", "file_name", "filename", "file"}
		prompt_keys = {"prompt", "text", "caption", "description"}

		data = {}

		def register(fname: str, prompt: str):
			if base_folder is not None:
				full = (base_folder / Path(fname).name).resolve()
				data[str(full)] = prompt
			data[Path(fname).name] = prompt
			data[Path(fname).stem] = prompt

		# ---- 1) Try JSONL first (SAFE for both .jsonl and .json) ----
		with path.open("r", encoding="utf-8") as f:
			lines = [line.strip() for line in f if line.strip()]

		jsonl_ok = True
		for line in lines:
			try:
				obj = json.loads(line)
			except json.JSONDecodeError:
				jsonl_ok = False
				break

			if not isinstance(obj, dict):
				continue

			fname = next((obj[k] for k in filename_keys if k in obj), None)
			# if fname is not None:
			# 	fname = fname.replace("img_", "")
			prompt = next((obj[k] for k in prompt_keys if k in obj), None)

			if fname and prompt:
				register(fname, prompt)

		if jsonl_ok and data:
			return data  # ✅ successfully parsed as JSONL

		# ---- 2) Fall back to normal JSON ----
		with path.open("r", encoding="utf-8") as f:
			obj = json.load(f)
		
		# Special case: Infinity manifest.json
		if isinstance(obj, list) and obj and isinstance(obj[0], dict) \
		    and "prompt" in obj[0] and "path" in obj[0]:
			for entry in obj:
				prompt = entry.get("prompt")
				path = entry.get("path")
				if prompt and path:
					data[str(Path(path).resolve())] = prompt
			return data


		if isinstance(obj, dict):
			# filename -> prompt mapping
			if all(isinstance(v, str) for v in obj.values()):
				for k, v in obj.items():
					register(k, v)
			else:
				for entry in obj.values():
					if not isinstance(entry, dict):
						continue
					fname = next((entry[k] for k in filename_keys if k in entry), None)
					prompt = next((entry[k] for k in prompt_keys if k in entry), None)
					if fname and prompt:
						register(fname, prompt)

		elif isinstance(obj, list):
			for entry in obj:
				if not isinstance(entry, dict):
					continue
				fname = next((entry[k] for k in filename_keys if k in entry), None)
				prompt = next((entry[k] for k in prompt_keys if k in entry), None)
				if fname and prompt:
					register(fname, prompt)

		return data



__all__ = ["DualFolderDataset"]

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Export DualFolderDataset to Infinity-compatible JSONL"
    )

    parser.add_argument("--folder1", required=True, help="Path to clean images folder")
    parser.add_argument("--folder2", default=None, help="Path to watermarked images folder")
    parser.add_argument("--p1", type=float, default=0.5, help="Proportion from folder1")
    parser.add_argument("--p2", type=float, default=0.5, help="Proportion from folder2")
    parser.add_argument("--num_images", type=int, required=True, help="Total number of samples")

    parser.add_argument(
        "--metadata",
        help="Path to MS-COCO Karpathy captions (json or jsonl)",
    )
    parser.add_argument(
        "--h_div_w",
        type=float,
        required=True,
        help="Height / width ratio template (e.g. 1.0 for 1024x1024)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL file path (e.g. 1.0_5000.jsonl)",
    )

    parser.add_argument("--shuffle", action="store_true", help="Shuffle before sampling")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    dataset = DualFolderDataset(
        folder1=args.folder1,
        folder2=args.folder2,
        p1=args.p1,
        p2=args.p2,
        num_images=args.num_images,
        return_type="pil",          # IMPORTANT: Infinity wants paths, not tensors
        return_prompt=True,
        return_path=True,
        metadata_path=args.metadata,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            img, prompt, path = dataset[i]

            if prompt is None:
                skipped += 1
                continue

            record = {
                "image_path": path,
                "h_div_w": args.h_div_w,
                "long_caption": prompt,
                "long_caption_type": "MSCOCO",
                "text": prompt,
                "short_caption_type": "user",
            }

            f.write(json.dumps(record) + "\n")
            written += 1

    print(
        f"✔ Infinity JSONL written to {out_path}\n"
        f"  Samples written: {written}\n"
        f"  Samples skipped (missing caption): {skipped}"
    )



