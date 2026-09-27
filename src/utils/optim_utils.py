import torch
from torchvision import transforms
from datasets import load_dataset

from PIL import Image, ImageFilter
import random
import numpy as np
from typing import Any, Mapping
import json
import os

def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)
    

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x


# for one prompt to multiple images
def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return (image_features @ text_features.T).mean(-1)


_PROMPT_KEYS = ("Prompt", "prompt", "text", "caption")


def get_dataset(args):
    datasets_cfg = args.datasets

    # Local JSONL prompt file takes priority over datasets.name.
    prompt_file = getattr(datasets_cfg, "prompt_file", None)
    if prompt_file:
        with open(prompt_file) as f:
            dataset = [json.loads(line) for line in f if line.strip()]
        if not dataset:
            raise ValueError(f"prompt_file is empty: {prompt_file}")
        first = dataset[0]
        prompt_key = next((k for k in _PROMPT_KEYS if k in first), None)
        if prompt_key is None:
            raise ValueError(
                f"Could not detect prompt key in {prompt_file}. "
                f"Expected one of: {_PROMPT_KEYS}"
            )
        return dataset, prompt_key

    prompts = getattr(datasets_cfg, "name", None)
    if not prompts:
        raise ValueError("Either datasets.prompt_file or datasets.name is required")
    if 'laion' in prompts:
        dataset = load_dataset(prompts, cache_dir=args.watermark.cache_path)['train']
        prompt_key = 'TEXT'
    elif 'coco' in prompts:
        base_dir = getattr(datasets_cfg, "coco_annotation_dir", None)
        if args.datasets.split == 'train':
            with open(f'{os.path.join(base_dir, "coco_karpathy_train.json")}') as f:
                data = json.load(f)
                # Keep only the first caption for each unique image_id
                seen_images = set()
                dataset = []
                for item in data:
                    if item['image_id'] not in seen_images:
                        seen_images.add(item['image_id'])
                        dataset.append(item)
                prompt_key = 'caption'
        else:
            with open(f'{os.path.join(base_dir, "coco_karpathy_val.json")}') as f:
                data = json.load(f)
                # Keep only the first caption for each unique image_id
                seen_images = set()
                dataset = []
                for item in data:
                    if item['image_id'] not in seen_images:
                        seen_images.add(item['image_id'])
                        dataset.append(item)
                prompt_key = 'caption'
    
    
    else:
        dataset = load_dataset(prompts, cache_dir=args.watermark.cache_path)[args.datasets.split]
        prompt_key = 'Prompt'

    return dataset, prompt_key


def get_prompts_from_config(args, iteration: int | None = None, num_images: int | None = None):
    """
    Return the list of prompts that would be used by the watermark generation pipeline for the
    given configuration.

    Args:
        args: Config-like object (OmegaConf/namespace) containing at least
              `datasets.name`, `datasets.split`, and `dataset_params.num_images`.
        iteration: Optional integer to override `args.watermark.iteration` (default uses config).
        num_images: Optional integer to override `args.dataset_params.num_images`.

    Returns:
        A tuple (prompts, start_idx, end_idx) where `prompts` is a list of prompt strings
        taken from the underlying dataset, and `start_idx`/`end_idx` indicate the slice used.

    Raises:
        ValueError: if the computed start index is out of range for the dataset.
    """
    dataset, prompt_key = get_dataset(args)

    if num_images is None:
        dataset_params = getattr(args, "dataset_params", None)
        num_images = dataset_params.num_images

    if iteration is None:
        iteration = int(getattr(args.watermark, "iteration", 0) or 0)

    start_idx = int(iteration) * int(num_images)
    dataset_len = len(dataset)

    if start_idx >= dataset_len:
        raise ValueError(f"iteration {iteration} out of range: start index {start_idx} >= dataset length {dataset_len}")

    end_idx = min(start_idx + int(num_images), dataset_len)

    # Extract prompts mirroring how generation_watermark builds them
    prompts = [dataset[i][prompt_key] for i in range(start_idx, end_idx)]

    return prompts, start_idx, end_idx


def get_finetuning_prompts(args):
    """
    Return the exact list of prompts used to finetune the model, leveraging the
    DualFolderDataset loading logic.
    
    During finetuning, a DualFolderDataset loads watermarked images from one or two
    folders with metadata.jsonl/metadata.json files containing the prompts.
    This function reconstructs the same set of prompts that would be seen during training.
    
    Args:
        args: Config-like object (OmegaConf/namespace) containing:
            - datasets.input_dir: primary training folder
            - datasets.dual_dataset.enabled: whether dual-folder mode is used
            - datasets.dual_dataset.input_dir_2: secondary folder (if enabled)
            - datasets.dual_dataset.p1, p2: sampling proportions
            - datasets.dual_dataset.shuffle: whether to shuffle combined samples
            - training.max_train_samples: max samples to include (None = use all)
            - seed: seed for deterministic behavior
    
    Returns:
        A dict with keys:
            - 'prompts': list of prompt strings (in the order seen by the dataset)
            - 'num_samples': total number of samples in the dataset
            - 'folder1_samples': number of samples from folder1
            - 'folder2_samples': number of samples from folder2 (if dual mode)
            - 'folder1_files': list of filenames from folder1 (in dataset order)
            - 'folder2_files': list of filenames from folder2 (in dataset order)
            - 'metadata': dict mapping image filename -> prompt (for reference)
    """
    from src.datasets.loader import DualFolderDataset
    from pathlib import Path
    
    input_dir = args.datasets.input_dir
    
    # Check if dual dataset mode is enabled
    dual_cfg = args.datasets.get("dual_dataset", {})
    use_dual = dual_cfg.get("enabled", False)
    
    if use_dual:
        input_dir_2 = dual_cfg.get("input_dir_2")
        p1 = dual_cfg.get("p1", 0.5)
        p2 = dual_cfg.get("p2", 0.5)
        shuffle = dual_cfg.get("shuffle", False)
    else:
        input_dir_2 = None
        p1 = 1.0
        p2 = 0.0
        shuffle = False
    
    # Determine num_images for the dataset
    max_train_samples = args.training.get("max_train_samples")
    # We use a large default if max_train_samples is None
    num_images = max_train_samples if max_train_samples is not None else 100000
    
    # Create the dataset (with return_prompt=True, return_type='pil')
    dataset = DualFolderDataset(
        folder1=input_dir,
        folder2=input_dir_2,
        p1=p1,
        p2=p2,
        num_images=num_images,
        return_type="pil",
        with_labels=False,
        return_prompt=True,
        metadata_path=input_dir if use_dual else None,  # pass input_dir to help find metadata
        shuffle=shuffle,
        seed=args.get("seed", 42),
    )
    
    # Extract prompts, metadata, and filenames in dataset order
    prompts = []
    metadata_dict = {}
    folder1_files = []
    folder2_files = []
    
    for idx in range(len(dataset)):
        img, prompt = dataset[idx]
        if prompt is not None:
            prompts.append(prompt)
        
        # Get the sample path and label
        path, label = dataset._samples[idx]
        
        # Store metadata for reference
        metadata_dict[path.name] = prompt
        
        # Store full filename based on which folder it came from
        if label == 0:  # from folder1
            folder1_files.append(str(path))
        elif label == 1:  # from folder2
            folder2_files.append(str(path))
    
    return {
        "prompts": prompts,
        "num_samples": len(dataset),
        "folder1_samples": len([s for s in dataset._samples if s[1] == 0]),
        "folder2_samples": len([s for s in dataset._samples if s[1] == 1]),
        "folder1_files": folder1_files,
        "folder2_files": folder2_files,
        "metadata": metadata_dict,
    }
