import os
import glob
import json
import logging
from typing import Any, Mapping, Iterable, Union, List, Callable, Optional, Tuple
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from src.utils.optim_utils import get_finetuning_prompts
import numpy as np
import re


def resolve_globs(glob_paths: Union[str, Iterable[str]]):
    """Returns filepaths corresponding to input filepath pattern(s)."""
    filepaths = []
    if isinstance(glob_paths, str):
        glob_paths = [glob_paths]

    for path in glob_paths:
        filepaths.extend(glob.glob(path))

    return filepaths


def read_jsonlines(filename: str) -> Iterable[Mapping[str, Any]]:
    """Yields an iterable of Python dicts after reading jsonlines from the input file."""
    file_size = os.path.getsize(filename)
    with open(filename) as fp:
        for line in tqdm(fp.readlines(), desc=f'Reading JSON lines from {filename}', unit='lines'):
            try:
                example = json.loads(line)
                yield example
            except json.JSONDecodeError as ex:
                logging.error(f'Input text: "{line}"')
                logging.error(ex.args)
                raise ex


def hf_read_jsonlines(filename: str, 
                   n: Optional[int]=None, 
                   minimal_questions: Optional[bool]=False,
                   unique_questions: Optional[bool] = False) -> Iterable[Mapping[str, Any]]:
    """Yields an iterable of Python dicts after reading jsonlines from the input file.
       Optionally reads only first n lines from file."""
    file_size = os.path.getsize(filename)
    # O(n) but no memory
    with open(filename) as f:
        num_lines= sum(1 for _ in f)
        if n is None: 
            n = num_lines

    # returning a generator with the scope stmt seemed to be the issue, but I am not 100% sure
    # I also don't know if there's a side effect, but I can't see how the scope wouldn't have
    # remained upen in the first place with the original version...
    # with open(filename) as fp:
    def line_generator():
        unique_qc_ids = set()
        # note, I am p sure that readlines is not lazy, returns a list, thus really only the
        # object conversion is lazy
        for i, line in tqdm(enumerate(open(filename).readlines()[:n]), desc=f'Reading JSON lines from {filename}', unit='lines'):
            try:
                full_example = json.loads(line)

                if unique_questions:
                    qc_id = full_example["object"]["qc_id"]
                    if qc_id in unique_qc_ids:
                        continue
                    else:
                        unique_qc_ids.add(qc_id)

                if not minimal_questions:
                    example = full_example
                else:
                    full_example = full_example
                    q_object = full_example["object"]
                    q_object.pop("question_info")
                    example= {}
                    example["object"] = {
                        "answer":q_object["answer"],
                        "clue_spans":q_object["clue_spans"],
                        "qc_id":q_object["qc_id"],
                        "question_text":q_object["question_text"],
                    }
                yield example

            except json.JSONDecodeError as ex:
                logging.error(f'Input text: "{line}"')
                logging.error(ex.args)
                raise ex
    return line_generator


def load_jsonlines(filename: str) -> List[Mapping[str, Any]]:
    """Returns a list of Python dicts after reading jsonlines from the input file."""
    return list(read_jsonlines(filename))


def write_jsonlines(objs: Iterable[Mapping[str, Any]], filename: str, to_dict: Callable = lambda x: x):
    """Writes a list of Python Mappings as jsonlines at the input file."""
    with open(filename, 'w') as fp:
        for obj in tqdm(objs, desc=f'Writing JSON lines at {filename}'):
            fp.write(json.dumps(to_dict(obj)))
            fp.write('\n')


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def write_json(obj: Mapping[str, Any], filename: str, indent:int=None):
    """Writes a Python Mapping at the input file in JSON format."""
    with open(filename, 'w') as fp:
        json.dump(obj, fp, indent=indent)


def print_json(d, indent=4):
    print(json.dumps(d, indent=indent))
    
    
def save_images(images, filenames):
    """Saves a list of PIL images to the corresponding list of filenames."""
    #handle normalization if needed
    if type(images[0]) is not Image.Image:
        images = [( (img + 1.0) / 2.0).clamp(0, 1) for img in images]
        images = [transforms.ToPILImage()(img.cpu()) for img in images]
    
    
    for img, fname in zip(images, filenames):
        img.save(fname)
        
def load_images_from_folder(folder):
    image_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    images = [
        Image.open(os.path.join(folder, f)).convert("RGB")
        for f in image_files
    ]

    return images, image_files


def _build_filenames_with_supervision_degree(
    clean_filenames: List[str],
    finetune_filenames: List[str],
    supervision_degree: float,
    limit: int | None = None,
) -> List[str]:
    """Compose filenames with a controlled amount of finetuning prompts.

    A fraction `supervision_degree` of the selected filenames comes from
    `finetune_filenames`, and the remainder comes from clean filenames that are
    not in the finetuning set.
    """
    if supervision_degree < 0:
        raise ValueError(f"supervision_degree must be >= 0, got {supervision_degree}")

    target_count = limit if limit is not None else len(clean_filenames)
    target_count = min(target_count, len(clean_filenames))

    clean_set = set(clean_filenames)
    filtered_finetune = [f for f in finetune_filenames if f in clean_set]
    finetune_set = set(filtered_finetune)
    non_finetune = [f for f in clean_filenames if f not in finetune_set]

    num_finetune = int(round(target_count * supervision_degree))
    num_finetune = max(0, min(num_finetune, target_count, len(filtered_finetune)))

    selected_finetune = filtered_finetune[:num_finetune]
    remaining_needed = target_count - len(selected_finetune)
    selected_non_finetune = non_finetune[:remaining_needed]

    filenames = selected_finetune + selected_non_finetune

    # If non-finetuning pool is too small, fill from remaining clean filenames.
    if len(filenames) < target_count:
        selected_set = set(filenames)
        remaining_clean = [f for f in clean_filenames if f not in selected_set]
        filenames.extend(remaining_clean[: target_count - len(filenames)])

    return filenames



def load_paired_images(
    clean_dir: str,
    watermarked_dir: str,
    config,
    finetuning_config,
    limit: int | None = None,
    watermarked_offset: int = 0,
    supervision_degree: float | None = None,
) -> Tuple[List, List, List]:
    """Load paired images from both clean and watermarked directories.
    
    Filenames are shuffled before loading so only `limit` images are ever
    opened — avoiding loading all 5k images into memory when only 200 are needed.
    
    Args:
        watermarked_offset: Offset to subtract from image numbers when searching in watermarked dir
        supervision_degree: Fraction of selected filenames that should come
            from finetuning prompts. If None, use legacy behavior.
    
    Returns:
        Tuple of (clean_images, watermarked_images, filenames)
        where each list contains images/filenames in the same order.
    """
    if clean_dir is None or watermarked_dir is None:
        raise ValueError("Both clean_dir and watermarked_dir must be set")

    if not os.path.exists(clean_dir):
        raise FileNotFoundError(f"Clean directory does not exist: {clean_dir}")
    if not os.path.exists(watermarked_dir):
        raise FileNotFoundError(f"Watermarked directory does not exist: {watermarked_dir}")

    # Normalize eval params
    if isinstance(config, dict):
        eval_params = config.get("eval_params", {}) or {}
    else:
        eval_params = getattr(config, "eval_params", {}) or {}

    # Check if watermark config has training_config_path
    has_training_config = False
    training_config_path = None
    
    if isinstance(config, dict):
        watermark_cfg = config.get("watermark", {})
        training_config_path = watermark_cfg.get("training_config_path")
        has_training_config = training_config_path is not None
    else:
        has_training_config = hasattr(config, "watermark") and hasattr(config.watermark, "training_config_path")
        if has_training_config:
            training_config_path = config.watermark.training_config_path

    # Get watermark name from config
    if isinstance(config, dict):
        watermark_cfg = config.get("watermark", {})
        wm_name = watermark_cfg.get("name")
    else:
        wm_name = getattr(config.watermark, "name", None) if hasattr(config, "watermark") else None

    # Load filenames only (not images yet)
    if finetuning_config:
        prompt_dict = get_finetuning_prompts(finetuning_config)
        filenames = [os.path.basename(f) for f in prompt_dict["folder2_files"]]
        clean_filenames = sorted([
            f for f in os.listdir(clean_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        i = 0
        while len(filenames) < len(clean_filenames):
            filenames.append(clean_filenames[i])
            i += 1
    else:
        # Only load filenames, not images
        filenames = sorted([
            f for f in os.listdir(clean_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    # Shuffle filenames BEFORE loading images so we only open `limit` files
    filenames = list(filenames)
    np.random.shuffle(filenames)

    # Verify pairs exist and load only up to limit
    clean_images = []
    watermarked_images = []
    valid_filenames = []

    from PIL import Image
    debug_counter = 0
    for fname in filenames:
        # Stop early once we have enough
        if limit and len(clean_images) >= limit:
            break

        # Find watermarked match
        watermarked_path = os.path.join(watermarked_dir, fname)
        if not os.path.exists(watermarked_path):
            matched_watermarked = find_matching_image(
                fname, watermarked_dir, wm_name=wm_name,
                is_watermarked_dir=True, custom_offset=watermarked_offset,
                debug=(debug_counter < 5)
            )
            if debug_counter < 5:
                print(f"DEBUG: fname={fname}, wm_name={wm_name}, offset={watermarked_offset}, matched_watermarked={matched_watermarked}")
                debug_counter += 1
            if matched_watermarked:
                watermarked_path = os.path.join(watermarked_dir, matched_watermarked)
            else:
                continue

        # Find clean match
        clean_path = os.path.join(clean_dir, fname)
        if not os.path.exists(clean_path):
            matched_clean = find_matching_image(
                fname, clean_dir, wm_name=wm_name, is_watermarked_dir=False
            )
            if matched_clean:
                clean_path = os.path.join(clean_dir, matched_clean)
            else:
                continue

        # Load both images only when we know the pair exists
        try:
            clean_img = Image.open(clean_path).convert("RGB")
            watermarked_img = Image.open(watermarked_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to load image pair for {fname}: {e}")
            continue

        clean_images.append(clean_img)
        watermarked_images.append(watermarked_img)
        valid_filenames.append(fname)

    if not clean_images:
        print(f"DEBUG: Total filenames from clean dir: {len(filenames)}")
        print(f"DEBUG: First few filenames: {filenames[:10]}")
        print(f"DEBUG: Watermarked dir: {watermarked_dir}")
        import subprocess
        result = subprocess.run(["ls", "-1", watermarked_dir], capture_output=True, text=True)
        print(f"DEBUG: First few watermarked files: {result.stdout.split()[:10]}")
        raise FileNotFoundError(f"No paired images found in {clean_dir} and {watermarked_dir}")

    print(f"DEBUG: Loaded {len(clean_images)} paired images (limit={limit})")
    print(f"DEBUG: First few valid filenames: {valid_filenames[:10]}")
    print(f"Loading {len(clean_images)} paired images from:")
    print(f"  Clean: {clean_dir}")
    print(f"  Watermarked: {watermarked_dir}")

    if limit:
        assert len(clean_images) == limit, f"Expected {limit} clean images, but got {len(clean_images)}"

    return clean_images, watermarked_images, valid_filenames


def find_matching_image(target_filename, search_dir, wm_name=None, is_watermarked_dir=False, custom_offset=None, debug=False):
    """Find an image in search_dir that has the same numeric identifier as target_filename.

    For bitmark watermark, filenames may be offset by +5000 (unsupervised set) or unshifted (supervised).
    Custom offset can be provided for other specific evaluation scenarios.
    
    Args:
        custom_offset: Custom offset to use instead of default bitmark offset (e.g., for treering unsupervised)
    
    Returns:
        The matching filename if found, None otherwise.
    """
    target_number = extract_number_from_filename(target_filename)
    if target_number is None:
        return None

    # Build candidate numeric identifiers to try when matching
    candidate_numbers = [target_number]

    # Custom offset takes precedence (used for special cases like treering cross-architecture)
    if custom_offset:
        if is_watermarked_dir:
            # Watermarked filenames are shifted down (e.g., img_005000 -> 00000)
            shifted = int(target_number) - custom_offset
            if shifted >= 0:
                candidate_numbers.append(str(shifted))
        else:
            # If ever needed in reverse direction, allow shifting up
            candidate_numbers.append(str(int(target_number) + custom_offset))

    # Default bitmark handling (only when no custom offset is provided)
    # elif wm_name == "bitmark":
    #     # In clean M2 evaluation, unsupervised clean images are often img_005000+ while
    #     # bitmark-generated images are zero-indexed (00000.jpg+).
    #     if is_watermarked_dir:
    #         # Matching clean -> watermarked: 005000 -> 00000
    #         if int(target_number) >= 5000:
    #             candidate_numbers.append(str(int(target_number) - 5000))
    #     else:
    #         # Matching watermarked -> clean (reverse direction): 00000 -> 005000
    #         candidate_numbers.append(str(int(target_number) + 5000))
    # Default bitmark handling (only when no custom offset is provided)
    elif wm_name == "bitmark":
        num = int(target_number)

        # Always consider original
        # candidate_numbers already has target_number

        # clean (0) -> watermarked (5000)
        candidate_numbers.append(str(num + 5000))

        # watermarked (5000) -> clean (0)
        if num >= 5000:
            candidate_numbers.append(str(num - 5000))
    
    if debug:
        print(f"DEBUG find_matching_image: target={target_filename}, target_num={target_number}, candidates={candidate_numbers}, is_wm={is_watermarked_dir}")

    # Get all image files in the directory
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    for fname in os.listdir(search_dir):
        if any(fname.endswith(ext) for ext in image_extensions):
            fname_number = extract_number_from_filename(fname)
            if fname_number in candidate_numbers:
                if debug:
                    print(f"DEBUG find_matching_image: FOUND match={fname}, fname_num={fname_number}")
                return fname
    if debug:
        print(f"DEBUG find_matching_image: NO MATCH FOUND")
    return None



def extract_number_from_filename(filename):
    """Extract numeric identifier from filename.
    
    Examples:
        'img_004490.png' -> '4490'
        '004490.jpg' -> '4490'
        'image_4490.png' -> '4490'
    """
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]
    # Find all numbers in the filename
    numbers = re.findall(r'\d+', name_without_ext)
    if numbers:
        # Return the last number found, stripped of leading zeros
        return str(int(numbers[-1]))
    return None