import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

MODEL_CONFIG: dict[str, dict[str, object]] = {
    "infinity": {
        "repo": "FoundationVision/Infinity",
        "files": [
            "infinity_2b_reg.pth",
            "infinity_vae_d32reg.pth",
            "infinity_8b_weights",  # Infinity 8b model
            "infinity_vae_d56_f8_14_patchify.pth"
        ],
        "subpath": Path("models/Infinity/checkpoints"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BitMark model checkpoints.")

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        default=Path("./weights"),
        help="Root directory where checkpoints will be saved.",
    )

    available_models = list(MODEL_CONFIG.keys())
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=f"Comma-separated list of models to download. Options: {available_models}. Defaults to 'all'.",
    )

    return parser.parse_args()


def download_file(repo_id: str, filename: str, output_dir: Path) -> None:
    """
    Downloads a single file from HF Hub to the specified output directory.
    Fails silently (returns) if the file already exists locally.
    """
    target_path = output_dir / filename

    if target_path.exists():
        print(f"[!] Skipping {filename} : File already exists at {target_path}")
        return

    print(f"[*] Downloading {filename} from {repo_id}...")

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"[+] Successfully downloaded {filename}")

    except (EntryNotFoundError, RepositoryNotFoundError) as e:
        print(f"[-] Failed to download {filename}: {e}")
    except Exception as e:
        print(f"[-] Unexpected error downloading {filename}: {e}")


def main() -> None:
    args = parse_args()

    # 1. Resolve output directory
    root_dir: Path = args.output_dir.resolve()
    if not root_dir.exists():
        print(f"[*] Creating directory: {root_dir}")
        root_dir.mkdir(parents=True, exist_ok=True)

    # 2. Determine models
    selected_models: list[str]
    if args.models.lower() == "all":
        selected_models = list(MODEL_CONFIG.keys())
    else:
        selected_models = [m.strip().lower() for m in args.models.split(",")]

    # 3. Download loop
    for model_key in selected_models:
        if model_key not in MODEL_CONFIG:
            print(f"[!] Warning: Unknown model '{model_key}'. Skipping.")
            continue

        config = MODEL_CONFIG[model_key]
        repo_id = str(config["repo"])
        files = list(config["files"])  # type: ignore

        # Resolve specific sub-folder from config
        subpath = Path(str(config["subpath"]))
        model_dir = root_dir / subpath

        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Processing {model_key} -> {model_dir} ---")
        for file_name in files:
            download_file(repo_id, str(file_name), model_dir)

    print("\n[=] Download process completed.")


if __name__ == "__main__":
    main()
