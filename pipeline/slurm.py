"""SLURM submission helpers for run_pipeline.py.

Separated so that the planner/orchestrator stays free of sbatch details.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from pipeline.stage import extract_arg

_FLAG_MAP = [
    ("partition",     "--partition"),
    ("gres",          "--gres"),
    ("cpus_per_task", "--cpus-per-task"),
    ("mem",           "--mem"),
    ("account",       "--account"),
    ("time",          "--time"),
    ("nodes",         "--nodes"),
    ("output",        "--output"),
    ("exclude",       "--exclude"),
    ("kill_on_invalid_dep", "--kill-on-invalid-dep"),
]


def _resolve_profile_values(cluster_cfg, gpu_profile: str | None) -> dict:
    
    profiles = cluster_cfg.get("gpu_profiles") if hasattr(cluster_cfg, "get") else None
    if profiles is None:
        if gpu_profile and gpu_profile != "default":
            print(
                f"[slurm] cluster has no gpu_profiles; ignoring requested profile "
                f"'{gpu_profile}' and falling back to top-level resource keys"
            )
        return {}
    name = gpu_profile or "default"
    profile = profiles.get(name)
    if profile is None:
        available = list(profiles.keys()) if hasattr(profiles, "keys") else []
        raise ValueError(
            f"gpu_profile '{name}' not defined in cluster config (have: {available})"
        )
    return {k: profile.get(k) for k in profile.keys()}


def submit_job(
    command: list[str],
    cluster: str,
    script_dir: Path,
    dependency_job_ids: list[str] | None = None,
    gpu_profile: str | None = None,
) -> str:
    """Submit a stage job via sbatch and return the job ID.

    Resource flags are read from script_dir/clusters/<cluster>.yaml so
    run_stage.sh stays free of hardcoded #SBATCH directives.
    GPU placement (partition/gres/exclude) comes from the named profile
    under `gpu_profiles:` in the cluster config; profile values override
    the flat top-level fallback.
    Dependencies are expressed as --dependency=afterok:<id>:<id>...
    """
    config_path = extract_arg(command, "--config_path")
    if not config_path:
        raise ValueError("Command has no --config_path argument, cannot submit to sbatch")

    module_idx = next((idx for idx, tok in enumerate(command) if tok == "-m"), None)
    if module_idx is None or module_idx + 1 >= len(command):
        raise ValueError(f"Command has no -m <module> token: {command}")
    module_path = command[module_idx + 1]

    cluster_cfg_path = script_dir / "clusters" / f"{cluster}.yaml"
    if not cluster_cfg_path.exists():
        raise FileNotFoundError(f"Cluster config not found: {cluster_cfg_path}")
    cluster_cfg = OmegaConf.load(str(cluster_cfg_path))
    profile_values = _resolve_profile_values(cluster_cfg, gpu_profile)

    run_stage_script = script_dir / "run_stage.sh"
    if not run_stage_script.exists():
        raise FileNotFoundError(f"Stage runner script not found: {run_stage_script}")

    cfg_p = Path(config_path)
    config_group = cfg_p.parent.name or "default"
    job_name = cfg_p.stem  # e.g. "finetuning_M2" — stage + run index from the config name

    sbatch_cmd = ["sbatch"]
    if dependency_job_ids:
        sbatch_cmd.append(f"--dependency=afterok:{':'.join(dependency_job_ids)}")
    for key, flag in _FLAG_MAP:
        if key == "output":
            continue  # overridden below to group logs by config
        val = profile_values.get(key) if key in profile_values else cluster_cfg.get(key)
        if val is not None:
            sbatch_cmd.append(f"{flag}={val}")

    # Group logs by config directory, name them <stage>_<jobid>.out
    log_root_template = cluster_cfg.get("output", "slurm_out/%j.out")
    log_root_dir = Path(log_root_template).parent or Path("slurm_out")
    log_dir = log_root_dir / config_group
    log_dir.mkdir(parents=True, exist_ok=True)
    sbatch_cmd.append(f"--output={log_dir}/%j_%x.out")
    sbatch_cmd.append(f"--job-name={job_name}")
    sbatch_cmd += [str(run_stage_script), module_path, config_path]

    result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout}")
    return match.group(1)
