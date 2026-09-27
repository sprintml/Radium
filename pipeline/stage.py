"""Common protocol for pipeline stage orchestrators."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from omegaconf import OmegaConf


@runtime_checkable
class StageOrchestrator(Protocol):
    name: str

    def load_config(self, config_path: str): ...
    def is_complete(self, config, command: list[str]) -> bool: ...


def extract_arg(command: list[str], flag: str) -> str | None:
    try:
        idx = command.index(flag)
        return command[idx + 1]
    except (ValueError, IndexError):
        return None


def save_stage_config_snapshot(
    config,
    output_dir: str | None,
    *,
    filename: str,
    overrides: dict[str, object] | None = None,
) -> str | None:
    """Persist a stage config snapshot into the stage's working/output directory."""
    if not output_dir:
        return None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if OmegaConf.is_config(config):
        snapshot = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    else:
        snapshot = OmegaConf.create(config)

    for key, value in (overrides or {}).items():
        if value is None:
            continue
        OmegaConf.update(snapshot, key, value, merge=True)

    target = out_path / filename
    OmegaConf.save(snapshot, str(target))
    return str(target)
