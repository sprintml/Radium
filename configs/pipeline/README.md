# Pipeline Configs

Top-level pipeline run configs consumed by `pipeline/run_pipeline.py`.

## Files

- `run_pipeline.yaml` — top-level orchestrator config for a generation / finetuning / detection / evaluation chain.
- `ldm_config.yaml` — backend helper config referenced by StableSignature.

## Usage

```bash
# Build plan and run all pending steps
python -m pipeline.run_pipeline --config_path configs/pipeline/run_pipeline.yaml

# Inspect the plan without launching any jobs
python -m pipeline.run_pipeline --config_path configs/pipeline/run_pipeline.yaml --dry-run
```

`--dry-run` builds the plan, writes the manifest, and prints the pending step count — no stage commands are launched.

## Stage names

The chain decomposes generation and detection by target. Each name selects a specific role:

| Stage | Target | Writes |
|---|---|---|
| `generation_m1_clean` | M1, prompt set `{i}` | `M1/clean/{dataset}/prompt_set_{i}/` |
| `generation_m1_clean_unsupervised` | M1, prompt set `{i+1}` | `M1/clean/{dataset}/prompt_set_{i+1}/` |
| `generation_m1_watermarked` | M1 watermarked (i=1 only) | `M1/{wm}/{wm_params}/{dataset}/prompt_set_1/` |
| `finetuning` | trains `M_{i+1}` | `…/{mi_model_type}/{run_id}/M_{i+1}/` |
| `generation_mi` | `M_{i+1}`, prompt set `{i}` | `…/{mi_model_type}/{run_id}/M_{i+1}/{dataset}/prompt_set_{i}/` |
| `generation_unsupervised_mi` | `M_{i+1}`, prompt set `{i+1}` | `…/{mi_model_type}/{run_id}/M_{i+1}/{dataset}/prompt_set_{i+1}/` |
| `detection_m1_supervised` | score M1 `prompt_set_{i}` | `…/clean/{dataset}/prompt_set_{i}/evaluations/scores/` |
| `detection_m1_unsupervised` | score M1 `prompt_set_{i+1}` | `…/clean/{dataset}/prompt_set_{i+1}/evaluations/scores/` |
| `detection_mi_supervised` | score Mi `prompt_set_{i}` | `…/{mi_model_type}/{run_id}/M_{i+1}/…/prompt_set_{i}/evaluations/scores/` |
| `detection_mi_unsupervised` | score Mi `prompt_set_{i+1}` | `…/{mi_model_type}/{run_id}/M_{i+1}/…/prompt_set_{i+1}/evaluations/scores/` |
| `evaluation` | statistical tests on cached scores | `…/{mi_model_type}/{run_id}/M_{i+1}/…/evaluations/` |

`generation_m1_watermarked` uses `only_iterations: [1]` — the M1 watermarked set is produced once and reused as training data for M2.

## Two-config split

Every stage consumes **two** configs:

- A **stage config** (`configs/<stage>/base.yaml`, or for finetuning one of `configs/finetuning/<model_family>/<file>.yaml`) — dataset params, training hyperparameters, runtime behavior. No watermark identity.
- A **watermark config** (`configs/watermarks/<method>.yaml`) — `watermark.method`, `watermark.type`, `watermark.params`, watermark-owned model paths. No stage-varying keys (`mode`, `iteration`, `run_id`).

Merge order: `watermark_config` ⊕ `stage_config` ⊕ `chain_overrides` (stage wins over watermark; overrides win over both). Switch watermarks end-to-end by changing `run.watermark_config`.

## Config structure

```yaml
run:
  name: my_run
  auto_generate_configs: true
  generated_configs_dir: output/pipeline_configs
  # Watermark identity — merged into every stage config:
  watermark_config:  configs/watermarks/treering.yaml
  # Generic per-stage bases:
  generation_config: configs/generation/base.yaml
  detection_config:  configs/detection/base.yaml
  finetuning_config: configs/models/sd21.yaml
  evaluation_config: configs/evaluation/base.yaml
  chain:
    start_i: 1
    end_i: 3
    stages: [generation_m1_clean, ..., evaluation]

stages:
  generation_m1_clean:
    # base_config / watermark_config: inherit run-level defaults
    chain_overrides:
      watermark.mode: clean_generate
      watermark.iteration: "{i}"

  finetuning:
    chain_overrides:
      watermark.iteration: "{next_i}"
    depends_on:
      - "generation_m1_watermarked_M{i}"             # i=1 only; filtered otherwise
      - "generation_unsupervised_mi_M{prev_i}"       # i>=2: Mi-1's output trains Mi

  detection_mi_supervised:
    chain_overrides:
      detection.target: mi
      watermark.iteration: "{i}"
    depends_on:
      - "generation_mi_M{i}"
```

Per-stage `base_config:` or `watermark_config:` keys override the run-level defaults.

### Auto-injected keys (in `auto_generate_configs: true` mode)

The pipeline writes one YAML per stage per iteration into `generated_configs_dir` with these keys injected on top of `chain_overrides`:

- `m1_model_type` — from the finetuning base config.
- `mi_model_type` — injected only for stages that touch Mi (generation_mi / detection_mi_* / finetuning / evaluation).
- `model.model_path` — injected only for stages that must **load** the Mi model (`generation_mi`, `detection_mi_*`). Points at `…/{mi_model_type}/{run_id}/M_{i+1}/model/`.
- `watermark.run_id` — injected for detection/evaluation stages. Derived from the finetuning config's training hyperparameters via `build_run_id`.
- `datasets.output_dir` — injected for generation stages so each prompt slice lands in a distinct directory.

See `pipeline/README.md` for planning modes, completion checks, and how to add a new stage.
