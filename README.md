# RADIUM: RadioActive Decay of Image-Underlaid Marks


## Abstract

Modern image generative models are able to produce photorealistic images. As those images become increasingly indistinguishable from real data and are published online, they are often scraped for subsequent training runs of new generative models. This practice of training on generated data has been shown to degrade model performance and cause model collapse. A possible mitigation lies in embedding radioactive watermarks into generated content. Radioactive watermarks are robust marks that are detectable in outputs of new models trained on watermarked data, enabling provenance tracing of generated content. In this work, we analyze the persistence of image watermarks across multiple training-generation runs. To do so, we introduce a novel statistical testing method RADIUM (RadioActive Decay of Image-Underlaid Marks) for reliable radioactivity detection across various watermarking methods. Using our RADIUM method, we observe disparate radioactivity across watermarking methods for image generative models. Only few watermarks remain detectable in subsequently trained models, while most decay severely, especially when the architecture of models differs. Our analysis highlights the critical need for more radioactive watermarking methods in the vision domain.


---

## Install

```bash
uv sync
```

Some watermarks require local model files (SD weights, ONNX encoder files). Ensure those exist at the paths referenced in your config before running.

---

## Running the full pipeline

The orchestrator reads a single top-level config, builds an execution plan, and runs only the pending steps:

```bash
python -m pipeline.run_pipeline --config_path configs/pipeline/run_pipeline.yaml
```

To inspect the plan without running anything (safe before launching GPU jobs):

```bash
python -m pipeline.run_pipeline --config_path configs/pipeline/run_pipeline.yaml --dry-run
```

The manifest (step names, statuses, commands) is written to `results/pipeline/run_manifest.json`.

See `pipeline/README.md` for full orchestrator documentation.

---

## Running individual stages

Each stage takes two configs: a generic stage config and a watermark config. The watermark config is merged under the stage config (stage wins for overlaps):

```bash
python -m pipeline.stages.generation \
    --config_path configs/generation/base.yaml \
    --watermark_config configs/watermarks/treering.yaml

python -m pipeline.stages.finetuning \
    --config_path configs/models/sd21.yaml \
    --watermark_config configs/watermarks/treering.yaml

python -m pipeline.stages.detection \
    --config_path configs/detection/base.yaml \
    --watermark_config configs/watermarks/treering.yaml

python -m pipeline.stages.evaluation \
    --config_path configs/evaluation/base.yaml \
    --watermark_config configs/watermarks/treering.yaml \
    --detection_mode evalue
```

Swap watermarks by pointing `--watermark_config` at a different file under `configs/watermarks/`. Each stage is idempotent — it checks whether output artifacts already exist and skips completed work.

---

## Config snapshots

Every stage writes a copy of its resolved config into its output directory at runtime. The file is named `{stage}_config.yaml` and sits alongside the stage's outputs. This makes every run self-documenting.

---

## Watermarks

See `src/watermarks/README.md` for the watermark registry, base classes, and how to add a new watermark.

Currently supported:

| Name | Type | Watermark config |
|---|---|---|
| TreeRing | in-generation | `configs/watermarks/treering.yaml` |
| StableSignature | in-generation | `configs/watermarks/stablesignature.yaml` |
| BitMark | in-generation | `configs/watermarks/bitmark.yaml` |
| RivaGAN | post-generation | `configs/watermarks/rivagan.yaml` |
| StegaStamp | post-generation | `configs/watermarks/stegastamp.yaml` |
| TrustMark | post-generation | `configs/watermarks/trustmark.yaml` |
| SIREN | post-generation | `configs/watermarks/siren.yaml` |


---

## Evaluation tests

Detection outputs are cached as JSON score files. The evaluation stage runs statistical tests on those files without reloading a model or GPU. Two tests are built in:

- `evalue` — e-value sequential test

---
