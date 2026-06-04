---
name: prep-hf-dataset-zephyr
description: Download a Hugging Face dataset and transform it to GCS parquet with a Zephyr pipeline.
---

# Prep HF Dataset (Zephyr)

Download a HF dataset and run a Zephyr pipeline to write processed parquet under
marin GCS, as a CPU-only iris job. Reference: `experiments/protein/prep_quality_splits.py`.

## Paths

```python
RAW_SUBPATH = f"<area>/raw/<dataset>-{HF_REVISION_SHORT}"
DATA_VERSION = "v1"  # bump when transform semantics change -> forks the cache
OUTPUT_SUBPATH = f"<area>/processed/<dataset>/<name>-{DATA_VERSION}"
out = f"{marin_prefix()}/{OUTPUT_SUBPATH}/{suffix}/"
```

`marin_prefix()` resolves to the region bucket via the GCE metadata server (pin
the job's `--region`). **Confirm with the operator** before running: `DATA_VERSION`,
`OUTPUT_SUBPATH`, and `NUM_OUTPUT_SHARDS` (size for ~4 GB/shard — see Pipeline; else 8).

## Download (skip if already done)

```python
marker = f"{raw_path}/provenance.json"              # written only on success
if not fs.exists(marker):
    download_hf_step("raw/<dataset>", hf_dataset_id=ID, revision=REV,
                     hf_urls_glob=[f"{cfg}/**/*.parquet" for cfg in CONFIGS]).fn(raw_path)
```

## Pipeline

A Zephyr `Dataset` chain. The transform is task-specific — `.filter` below is just
*one* example stage; swap in whatever map / flat_map / filter the task needs:

```python
pipeline = (
    Dataset.from_files(f"{raw_path}/{cfg}/{split}/*.parquet")
        .flat_map(load_rows)         # e.g. stamp each row with its HF config
        .filter(predicate)           # EXAMPLE transform — replace per task
        .reshard(NUM_OUTPUT_SHARDS)  # see below
        .write_parquet(f"{out}/data-{{shard:05d}}-of-{{total:05d}}.parquet",
                       skip_existing=True)
)
ZephyrContext(name=..., resources=ResourceConfig(cpu=1, ram="8g")).execute(pipeline)
```

**Reshard, don't inherit input sharding:** without `.reshard`, `write_parquet`
emits one output per *input* shard; a narrow filter leaves most empty, and marin
tokenize only inspects the first file (empty shard 0 → `IndexError`).

**Sizing `NUM_OUTPUT_SHARDS`:** target **~4 GB/shard** — ask the operator to estimate the
bucket's total output bytes and divide (round up). Default to `8` if that estimate isn't done.

## Notes

- Submit **one iris job per config/doctype**, selecting the subset with `-e RUNS`
  (CSV substring filter over bucket suffixes; empty = all).
