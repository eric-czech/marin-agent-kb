# MarinFold checkpoint sizes

This is a storage-planning reference for the Qwen3 protein models used in the
MarinFold 1.5B and 3B sweeps. Sizes are measured GCS object payloads beneath one
permanent checkpoint step prefix, as of 2026-08-17. They do not include the run
root, evaluation logs, or temporary checkpoints with TTLs.

Sizes use binary units: 1 GiB is 2^30 bytes.

## Quick reference

| Model | Checkpoint type | Reference size | Normal observed range | Observations |
| --- | --- | ---: | ---: | ---: |
| 1.5B | Levanter | **16.44 GiB** | 16.444–16.461 GiB | 375 normal |
| 1.5B | HF | **5.481 GiB** | 5.481406–5.481407 GiB | 184 |
| 3B | Levanter | **33.56 GiB** | 33.563–33.567 GiB | 106 normal |
| 3B | HF | **11.188 GiB** | exactly 11.187794 GiB | 48 |

| Model | One Levanter checkpoint plus one HF export |
| --- | ---: |
| 1.5B | **21.93 GiB** |
| 3B | **44.75 GiB** |

For rough budgeting, use 16.45 GiB per Levanter checkpoint and 5.49 GiB per HF
export at 1.5B; use 33.57 and 11.19 GiB respectively at 3B. For `N` permanent
native steps and one final HF export, budget `N * Levanter + HF`.

## Reference architectures

The model-size label alone is not enough to predict checkpoint size. These are
the concrete architectures behind the table:

| Property | 1.5B reference | 3B reference |
| --- | ---: | ---: |
| Trainable parameters | 1,471,374,336 | 3,003,164,160 |
| Transformer layers | 24 | 30 |
| Hidden dimension | 2,048 | 2,560 |
| Intermediate/FFN dimension | 8,192 | 10,240 |
| Query heads | 32 | 48 |
| KV heads | 8 | 16 |
| Head dimension | 64 | 64 |
| QK normalization | enabled | disabled |
| Maximum sequence length | 8,192 | 8,192 |
| Vocabulary size | 2,845 | 2,845 |
| Parameter/checkpoint dtype | FP32 | FP32 |
| Training compute/output dtype | BF16 | BF16 |

Both are Qwen3-style, use rotary position embeddings, and do not tie input and
output embeddings. The tokenizer is the contacts-v1 tokenizer at revision
`5d68a24a899f` (published under both the `timodonnell` and historical `eczech`
namespaces).

The 1.5B measurements span closely related checkpoints from experiments 75,
117, 166, and 199. The architecture above is the exp199 reference, whose
structured W&B metadata reports 1,471,374,336 parameters. The 3B measurements
are from experiment 146, whose metadata reports 3,003,164,160 parameters.

## What determines the size

For these runs, the measured sizes are almost entirely explained by parameter
count, precision, and optimizer state:

- An HF export contains FP32 model weights, so its payload is approximately
  `4 * parameter_count` bytes, plus small config and tokenizer files.
- A native Levanter AdamW checkpoint contains FP32 parameters and two FP32 Adam
  moment arrays, so its payload is approximately `12 * parameter_count` bytes,
  plus small training-state and TensorStore metadata.
- The resulting Levanter checkpoint is therefore about three times the size of
  the HF export.

The table will not transfer directly if any of these change:

- model width, depth, attention dimensions, FFN dimensions, or vocabulary size;
- tied versus untied embeddings;
- parameter or export precision (for example, a BF16 HF export should be about
  half the FP32 size);
- optimizer or optimizer-state precision (SGD, Adafactor, 8-bit Adam, and mixed
  precision states have different multipliers);
- extra saved state such as model averaging or additional optimizer slots.

Context length is 8,192 here, but it does not materially affect checkpoint size:
RoPE has no learned position-embedding table. Context length would matter for a
model with learned positional embeddings. Batch size, number of epochs,
gradient checkpointing, data mixture, and accelerator slice affect training but
not the persistent parameter/optimizer array shapes.

## Observed variation and anomalies

HF exports are effectively invariant. The complete 1.5B HF sample varied by
only 1,472 bytes, and every measured 3B HF export had exactly the same size.

Normal Levanter prefixes have small physical-layout variation:

- The 1.5B normal range spans about 17.8 MiB, or 0.11%.
- The 3B normal range spans about 3.6 MiB, or 0.01%.
- One apparently complete older exp75 checkpoint is 16.772 GiB, about 2% above
  the main 1.5B cluster. It has a complete, single-wave object set rather than
  the signature of an interrupted save; layout, sharding, or additional saved
  state may explain the difference. Treat it as exceptional for budgeting.

There are also prefixes that should not be treated as checkpoint-size samples:

- Seven 1.5B prefixes contain only an 80–81 byte `metadata.json`; these are
  incomplete stubs, not usable checkpoints.
- Two 3B prefixes consume 50.162 and 66.211 GiB. Each has two distinct write
  waves: an earlier partial save and a later complete ~33.56 GiB save. The
  excess is consistent with stale objects left by interrupted or resubmitted
  writes into the same step prefix, not a larger logical checkpoint.

Object-store usage can therefore exceed the logical checkpoint size when a step
prefix is reused. A fresh, unique step prefix is the cleanest unit for storage
accounting.

## Example checkpoint paths

These prefixes existed when this document was written. Each root is shown once;
append the listed suffix to inspect the native or HF form.

### 1.5B on GCS

Root:

```text
gs://marin-us-east1/protein-structure/MarinFold/exp199_optimize_contacts_v1_afdb_esm/checkpoints/protein/prot-exp199-cv1-s01-m1-p03-aug/2026.08.07.1
```

- Levanter: `checkpoints/step-72599`
- HF: `hf/step-72599`

### 1.5B on CoreWeave S3

Root:

```text
s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp199_continue_contacts_v1_cw/checkpoints/protein/prot-exp199-cw-cv1-p06-cool-s01/2026.08.14.1
```

- Levanter: `checkpoints/step-290400`
- HF: `hf/step-290400`

### 3B on GCS

Root:

```text
gs://marin-us-east1/checkpoints/protein/prot-exp146-cv1-s01-3b-e8-lr3p162e-3-wd0p4-bs256-us-east1/2026.07.23.01
```

- Levanter: `checkpoints/step-17839`
- HF: `hf/step-17839`

### 3B-class on CoreWeave S3

Root:

```text
s3://marin-us-east-02a/MarinFold/exp108_qwen_3b_contacts_v1/checkpoints/plm-exp108-cv1-3b-e16-lr1e-3-wd0p2-v3
```

- Levanter: `checkpoints/step-71311`
- HF: `hf/step-71311`

The CoreWeave example is exp108's approximately 2.78B-parameter, width-scaled
"3B-class" model, not exp146's exact 3.003B architecture. It is useful for
inspecting the same checkpoint forms on CoreWeave S3, but its byte size should
not be substituted for the exp146 3B row above.

## Measurement scope

The size survey grouped object bytes by permanent `checkpoints/step-*` and
`hf/step-*` prefixes. It covered 723 GCS prefixes from exp75, exp117, exp146,
exp166, and the original exp199 TRC namespace. Temporary checkpoint trees and
run-root metadata were excluded. Model identity and parameter counts came from
structured experiment configuration and W&B tags rather than parsing run IDs.
