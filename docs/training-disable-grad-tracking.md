# Disable Grad/Param Tracking

Per-parameter gradient/weight tracking (Levanter's `WatchConfig`) logs a lot of data to
W&B and can cost non-trivial compute (and HBM pressure). To disable it, set an empty
watch on the train config's trainer:

```python
WatchConfig(watch_targets=[], interval=0)
```

e.g. `dataclasses.replace(trainer, watch=WatchConfig(watch_targets=[], interval=0))`
(`WatchConfig` lives in `levanter.callbacks.watch`).
