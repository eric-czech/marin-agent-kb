---
name: fleet-capacity
description: Region × TPU-slice fleet usage — the capacity dashboard's "Fleet Overview", from the CLI.
---

# Fleet Capacity

*What TPU do we have, where, and how much is busy?* The dashboard's Fleet Overview has
**no endpoint** — it aggregates in the browser. Rebuild it; don't hunt for one call.

## Source

Two controller RPCs, via the generic bridge exposing every controller method as a
kebab-case JSON subcommand. Names drift — re-discover:

```bash
iris --cluster marin rpc controller --help                  # current names
iris --cluster marin rpc controller get-autoscaler-status   # the fleet: inventory + capacity signal
iris --cluster marin rpc controller get-scheduler-state     # optional: who is running what
```

The autoscaler response is the whole answer except band attribution.

## Aggregation intent

- **Region and chip type are not fields.** Both are encoded in the scale group's name
  (family, slice size, zone). Parse loosely; strip the zone suffix for a region; keep
  unmatched names rather than dropping groups. Skip CPU.
- **Count slices, not VMs** — the slice is what users request. Ready ones only.
- **In use** = the slice has any task running.
- **Capacity is worst-wins per region.** Groups carry an availability signal (quota /
  max-slices / backoff-flavored). Report the most restrictive; pass unrecognized values
  through verbatim rather than assuming healthy.
- **Band mix** joins scheduler buckets to slices by worker. Give each slice one dominant
  band so shares partition rather than double-count.

Verify: card totals must sum to the ready non-CPU slice count.

## Target output

One card per chip type, largest first, split by region:

```
REGION TOTALS: europe-west4=132  us-central1=130  us-east5=84  us-west4=53  us-east1=2  us-central2=1

  96 v6e-4      100% in use   avg 4h 32m    73% batch  27% interactive
       europe-west4       76   [~] At TRC Capacity
       us-east5           18   [~] At TRC Capacity
       us-east1            2   [OK] Compute Potentially Available

  92 v5p-8      100% in use   avg 12h 54m   59% interactive  41% batch
       us-central1        70   [X] At Region Quota
       us-east5           22   [OK] Compute Potentially Available

  71 v5e-16     100% in use   avg 8h 32m    100% batch
       europe-west4       37   [~] At TRC Capacity
       us-west4           34   [~] At TRC Capacity

  51 v5p-16     100% in use   avg 8h 1m     98% batch  2% interactive
       us-central1        33   [X] At Region Quota
       us-east5           18   [OK] Compute Potentially Available

   8 v5p-64     100% in use   avg 9h 42m    75% batch  25% interactive
       us-east5            5   [OK] Compute Potentially Available
       us-central1         3   [X] At Region Quota

   1 v4-2048    100% in use   avg 4d 9h     100% production
       us-central2         1   [~] At Max Slices

402 ready slices, every class saturated. v5p in us-central1 is quota-blocked while
us-east5 has headroom; v6e-4 and v5e-16 are at TRC capacity in both their regions.
```
