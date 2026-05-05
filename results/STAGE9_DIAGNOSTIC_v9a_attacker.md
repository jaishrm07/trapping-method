# Stage 9 Diagnostic - Saved Persistent Attacker Transfer

Date: 2026-05-04

## Question

v9a made the live persistent attacker look bad under CE during immunization, but
fresh post-hoc probes adapted normally. This diagnostic asks:

```text
Did v9a damage the adapter-accessible Cars signal,
or only the one saved B/head attacker state?
```

## Setup

Checkpoint:

```text
results/trap_v9a_persistent_bonly_resnet18_cars/extractor.pt
results/trap_v9a_persistent_bonly_resnet18_cars/persistent_attacker.pt
```

Script:

```text
experiments/run_v9_attacker_diagnostic.py
```

Run:

```text
role-lab GPU 1
results/diagnose_v9a_persistent_attacker/results.json
logs/diagnose_v9a_persistent_attacker.out
logs/diagnose_v9a_persistent_attacker.err
```

## Saved Attacker Evaluation

| Split | Accuracy | CE |
|---|---:|---:|
| Cars train | 41.18% | 6.122 |
| Cars test | 36.55% | 6.788 |

`log(196) ~= 5.278`, so the saved attacker has high CE even though it still
contains nontrivial class signal. This means v9a did not make the live attacker
purely random; it made it poorly calibrated/high-loss.

## Fresh Restarts

Each restart trains B/head for 10 epochs on the v9a extractor.

### Saved A, reset B/head

| Seed | Epoch-10 Cars test acc |
|---:|---:|
| 42 | 34.71% |
| 43 | 32.78% |
| 44 | 33.59% |
| mean | 33.69% |

### Fresh A, reset B/head

| Seed | Epoch-10 Cars test acc |
|---:|---:|
| 42 | 34.57% |
| 43 | 33.76% |
| 44 | 33.48% |
| mean | 33.94% |

## Interpretation

The saved A basis is not damaged. New B/head states recover rapidly on both:

```text
saved A
fresh random A
```

Therefore v9a failed because it overfit one live attacker state, not because it
removed Cars information from the adapter-reachable feature space.

This supports the v10 population/replay direction, but also raises the bar:
any population method must generalize across fresh B/head optimization paths,
not merely keep several current attackers high-loss.

