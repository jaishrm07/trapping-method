# Stage 10 Plan - Population Persistent B-only LoRA Attackers

Date: 2026-05-04

## Why v10 exists

v9a was a hard negative:

```text
single persistent B-only attacker
-> B-only RFD 0.32, signed RFD 0.02
-> LoRA-r8 RFD 0.39, signed RFD -0.08
```

The likely failure is attacker-state overfitting:

```text
the defender makes one live B/head state bad
but fresh B/head states still adapt normally
```

v10 tests whether a small population of persistent attackers gives the defender
a less brittle target.

## Core idea

Maintain several persistent attackers:

```text
attacker_j = fixed random A_j + trainable B_j + trainable head_j
```

Each attacker has:

```text
independent A seed
independent optimizer state
independent minibatch history
persistent B/head across defender steps
```

The defender blocks the strongest current attacker:

```text
L_block = softplus(log(C_H) - min_j CE_H(theta, attacker_j))
```

The `min_j` selects the attacker with lowest harmful CE on the defender batch.

## v10a initial config

```yaml
run_name: trap_v10a_population_bonly_resnet18_cars
num_attackers: 4
rank: 8
attackers_per_outer: 2
attacker_steps_per_selected: 1
attacker_optimizer: adamw
attacker_lr: 1.0e-3
lambda_block: 0.3
ce_threshold: null
iterations: 2500
eval_every: 0
```

Mid-run RIR is disabled because v9a showed it dominates wall time. v10a keeps
final RIR and final ImageNet top-1.

## Implemented

```text
experiments/run_immunization_v10_population.py
configs/immunize_v10a_population_bonly.yaml
```

Remote smoke test:

```text
role-lab GPU 3
run_name: smoke_v10a_population_bonly
iterations: 5
skip_final_eval: true
```

Result:

```text
results/smoke_v10a_population_bonly/extractor.pt
results/smoke_v10a_population_bonly/population_attackers.pt
results/smoke_v10a_population_bonly/results.json
```

100-iteration behavior smoke:

```text
run_name: smoke_v10a_population_bonly_100
step 0:  min_harm=7.102, block=0.150
step 50: min_harm=5.640, block=0.528
```

The population attackers began to lower harmful CE toward `log(196) ~= 5.278`,
activating the defender block.

## Full v10a run

Remote:

```text
role-lab GPU 3
logs/v10a_population_bonly_20260504_144545.out
logs/v10a_population_bonly_20260504_144545.err
```

Result:

```text
results/trap_v10a_population_bonly_resnet18_cars/
Final RIR: 0.657
Final primary ImageNet top-1: 56.54%
```

Last logged train state:

```text
step=2450
loss_primary=1.176
loss_block=1.604
loss_harm_min=3.899
attacker_acc_min_idx=0.219
```

Interpretation:

v10a created stronger population attacker pressure than v9a, but it failed the
primary utility requirement before post-hoc probing:

```text
primary target >= 63%
observed 56.54%
```

Probe matrix:

| Probe | Baseline final | v10a final | RFD | Signed RFD |
|---|---:|---:|---:|---:|
| linear_probe | 41.23% | 43.02% | 4.94 | -2.82 |
| lora_bonly_r8 | 51.19% | 50.75% | 2.48 | 2.48 |
| lora_r8 | 73.85% | 74.22% | 0.98 | 0.32 |
| lora_r32 | 78.15% | 78.87% | 0.81 | 0.50 |
| full_ft_upper | 80.76% | 80.05% | 0.75 | -0.01 |

Conclusion:

```text
population pressure slightly delays B-only,
does not block full LoRA,
and collapses primary utility.
```

Full report:

```text
results/STAGE10_v10a_REPORT.md
```

## Evaluation matrix

After full immunization:

```text
linear_probe
lora_bonly_r8
lora_r8
lora_r32
full_ft_upper
```

## Success criteria

Minimum useful:

```text
B-only LoRA-r8 signed RFD > 10%
primary ImageNet top-1 >= 63%
```

Strong:

```text
LoRA-r8 signed RFD > 3-5%
B-only signed RFD > 10%
primary ImageNet top-1 >= 63%
```

Hard negative:

```text
B-only signed RFD remains near zero
```

Then trajectory trapping against B/head states is probably the wrong target,
and the next direction should be representation-level adapter-reachable label
information removal.
