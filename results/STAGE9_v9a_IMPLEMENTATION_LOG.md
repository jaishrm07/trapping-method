# Stage 9 v9a Implementation Log

Date: 2026-05-04

## Implemented

Added persistent B-only LoRA immunization:

```text
experiments/run_immunization_v9_persistent.py
configs/immunize_v9a_persistent_bonly.yaml
```

The attacker state persists across defender steps:

```text
A: fixed random LoRA basis
B: persistent trainable LoRA B factors
omega, bias: persistent harmful classifier head
```

Each outer iteration:

```text
1. attacker update:
   minimize harmful CE over B/head for attacker_steps_per_outer minibatches

2. defender update:
   minimize primary CE
   + condition-number regularizers
   + lambda_block * softplus(log(C_H) - CE_H(current persistent attacker))
```

## Initial config

```yaml
run_name: trap_v9a_persistent_bonly_resnet18_cars
rank: 8
attacker_steps_per_outer: 3
attacker_optimizer: adamw
attacker_lr: 1.0e-3
lambda_block: 0.3
ce_threshold: null
batch_size: 64
iterations: 2500
```

## Local validation

```text
python3 -m py_compile experiments/run_immunization_v9_persistent.py
yaml.safe_load(configs/immunize_v9a_persistent_bonly.yaml)
```

Both passed.

## Remote smoke tests

Server:

```text
role-lab
CUDA_VISIBLE_DEVICES=1
/home/jaisharma/miniconda3/envs/trap/bin/python
```

5-iteration graph smoke:

```bash
python experiments/run_immunization_v9_persistent.py \
  --config configs/immunize_v9a_persistent_bonly.yaml \
  --iterations 5 \
  --run-name smoke_v9a_persistent_bonly \
  --skip-final-eval
```

Result:

```text
Saved immunized extractor -> results/smoke_v9a_persistent_bonly/extractor.pt
Saved persistent attacker -> results/smoke_v9a_persistent_bonly/persistent_attacker.pt
```

100-iteration behavior smoke:

```bash
python experiments/run_immunization_v9_persistent.py \
  --config configs/immunize_v9a_persistent_bonly.yaml \
  --iterations 100 \
  --run-name smoke_v9a_persistent_bonly_100 \
  --skip-final-eval
```

Result:

```text
step 0-ish:  att_acc=0.000, harm=6.872, block=0.185
step 50-ish: att_acc=0.031, harm=5.044, block=0.817
```

This confirms the persistent attacker begins learning: harmful CE drops below
`log(196) ~= 5.278`, activating the defender block loss.

Output:

```text
results/smoke_v9a_persistent_bonly_100/extractor.pt
results/smoke_v9a_persistent_bonly_100/persistent_attacker.pt
results/smoke_v9a_persistent_bonly_100/results.json
```

## Full v9a run

Server:

```bash
cd ~/trapping-method
CUDA_VISIBLE_DEVICES=1 /home/jaisharma/miniconda3/envs/trap/bin/python \
  experiments/run_immunization_v9_persistent.py \
  --config configs/immunize_v9a_persistent_bonly.yaml
```

Remote log:

```text
logs/v9a_persistent_bonly_20260504_122919.out
logs/v9a_persistent_bonly_20260504_122919.err
```

Completed successfully.

Training/diagnostic checkpoints:

| Step | RIR | kappa_H immunized | kappa_P immunized |
|---:|---:|---:|---:|
| 500 | 0.549 | 2.195e10 | 3.087e10 |
| 1000 | 0.789 | 3.009e10 | 3.258e10 |
| 1500 | 0.721 | 2.865e10 | 3.623e10 |
| 2000 | 0.942 | 3.472e10 | 2.962e10 |

Final:

```text
Final RIR: 0.783
Final primary ImageNet top-1: 64.23%
Saved immunized extractor -> results/trap_v9a_persistent_bonly_resnet18_cars/extractor.pt
Saved persistent attacker -> results/trap_v9a_persistent_bonly_resnet18_cars/persistent_attacker.pt
Saved JSON -> results/trap_v9a_persistent_bonly_resnet18_cars/results.json
```

Last logged training state:

```text
step=2450
loss_primary=0.948
loss_block=0.022
loss_harm_current=9.091
attacker_acc_last=1.000
attacker_acc_defender_batch=0.3125
```

Interpretation:

The persistent attacker learned during its own minibatch steps, but the
defender's current-state blocking objective pushed the evaluated harmful CE
well above `log(196) ~= 5.278` by the end of training. This is not yet evidence
of real LoRA robustness; the post-hoc 50-epoch probes below are the decisive
test.

## Probe matrix

Launched on role-lab after the immunization artifacts were saved.

| Probe | GPU | Result dir | Final | Best |
|---|---:|---|---:|---:|
| linear_probe | 1 | `results/adv_v9a_linear_probe_cars/` | 42.48% | 43.35% |
| lora_bonly_r8 | 7 | `results/adv_v9a_lora_bonly_r8_cars/` | 51.18% | 51.22% |
| lora_r8 | 3 | `results/adv_v9a_lora_r8_cars/` | 73.32% | 74.06% |
| lora_r32 | 2 | `results/adv_v9a_lora_r32_cars/` | 78.76% | 78.76% |
| full_ft_upper | 1 | `results/adv_v9a_full_ft_upper_cars/` | 79.72% | 81.62% |

RFD against baseline:

| Probe | RFD | Signed RFD |
|---|---:|---:|
| linear_probe | 5.11 | -4.46 |
| lora_bonly_r8 | 0.32 | 0.02 |
| lora_r8 | 0.39 | -0.08 |
| lora_r32 | 0.41 | 0.07 |
| full_ft_upper | 0.68 | -0.16 |

Full report:

```text
results/STAGE9_v9a_REPORT.md
```
