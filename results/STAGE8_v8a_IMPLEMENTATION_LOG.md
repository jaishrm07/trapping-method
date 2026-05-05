# Stage 8 v8a Implementation Log

Date: 2026-05-04

## Implemented

Stage 8 v8a is now scaffolded in code. The new operator targets the Stage 7
failure mode: fixed-A LoRA adapter-basis accessibility.

Code changes:

- `src/trap_loss.py`
  - Added `trap_loss_lora_bonly_ce_block`.
  - Added `lora_bonly_r<int>` dispatch to `trap_loss_multiop`.
  - Added local self-test coverage for direct B-only loss and multi-op B-only
    dispatch.
- `experiments/run_immunization_cn.py`
  - Added config wiring for:
    - `bonly_ce_threshold`
    - `bonly_inner_create_graph`
    - `bonly_detach_inner_updates`
- `configs/immunize_v8a_bonly.yaml`
  - Added first full v8a config.

## Objective

The B-only inner attacker freezes random LoRA A and trains:

```text
LoRA B + harmful classifier head
```

for `k` steps. The defender then receives:

```text
L_block = softplus(log(C_H) - CE_H(theta, B_k, head_k))
```

where `C_H` is the harmful-task class count. For Cars:

```text
C_H = 196
log(C_H) ~= 5.28
```

This penalizes successful harmful adaptation without unboundedly maximizing
cross-entropy.

## Initial config

```yaml
run_name: trap_v8a_bonly_resnet18_cars
trap_operators:
  - linear_probe
  - lora_bonly_r8
trap_k_inner: 10
trap_eta_inner: 0.01
lambda_trap: 0.3
bonly_ce_threshold: null
bonly_inner_create_graph: false
bonly_detach_inner_updates: true
```

## Validation

Local:

```text
python3 -m py_compile src/trap_loss.py experiments/run_immunization_cn.py experiments/run_adversary.py
python3 src/trap_loss.py
```

Passed:

```text
trap_loss_bonly  passed
trap_loss_multiop passed (... bonly=...)
```

Remote `role-lab`:

```text
CUDA_VISIBLE_DEVICES=5 python -m py_compile ...
CUDA_VISIBLE_DEVICES=5 python src/trap_loss.py
```

Passed.

Real ResNet/Cars mini-batch smoke:

```text
device=cuda loss=5.132852 grad_norm=1.065704 num_classes=196
```

This confirms the B-only v8 loss differentiates through the ResNet upper block
on the target server.

## Full run status

The full 2500-step v8a immunization and probe matrix are complete.

Executed command:

```bash
cd ~/trapping-method
CUDA_VISIBLE_DEVICES=1 /home/jaisharma/miniconda3/envs/trap/bin/python \
  experiments/run_immunization_cn.py \
  --config configs/immunize_v8a_bonly.yaml
```

Probe matrix:

```text
linear_probe
lora_bonly_r8
lora_r8
lora_r32
full_ft_upper
```

Final report:

```text
results/STAGE8_v8a_REPORT.md
```
