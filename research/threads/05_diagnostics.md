# Thread 05 — Diagnostics we should be running

**Status:** open
**Type:** empirical
**Owner:** —

## Question

What instrumentation would let us *see* whether the multi-op trap is
producing useful LoRA signal, separately from "is it producing LP
signal"? Right now we observe one trap value per step (sampled
operator) and one final RFD per probe. Everything in between is
hidden.

## Why this matters

If LoRA branch produces consistent useful gradient → we should expect
LoRA RFD to scale with k_inner (more inner steps = better fit = more
signal). If LoRA branch produces noise → no amount of k_inner helps
and we should pivot to a different formulation (thread 01 / 03).

We can't distinguish these from current logs.

## Diagnostics to add

### D1 — Per-operator trap value histogram

Instrument `trap_loss_multiop` to log `(operator, trap_value, step)`
tuples. Plot per-operator distributions over training. Cheap (one
extra log line per defender step).

**Predicts:** if LoRA branch is signal, mean LoRA trap value should
*decrease* over training as defender adapts. If LoRA branch is noise,
distribution stays flat.

### D2 — Per-operator gradient-norm contribution

Run two extra backwards per defender step (one with only LP trap,
one with only LoRA trap), record `||grad_θ||` for each. Tells us how
much each operator branch is actually moving the defender.

Cost: ~3× backward pass per step. Run a short v2 variant (500 steps)
to compute stats, not all 2500.

### D3 — Trap-vs-RFD correlation at intermediate checkpoints

Save extractor at steps {500, 1000, 1500, 2000, 2500} (or every 500),
run the 5-probe array on each. Plot RFD trajectory per operator.
Tells us *when* defense forms (early vs late) and whether multi-op
trap lags or leads on LoRA vs LP.

Cost: 5× the probe budget per immunization run. Big but bounded.

### D4 — Inner-loop convergence trace

Inside `trap_loss_lora`, log the inner harmful loss at each of the k
inner steps. We want to see *whether* the LoRA inner loop is actually
descending or just bouncing. If bouncing, η is wrong; if descending
but slow, k is too small; if descending then plateauing, that's the
right regime.

Cheap (k extra scalar logs per defender step).

### D5 — LoRA delta-weight magnitude

After each defender step, log `||B@A||_F` summed over conv layers in
the inner loop. Tells us how much the inner adversary actually moved
in weight space — independent of loss.

Predicts:
- ||delta|| stays near 0 → adversary not fitting (k or η too small).
- ||delta|| explodes → instability (we're back to the v2-attempt-1
  failure mode but pre-clamp).
- ||delta|| grows then stabilizes → healthy regime.

## Priority

Run all five — they're all cheap relative to a full immunization. D1
and D4 should be added immediately to `trap_loss_multiop` /
`trap_loss_lora` (one logging line each). D3 is the biggest insight
generator but slowest to run.

## Acceptance criterion

Thread closes when we have:
- D1, D4, D5 instrumented and run on Stage 5 v3 (already in flight as
  job 386356) or v4, **AND**
- A 1-paragraph diagnosis: which of (a)/(b)/(c) from
  `empirical_state.md` is the actual mechanism gap.
