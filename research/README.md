# Research Hub — Open-Weight Model Immunization

Living research workspace for the trapping-method line of work. Organized
around the **mechanism gap** we just observed empirically: linear-probe trap
defense works (RFD ≈ 50), the same trap against LoRA-r8 doesn't (RFD ≈ 1).
Operator-randomization (Plan C v2) cuts that gap by ~2× — first non-noise
LoRA defense — but is still tiny in absolute terms. More tuning won't close
this. We need to understand *why* the mechanism is so brittle across
operators, and what a defensible LoRA trap should look like in theory.

## Central question

> Can a single immunized backbone be made hard to fine-tune across an
> operator family (LP, LoRA-r, full-FT, …) simultaneously, and what does
> the trap loss need to look like to make that tractable?

## Sub-questions (drive the threads below)

1. What is the **right inductive bias** for a LoRA-aware trap? The paper's
   `softplus(ΔL_act − ΔL_exp)` exploits LP's quadratic geometry in `ω_H`.
   LoRA's parameter geometry is different (low-rank delta on conv tensors).
   Is there a tractable Hessian/predictor for the LoRA inner loop?
2. What's known about **stability of bilevel adversarial training** when
   the inner adversary is a non-trivial fine-tuner (LoRA, full-FT)?
3. Should the operator mixture be **uniform random per step (Plan C)**, or
   adversarially weighted (DRO / min-max over operators)?
4. What is the **current state of HFT-attack defenses** post-Lermen and
   post-shadow-alignment? Has anyone tried defending LoRA fine-tuning
   specifically (vs. just prompt-injection / activation-steering)?
5. What **diagnostics** would distinguish "trap signal exists but is weak"
   from "trap signal is fundamentally pointing the wrong way"?

## Threads

| # | Title | Type | Status |
|---|---|---|---|
| 01 | LoRA trap theory — closed-form predictor over rank-r subspace | theory | tested → predictor path broken via HVP cancellation; see thread |
| 02 | Bilevel adversarial training — stability literature | lit | done (Agent B 2026-05-02) |
| 03 | DRO / min-max over operator mixtures | theory + lit | tested → running-mean DRO null (v5a, 2026-05-03) |
| 04 | HFT defense landscape (post-2024) | lit | done (Agents A+B 2026-05-02); PEFT-family operator tested → NaN (v5b) |
| 05 | Diagnostics we should be running | empirical | open (D1-D5 still TODO) |
| 06 | First-principles reformulation: robust optimization in LoRA perturbation space | theory + design | superseded by thread 07 |
| 07 | v7 design — adversarial training in LoRA-r weight ball with ridge-LP inner | theory + impl | committed 2026-05-03; 2-week plan |

Each thread has its own MD under `threads/`. Status flips to `in_progress`
when work starts and `closed` when the thread converges (with a one-line
takeaway in this index).

## Empirical state

See [`empirical_state.md`](empirical_state.md) — what we know from the
experiments so far, distilled to facts (not speculation). Read this before
starting any thread; it's the constraint set.

## Paper notes

See [`papers/`](papers/). One MD per paper, named by `<lastname><year>_<slug>.md`.

## Conventions

- **Threads** are speculative — they can be wrong, half-finished, or
  abandoned. Mark them as such; don't delete.
- **Empirical claims** must cite a run name from
  `trapping-method/experiments/REGISTRY.md` or a paper note in `papers/`.
- **Decisions** that reshape the experimental plan get a one-line entry
  in `empirical_state.md` so we don't lose them.
- Everything is iterative — overwrite freely as understanding grows.
