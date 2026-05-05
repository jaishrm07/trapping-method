# The Impact of Initialization on LoRA Finetuning Dynamics

**Authors / venue / year**: Soufiane Hayou, Nikhil Ghosh, Bin Yu / ICML 2024 / 2024
**Bib key**: hayou2024loraplus_init
**Read for thread(s)**: 02, 01
**Read on**: 2026-05-02

## TL;DR (3 sentences)

The paper analyses LoRA dynamics for the two natural inits: Init[A] = "A random, B=0" (the Hu et al. default) versus Init[B] = "A=0, B random". They prove these are *not* symmetric: Init[B] tolerates a larger learning rate while remaining stable, leading to better feature learning, while Init[A] is conservative but sub-optimal because B is undertrained. They give a per-matrix LR scaling rule (η_A ≠ η_B) — same family as LoRA+.

## Why we read it

Thread 02 — directly addresses the gradient asymmetry we hit. With Init[A] (which we and the Lermen attack both use) and η=0.1, the dynamics are predictably unstable; the paper explains *why* and gives a principled fix (different LRs for A and B, or switch init).

## Key claims (with location)

1. With B=0 init at η, the first-step update is O(η) on B and O(η²) on A — asymmetric (Sec. 3, Lem. 1).
2. Maximum stable LR for Init[A] is smaller than for Init[B] by a factor depending on width (Sec. 4, Thm. 2).
3. Optimal practice: η_B ≫ η_A (LoRA+ style ratio ~16x) (Sec. 5).

## Methods we could borrow / discard

- **Borrow**: when simulating the attacker's inner loop, parameterise η as (η_A, η_B) so we can sweep both Init[A] and Init[B] regimes. Defender should be robust to either.
- **Borrow as diagnostic**: if our inner loop gradient norm is dominated by ∂L/∂B at early steps and by ∂L/∂A later, that *signature* matches the predicted dynamics — a clean way to verify nothing else is broken.

## Open questions / disagreements

- Their analysis is per-step / per-matrix; doesn't directly prescribe the *meta-gradient* through k unrolled steps with `create_graph=True`. The asymmetry may compound differently in the bilevel setting.

## Citation

arXiv:2406.08447. Hayou, Ghosh, Yu (2024).
