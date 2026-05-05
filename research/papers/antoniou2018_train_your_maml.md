# How to Train Your MAML (MAML++)

**Authors / venue / year**: Antreas Antoniou, Harrison Edwards, Amos Storkey / ICLR 2019 / 2018
**Bib key**: antoniou2018trainmaml
**Read for thread(s)**: 02
**Read on**: 2026-05-02

## TL;DR (3 sentences)

The paper diagnoses MAML's training instability and proposes "MAML++": (1) per-layer per-step learnable inner LRs (LSLR), (2) multi-step *outer* loss (sum the outer loss after each inner step, not just the last), (3) cosine-annealed meta-LR, (4) per-step BatchNorm running stats, (5) gradient instability fixes from second-order term truncation. Stability and final accuracy both improve substantially. Most of these tricks are directly applicable to bilevel adversarial training.

## Why we read it

Thread 02 — this is the practical "MAML stability cookbook". Several tricks (especially LSLR and multi-step loss) directly target the failure mode we observe.

## Key claims (with location)

1. LSLR — each (layer, inner-step) gets its own learnable α; meta-learnt jointly with θ (Sec. 4.4).
2. Multi-Step Loss (MSL): outer loss = Σ_i w_i L_outer(φ_i) for i = 1..k, with annealed weights (Sec. 4.2). This *propagates outer gradient through every inner step*, not just the last — empirically much more stable.
3. Derivative-Order Annealing: use FOMAML for first ~50 epochs, switch to second-order for the rest (Sec. 4.3).
4. Cosine-annealed outer LR (Sec. 4.5).

## Methods we could borrow / discard

- **Borrow MSL**: instead of computing outer loss only at φ_k, compute it at every φ_i and sum with annealed weights. This *evens out* the contribution of each inner step to the outer gradient, killing the "k=10 → step-10 gradient is enormous" pattern.
- **Borrow LSLR**: meta-learn (η_A, η_B) per inner step. Lets the system *learn* its own stable inner-loop schedule rather than forcing η=0.1 fixed.
- **Borrow Derivative-Order Annealing**: warm up with FOMAML, switch to second-order once outer loss is in a stable regime.
- **Discard for now**: per-step BatchNorm running stats — we don't have BN in our LoRA branch (no BN in attention).

## Open questions / disagreements

- LSLR adds parameters to learn; for our bilevel setting where the inner adversary is *adversarial*, learning η would mean the *defender* picks η, not the attacker. That's a wrong threat model — we need η to be sampled from the attacker's plausible distribution, not optimised by us. Use LSLR as a *meta-trained schedule of attack LRs*, then test on out-of-distribution η at eval.

## Citation

arXiv:1810.09502. Antoniou, Edwards, Storkey (2019).
