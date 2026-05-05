# Understanding and Robustifying Differentiable Architecture Search (RobustDARTS)

**Authors / venue / year**: Arber Zela, Thomas Elsken, Tonmoy Saikia, Yassine Marrakchi, Thomas Brox, Frank Hutter / ICLR 2020 / 2019-2020
**Bib key**: zela2020robustdarts
**Read for thread(s)**: 02
**Read on**: 2026-05-02

## TL;DR (3 sentences)

The paper diagnoses DARTS' collapse: as outer training progresses, the dominant eigenvalue λ_max(∇²_α L_val) explodes, correlating tightly with degenerate (skip-connection-dominated) found architectures. They propose DARTS-ES (early-stop when λ_max spikes) and stronger inner regularisation (data augmentation, L2 on weights, dropout) — both let DARTS find good architectures across all 12 of their NAS benchmarks. The "bilevel Hessian eigenvalue is the canary" insight is general.

## Why we read it

Thread 02 — gives us a *concrete diagnostic* (track λ_max of the outer Hessian during training) and *concrete fixes* (regularise the inner more, early-stop the outer). Generalises far beyond NAS.

## Key claims (with location)

1. λ_max(∇²_α L_val) rises monotonically through DARTS training; collapse correlates with rapid spike (Sec. 3, Fig. 4).
2. Early-stopping when λ_max spikes recovers good architectures (Sec. 4.1).
3. Stronger inner regularisation (L2, augment, dropout) flattens the outer loss landscape and prevents collapse (Sec. 4.2-4.3).

## Methods we could borrow / discard

- **Borrow diagnostic**: in our bilevel, periodically estimate λ_max(∇²_θ L_outer) via power iteration with HVP. If it spikes, snapshot weights and reset / early-stop / lower outer LR.
- **Borrow regulariser**: add explicit weight decay or L2-prox on the inner LoRA branch. This is *exactly* iMAML's regulariser (cf. rajeswaran2019_imaml note) — converging evidence that proximal regularisation of the inner is the right structural fix.
- **Borrow**: monitor the *ratio* outer-grad-norm / inner-loss; in our system that ratio diverged before collapse in Stage 5 v2 — same canary.

## Open questions / disagreements

- Adversarial bilevel: inner is maximising, so its Hessian is indefinite. λ_max of the outer Hessian still well-defined but interpretation changes (concave inner curvature *helps* the outer). Need to think carefully about sign conventions.

## Citation

arXiv:1909.09656. Zela, Elsken, Saikia, Marrakchi, Brox, Hutter (2020).
