# DARTS: Differentiable Architecture Search

**Authors / venue / year**: Hanxiao Liu, Karen Simonyan, Yiming Yang / ICLR 2019 / 2018-2019
**Bib key**: liu2019darts
**Read for thread(s)**: 02
**Read on**: 2026-05-02

## TL;DR (3 sentences)

DARTS treats NAS as a bilevel problem: outer optimises architecture α, inner optimises weights w*(α). Solved with a *one-step* unroll approximation: w'(α) ≈ w - η∇_w L_train(w, α). The paper notes (and follow-ups confirm) this is fragile — DARTS often collapses to degenerate skip-connection architectures driven by Hessian eigenvalue blow-up.

## Why we read it

Thread 02 — DARTS is the bilevel that most resembles ours: large network, gradient-through-unroll, known instability. The follow-ups (RobustDARTS, DARTS-, etc.) provide an empirical playbook for diagnosing and fixing bilevel collapse.

## Key claims (with location)

1. One-step unroll (k=1) approximation of the outer gradient (Sec. 2.4, Eq. 7).
2. Second-order term involves Hessian-vector product approximated by finite differences (Sec. 2.4, Eq. 8).
3. First-order DARTS (drop second-order term entirely) loses ~0.5pp accuracy but is much faster (Sec. 3).

## Methods we could borrow / discard

- **Borrow**: finite-difference HVP as a cheaper alternative to `create_graph=True`. ∇²f · v ≈ (∇f(θ+εv) - ∇f(θ-εv)) / (2ε). No graph storage.
- **Borrow from RobustDARTS (Zela 2020)**: monitor the dominant eigenvalue of the inner Hessian during outer training; early-stop the outer when it spikes.
- **Borrow from DARTS- (Chu 2021)**: regularise the inner objective so the bilevel doesn't admit degenerate equilibria.

## Open questions / disagreements

- DARTS uses k=1; we use k=3-10. Going to k=1 (or k=2) is a free stability win if our task tolerates it.

## Citation

arXiv:1806.09055. Liu, Simonyan, Yang (2019).
