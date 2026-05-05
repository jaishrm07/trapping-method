# Meta-Learning with Implicit Gradients (iMAML)

**Authors / venue / year**: Aravind Rajeswaran, Chelsea Finn, Sham Kakade, Sergey Levine / NeurIPS 2019 / 2019
**Bib key**: rajeswaran2019imaml
**Read for thread(s)**: 02
**Read on**: 2026-05-02

## TL;DR (3 sentences)

iMAML replaces the differentiate-through-unroll approach with an implicit-function-theorem (IFT) gradient at the *fixed point* of a regularised inner objective L_τ(φ) + (λ/2)‖φ-θ‖². The outer gradient becomes (I + (1/λ)∇²L)^{-1} ∇_φ L_outer, computed via conjugate gradient (Hessian-vector products only, no graph). Memory and stability are independent of inner step count k.

## Why we read it

Thread 02 — this is the *most stable* known way to do bilevel gradient when you can solve the inner problem to (approximate) optimality. Our LoRA inner adversary is small enough that running it to convergence and then doing one HVP is plausible. Eliminates the "k → outer gradient norm scales with k" failure mode entirely.

## Key claims (with location)

1. Implicit gradient: d_θ θ*(θ) = (I + (1/λ) H)^{-1} where H is inner Hessian at fixed point (Sec. 3, Eq. 4).
2. CG with t iterations gives approximation error O((1 - λ/(λ+L_max))^t) (Sec. 3.2).
3. Memory O(1) in inner steps vs O(k) for MAML (Sec. 3.3, Tab. 1).
4. Empirical match or improvement vs MAML on standard benchmarks (Sec. 4).

## Methods we could borrow / discard

- **Borrow**: if k=10 unrolled inner steps is unstable, replace with `inner.solve_to_convergence(); outer_grad = CG(I + H/λ, ∇_φ L_out, iters=10)`. Same compute, dramatically more stable, *and* well-defined (no path-dependence on inner LR schedule).
- **Borrow regulariser**: even if we keep unrolling, adding (λ/2)‖φ - θ‖² to the inner objective serves as a damping term that bounds the inner Hessian spectrum and hence the outer-gradient magnitude.

## Open questions / disagreements

- Our adversary is *adversarial* (inner *maximises*, not minimises). IFT machinery still applies to saddle points but extra care needed: the inner Hessian in maximisation has reversed sign, can be indefinite, and CG may not converge. Use damped Newton or Lanczos-truncated solves.
- For LoRA, the inner Hessian has a special block structure (A and B blocks); could exploit for cheaper HVP.

## Citation

arXiv:1909.04630. Rajeswaran, Finn, Kakade, Levine (2019).
