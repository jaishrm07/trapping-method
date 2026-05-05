# On First-Order Meta-Learning Algorithms (Reptile / FOMAML)

**Authors / venue / year**: Alex Nichol, Joshua Achiam, John Schulman (OpenAI) / arXiv / 2018
**Bib key**: nichol2018reptile
**Read for thread(s)**: 02
**Read on**: 2026-05-02

## TL;DR (3 sentences)

The paper proves that FOMAML and Reptile (which never differentiates through the inner loop at all — outer update = θ - β(θ - φ_k) where φ_k is post-inner-loop weights) achieve similar performance to full MAML on standard meta-learning benchmarks. Their key analytical insight: the meta-gradient decomposes into a sum of inner-product terms between gradients at consecutive steps; both FOMAML and Reptile approximate this sum, just with different terms. For us, this means we can probably drop `create_graph=True` and lose little.

## Why we read it

Thread 02 — gives a principled justification for the cheaper / more stable first-order alternative. If our second-order MAML-style unroll is the proximate cause of gradient explosion, switching to Reptile-style "outer = movement to inner-loop end-point" eliminates the entire problematic backprop graph.

## Key claims (with location)

1. Reptile update: θ ← θ + ε(φ_k - θ), where φ_k is k inner SGD steps (Sec. 2, Alg. 1).
2. Taylor expansion shows Reptile's expected update is a weighted sum of gradient inner-products E[g_i · g_j] (Sec. 3, Thm. 1) — same form as FOMAML, different weighting.
3. Experimentally matches MAML within ~1pp on Omniglot/MiniImageNet with 4-10x faster training (Sec. 5).

## Methods we could borrow / discard

- **Borrow as primary fallback**: if FOMAML still unstable, drop to Reptile — completely sidesteps create_graph and second-order memory blow-up.
- **Borrow analytical lens**: the inner-product decomposition tells us *which inner step* contributes most variance; we can monitor per-step gradient inner-products as a diagnostic.

## Open questions / disagreements

- Reptile assumes the inner objective is *aligned* with the outer — for adversarial bilevel (inner = attacker, outer = defender), the inner end-point is *worst* for the outer. Need to flip sign in the Reptile-style update; the analytical result still applies but interpretation differs.

## Citation

arXiv:1803.02999. Nichol, Achiam, Schulman (2018).
