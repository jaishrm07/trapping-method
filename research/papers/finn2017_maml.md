# Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks

**Authors / venue / year**: Chelsea Finn, Pieter Abbeel, Sergey Levine / ICML 2017 / 2017
**Bib key**: finn2017maml
**Read for thread(s)**: 02
**Read on**: 2026-05-02

## TL;DR (3 sentences)

MAML learns an initialisation θ such that one (or a few) SGD steps on a new task yield good performance: outer loss is L_task(θ - α∇L_task(θ)) averaged over tasks. The outer gradient flows *through* the inner SGD step (second-order), so requires `create_graph=True` and Hessian-vector products. First-order MAML (FOMAML) drops this and uses only the *post-step* gradient — much cheaper, only marginal accuracy loss for many tasks.

## Why we read it

Thread 02 — our setup is structurally MAML-shaped (inner SGD on a parameterised adversary, outer gradient through the unroll). The original paper documents the second-order vs first-order tradeoff that we need to make.

## Key claims (with location)

1. Inner update: θ' = θ - α∇_θ L_τ(θ); outer: ∇_θ Σ_τ L_τ(θ' (θ)) (Sec. 2.2, Eq. 1-2).
2. FOMAML approximation drops the term involving the inner Hessian — small accuracy hit, large compute saving (Sec. 5.2 ablation).
3. Inner step count k=1 suffices for many tasks (Sec. 5.1); they report up to k=5.

## Methods we could borrow / discard

- **Borrow**: try FOMAML for the LoRA branch first; only escalate to full second-order if the first-order signal is too weak. Cheap diagnostic.
- **Borrow**: keep inner k small (1-3) and only ramp later — many MAML papers report diminishing returns past k=5.

## Open questions / disagreements

- The paper uses inner α ~ 1e-2 to 1e-3 on small CNN problems. Our η = 0.1 on a *much larger* model is highly suspect by comparison — likely the proximate cause of our outer-gradient explosions.

## Citation

arXiv:1703.03400. Finn, Abbeel, Levine (2017).
