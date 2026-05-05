# Model Immunization from a Condition Number Perspective

**Authors / venue / year**: Amber Yijia Zheng, Cedar Site Bai, Brian Bullins, Raymond A. Yeh / ICML 2025 (Oral) / 2025 (arXiv 2505.23760)
**Bib key**: zheng2025condnum
**Read for thread(s)**: 01, 04
**Read on**: 2026-05-02

## TL;DR (3 sentences)

ICML 2025 oral that formalizes model immunization through the *condition number of the Hessian*: well-condition the desired (utility) directions, ill-condition the restricted (harmful) directions, so that gradient-based fine-tuning on the restricted task converges slowly while benign fine-tuning is unaffected. Builds a clean framework for linear models with regularization terms that control the resulting condition numbers. The defense is curvature-based and local — Sarker et al. critique it for not guaranteeing trajectory-level persistence under multi-step harmful fine-tuning, which is the gap their trap-induction targets.

## Why we read it

Thread 01 — condition-number immunization is one of the two methods we are extending, and the natural first-order baseline against which trap-induction is compared. Thread 04 — same authors' IMMA paper (ECCV 2024) is the most direct LoRA-defense precedent, so the lineage matters.

## Key claims (with location)

1. Framework: bound condition number of Hessian along utility / restricted directions; control via regularization (§3, §4).
2. Algorithm: alternating minimization to satisfy both well-conditioned and ill-conditioned subspaces (§4).
3. Theory: linear-model setting with provable bounds (§4).
4. Empirical: outperforms baselines on linear and small-network tasks (§5).
5. **Limitation**: local curvature analysis does not bound multi-step harmful fine-tuning trajectories — a point Sarker et al. emphasize in their trapping paper.

## Methods we could borrow / discard

- **Borrow**: condition-number bounds as a *local* metric to combine with trap-induction's *trajectory* metric. Stack the two: trap-induction commits the basin, condition-number bounds guarantee local resistance.
- **Borrow**: the linear-model theory as the analytic backbone — anything we prove about LoRA-aware trap-induction should reduce to a CN statement in the linear case.
- **Discard**: alone, CN is insufficient under multi-step LoRA fine-tuning (per Sarker reproduction). Don't claim it as a standalone defense.

## Open questions / disagreements

- Does CN-immunization extend to LoRA *natively* — i.e., conditioning the Hessian along the LoRA-reachable subspace instead of the full parameter space? This is the candidate operator-aware extension and is exactly the question thread 01 asks.

## Citation

arXiv:2505.23760. ICML 2025 Oral. Zheng, Bai, Bullins, Yeh.
