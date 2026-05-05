# Booster: Tackling Harmful Fine-tuning for LLMs via Attenuating Harmful Perturbation

**Authors / venue / year**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu / ICLR 2025 (Oral) / 2024-2025 (arXiv 2409.01586)
**Bib key**: huang2025booster
**Read for thread(s)**: 04
**Read on**: 2026-05-02

## TL;DR (3 sentences)

Booster is an alignment-stage defense that adds a regularizer ensuring harmful loss does not decrease much under a *simulated* one-step harmful perturbation; it is essentially TAR/RepNoise simplified to a one-step inner loop with a perturbation-attenuation interpretation. The paper trains and evaluates on LoRA (rank 32, α 4) — making it one of the few alignment-stage defenses that demonstrates results under PEFT in its primary experiments rather than only full-FT. ICLR 2025 Oral, and the same group publishes the dominant HFT survey, so this is the centroid of the alignment-stage school.

## Why we read it

Thread 04 — Booster is the strongest claimed alignment-stage defense as of ICLR 2025 and is the most direct competitor to a trap-method paper that targets the same threat model. Also: Booster's first-order simplification of TAR is a useful structural reference for explaining why higher-order trap-induction or condition-number control might add or not add value.

## Key claims (with location)

1. Diagnosis: harmful perturbation over weights causes alignment break (§3).
2. Method: alignment-stage loss + regularizer that simulates one harmful gradient step and penalizes the loss reduction along it (§4).
3. Eval setup: LoRA rank 32, α=4 — the experiments that ground the defense are PEFT (§5).
4. Claim: outperforms Vaccine, RepNoise, TAR-style baselines on harmful score while preserving downstream task performance (§5 tables).
5. **Caveat**: Booster targets a single inner step and a single attack distribution; robustness to multi-step / variable-rank attackers is under-evaluated.

## Methods we could borrow / discard

- **Borrow**: the LoRA-native evaluation protocol. Booster's headline is computed over LoRA fine-tuning, which is exactly the operator class we need to defend against — adopt the same setup as a baseline.
- **Borrow**: the one-step attenuation framing as the "first-order" point on a hierarchy whose higher-order points are condition-number (Zheng) and trap-induction (Sarker). Cleanly positions our work.
- **Discard**: the one-step simulation is genuinely weaker than what we want — we should explicitly contrast with trajectory-style trap induction.

## Open questions / disagreements

- Does Booster's defense survive LoRA-rank variation (8/16/32/64) and lr variation? Their grid is narrow.
- Is Booster's "attenuation" mathematically equivalent to a particular condition-number bound? Worth checking against thread 01.

## Citation

arXiv:2409.01586. ICLR 2025 Oral. Huang, Hu, Ilhan, Tekin, Liu.
