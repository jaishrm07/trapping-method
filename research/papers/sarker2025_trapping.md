# Model Immunization by Trapping Harmful Finetuning

**Authors / venue / year**: Najibul Haque Sarker, Zaber Ibn Abdul Hakim, Alvi Md Ishmam, Chia-Wei Tang, Chris Thomas / NeurIPS 2025 Workshop on Lock-LLM / 2025
**Bib key**: sarker2025trapping
**Read for thread(s)**: 01, 04
**Read on**: 2026-05-02 (workshop list confirmation only; full content read in `/papers/`)

## TL;DR (3 sentences)

Workshop paper from the Thomas group at Virginia Tech that introduces "trap-induction" — an immunization objective that drives the model into a basin where harmful fine-tuning gets stuck — as an alternative to condition-number-based immunization (Zheng et al. ICML 2025) which the authors argue does not guarantee persistence under the harmful-FT trajectory. Reports favorable RFD (Relative Fine-Tuning Deviation) on linear-probing attacks. **Open weakness**: evaluation uses linear probing (RFD ≈ 47) but LoRA RFD is only ≈ 1.16 — i.e., the trap does not currently bound LoRA fine-tuning attacks, which is exactly the gap our extension targets.

## Why we read it

Thread 01 — the trapping objective is the foundation of our extension. Thread 04 — Sarker et al. is the OpenReview-confirmed Lock-LLM workshop entry (id `gfAn827WAW`), and we want to position our contribution relative to it precisely.

## Key claims (with location)

1. Distinction from condition-number framing: shape the *trajectory* of harmful fine-tuning, not just the local Hessian (§1, §3).
2. RFD metric: *relative* fine-tuning deviation as an extrinsic immunization-retainment measure (§4).
3. Outperforms curvature-only baselines (Zheng condition-number) on RFD on the linear-probing attack class (§5).
4. **Reported gap**: LoRA-attack RFD ≈ 1.16 (much weaker than linear-probing RFD ≈ 47) — explicit acknowledgement that LoRA is not yet defended (§5/§6, our reproduction).

## Methods we could borrow / discard

- **Borrow**: trap-induction as the core mechanism; extend with operator-conditioning to bound LoRA attackers.
- **Borrow**: RFD as the primary metric, but augment with per-operator breakdown and an attacker-budget axis (steps × lr × rank).
- **Discard**: linear-probing as the *primary* attack — the field needs the LoRA result to take this seriously (per Lermen 2024 evidence and TAR's LoRA hole).

## Open questions / disagreements

- Is the trap geometry *operator-specific* (shaped against linear-probing, fails for LoRA) or *operator-agnostic but weaker than LoRA's gradient direction*? Thread 01.
- How does Sarker's trapping relate to CTRAP (Yi et al. 2025) and SOPHON (Deng et al. 2024)? The workshop paper should make this explicit. Likely positioning: SOPHON predecessor → CTRAP collapse-trap variant → Sarker geometric-trap variant.

## Citation

NeurIPS 2025 Lock-LLM Workshop. OpenReview: https://openreview.net/forum?id=gfAn827WAW. Sarker, Hakim, Ishmam, Tang, Thomas.
