# Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey

**Authors / venue / year**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu / arXiv (continuously updated; v6 April 2026) / 2024–2026
**Bib key**: huang2024hftsurvey
**Read for thread(s)**: 04
**Read on**: 2026-05-02

## TL;DR (3 sentences)

Living survey of harmful fine-tuning (HFT) attacks and defenses, organized into a four-stage taxonomy: pre-training, alignment-stage, fine-tuning-stage, post-fine-tuning. The survey groups defenses by *where in the pipeline* they intervene rather than by threat-model assumption, which under-emphasizes whether a defense actually withstands a parameter-modification adversary. Key open problems flagged in §5: robustness to hyper-parameter variation (the attacker's freedom over rank/lr/optimizer), generalization across attack scenarios, and mechanistic understanding of why defenses break.

## Why we read it

Thread 04 — establish the canonical defense taxonomy and locate where "trapping" fits. Also: the survey is curated by the Vaccine/Booster authors (Huang et al., git-disl), so it reflects how the dominant alignment-stage school positions the field.

## Key claims (with location)

1. Defense taxonomy (§4): pre-training (Deep Ignorance), alignment-stage (Vaccine, RepNoise, CTRL, TAR, Booster, T-Vaccine, CTRAP), fine-tuning-stage (LDIFS, Freeze, Constrain-SFT, ML-LR, Freeze+, SaLoRA, SafeInstr, VLGuard, Lisa, Paraphrase, BEA, PTST), post-fine-tuning (Security Vectors, Resta, LAT, SOMF, Safe LoRA, Antidote, SafetyLock, IRR, Panacea).
2. Attack taxonomy: explicit harmful-data, implicit/dual-use, identity-shift, backdoor; full-FT vs PEFT; with vs without guardrail moderation.
3. CTRAP (Yi et al. 2025) is identified as a "collapse-trap" defense — the closest neighbour to the Sarker et al. trapping framing this survey covers.
4. Open problems (§5): (a) defenses must be robust across hyper-parameter choices; (b) defenses need to generalize across attack scenarios (single-method robustness is insufficient); (c) mechanistic interpretability of why specific defenses break is largely missing.
5. Sarker et al.'s NeurIPS 2025 trapping paper does **not** appear in v6 (April 2026) — confirmed by absence of Sarker reference in defense list. Either still propagating or deemed adjacent to CTRAP.

## Methods we could borrow / discard

- **Borrow**: the four-stage taxonomy as the spine of our positioning section. Cite Huang's open problem (a) — robustness to attacker hyper-parameters — as direct motivation for the operator-aware/DRO angle in thread 03.
- **Discard**: their grouping by *pipeline stage* obscures the more important axis of "what threat model is actually defended." We should re-cut the field map by threat model (parameter-mod vs prompt vs activation) rather than stage.

## Open questions / disagreements

- The survey lists 30+ defenses but does not separate "evaluated against LoRA" from "evaluated against full-FT only." A LoRA-specific defended-vs-not-defended audit is missing — this is the gap the trap-method angle could fill.
- The survey stays neutral on whether defense is even *possible* against an unrestricted parameter-mod adversary. Compare with the more pessimistic Lermen et al. and Tamirisa et al. (LoRA breaks TAR) findings.

## Citation

arXiv:2409.18169 (v6, April 2026). Huang, Hu, Ilhan, Tekin, Liu.
