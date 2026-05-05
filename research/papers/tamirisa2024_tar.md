# Tamper-Resistant Safeguards for Open-Weight LLMs (TAR)

**Authors / venue / year**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zou, Mantas Mazeika, et al. / ICLR 2025 / 2024
**Bib key**: tamirisa2024tar
**Read for thread(s)**: 04
**Read on**: 2026-05-02

## TL;DR (3 sentences)

TAR builds tamper-resistant safeguards into open-weight LLMs by training the *base weights* such that adversarial fine-tuning (including LoRA) for hundreds of steps fails to recover the unsafe capability while benign capabilities are preserved. They use a meta-learning-style outer loop in which the inner solves an adversarial fine-tune; the outer maximises the adversary's *post-fine-tune* loss on harmful behaviour. This is the most directly comparable prior work to ours.

## Why we read it

Thread 04 — most relevant baseline / sibling work for our defense. Confirms (a) the field has moved to bilevel-adversarial-training defenses, (b) those defenses target the LoRA threat model directly, and (c) the architecture is meta-learning-shaped (so MAML stability tricks apply).

## Key claims (with location)

1. Meta-learning outer loop with k=64 inner adversarial fine-tune steps (Sec. 4.2).
2. Tamper-resistance survives 100s of fine-tune steps with various LoRA configs (Tab. 2).
3. Benign capability preserved (within 1-2% of base on MMLU) (Tab. 3).
4. Critically: they use FOMAML-style (no second-order through the unroll) for compute reasons (Appendix B).

## Methods we could borrow / discard

- **Borrow**: their FOMAML choice is converging evidence — both they and our reading of MAML literature say drop second-order. Strong signal we should too.
- **Borrow**: their k=64 vs our k=3-10 — they get stability with much *deeper* inner unroll precisely because they don't differentiate through it.
- **Borrow as baseline**: reproduce TAR and compare directly. Their public code (rishub-tamirisa/tamper-resistance on GitHub) makes this tractable.

## Open questions / disagreements

- Durability of safeguards has been questioned — see Qi et al. 2024 ("On Evaluating the Durability of Safeguards", arXiv:2412.07097), which broke several TAR-style defenses with stronger attacks. Need to read that critique before we claim TAR-comparability.
- **LoRA-evaluation gap (Agent A, 2026-05-02 update).** Cross-checking: of the "28 red-team adversaries" headlined in TAR, only 2 use PEFT/LoRA — the bulk of the tamper-resistance numbers come from full-FT attacks. The Huang HFT survey (arXiv 2409.18169 v6) and a 2025 MIT EECS thesis (Zhang) report TAR's safeguard "largely breaks" under LoRA fine-tuning configurations. So the original Tab. 2 LoRA claim is real but narrow (specific rank/lr/optimizer); broader LoRA sweep is not in the paper. This is the same LoRA-shaped hole the Sarker trapping paper has and that we are extending to fill. When citing TAR as a baseline we should report per-operator (full-FT vs LoRA-r-X) numbers, not the aggregate.

## Citation

arXiv:2408.00761. Tamirisa et al. (ICLR 2025).
