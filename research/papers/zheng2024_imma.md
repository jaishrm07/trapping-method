# IMMA: Immunizing text-to-image Models against Malicious Adaptation

**Authors / venue / year**: Amber Yijia Zheng, Raymond A. Yeh / ECCV 2024 / 2024 (arXiv 2311.18815)
**Bib key**: zheng2024imma
**Read for thread(s)**: 04
**Read on**: 2026-05-02

## TL;DR (3 sentences)

IMMA is the *vision* counterpart to Zheng's later condition-number immunization for LLMs: a meta-learning-style outer loop that learns model parameters which are difficult to adapt via three concrete diffusion-fine-tuning operators (LoRA, Textual Inversion, DreamBooth). It is one of the few defenses in any modality that explicitly defends against **LoRA fine-tuning** in its primary evaluation. The framing — defend against a known adaptation operator, not against arbitrary fine-tuning — is the operator-aware framing thread 01 needs and is closer to the LoRA-specific defense angle than most LLM-side work.

## Why we read it

Thread 04 — IMMA is the closest demonstrated LoRA-defense in the literature (vision side). Thread 01 — IMMA's operator-conditioned objective is methodologically the right shape for what we want to do for LLMs.

## Key claims (with location)

1. Threat model: malicious adaptation of an open-source text-to-image model via fine-tuning (§1, §3).
2. Method: meta-learning-style objective optimizing parameters that are hard to fine-tune via the specified adaptation operator (§3, §4).
3. **Evaluated against three operators including LoRA, Textual Inversion, DreamBooth** (§5).
4. Claim: effective against mimicking artistic style and learning unauthorized content (§5).

## Methods we could borrow / discard

- **Borrow**: the operator-conditioned defense framing. This is the cleanest precedent for "defense against LoRA specifically" — explicitly cite IMMA as proof-of-concept that operator-aware defense is feasible.
- **Borrow**: the multi-operator evaluation protocol. Report defense efficacy *per operator* (LoRA-r-X, Textual Inversion, DreamBooth analogs), not aggregated.
- **Discard**: the diffusion-specific machinery; the conceptual structure is what transfers.

## Open questions / disagreements

- IMMA's results are on small-scale diffusion fine-tuning. Whether the operator-conditioned approach scales to LLMs and to LoRA-rank variation is open — exactly the question we're asking.
- IMMA + Sarker's trapping is a natural combination (operator-aware geometric trap) and not yet explored.

## Citation

arXiv:2311.18815. ECCV 2024. Zheng, Yeh.
