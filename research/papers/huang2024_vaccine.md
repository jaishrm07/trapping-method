# Vaccine: Perturbation-aware Alignment for LLMs against Harmful Fine-tuning Attack

**Authors / venue / year**: Tiansheng Huang, Sihao Hu, Ling Liu / NeurIPS 2024 / 2024 (arXiv 2402.01109)
**Bib key**: huang2024vaccine
**Read for thread(s)**: 04
**Read on**: 2026-05-02

## TL;DR (3 sentences)

Vaccine is the first paper of the Huang/git-disl alignment-stage school: identifies a "harmful embedding drift" phenomenon where alignment breaks because a few harmful fine-tuning steps cause large drift in hidden representations, and proposes producing *invariant* hidden embeddings during alignment by adding adversarial perturbations to embeddings progressively. The defense is alignment-stage and pre-dates Booster (which is its evolution). Direct precursor to RepNoise, T-Vaccine, and Booster — together they form the dominant non-Hendrycks alignment-stage line.

## Why we read it

Thread 04 — Vaccine is the methodological starting point of the Huang school and frequently appears as a baseline in Booster, CTRAP, and the HFT survey.

## Key claims (with location)

1. Diagnosis: harmful embedding drift causes alignment break under HFT (§3).
2. Method: adversarial perturbation on embeddings during alignment, producing perturbation-invariant hidden states (§4).
3. Claim: improves robustness across Llama2/Opt/Vicuna while preserving reasoning (§5).
4. Evaluated against harmful-prompt-induced drift; multi-step LoRA stress test less prominent.

## Methods we could borrow / discard

- **Borrow**: nothing technical beyond the embedding-drift diagnosis, which is one of three or four credible mechanistic accounts of why HFT works. Useful for related-work positioning.
- **Discard**: superseded by Booster in the same lineage.

## Open questions / disagreements

- Vaccine is alignment-stage; trap-induction is also alignment-stage. The relationship is parallel: Vaccine modifies the *embedding manifold*, trap-induction modifies the *loss landscape*. Worth one comparison sentence in our paper.

## Citation

arXiv:2402.01109. NeurIPS 2024. Huang, Hu, Liu.
