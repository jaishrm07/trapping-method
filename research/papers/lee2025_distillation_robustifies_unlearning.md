# Distillation Robustifies Unlearning

**Authors / venue / year**: Bruce W. Lee, Addie Foote, Alex Infanger, Leni Shor, Harish Kamath, Jacob Goldman-Wetzler, Bryce Woodworth, Alex Cloud, Alexander Matt Turner / arXiv 2506.06278 / June 2025
**Bib key**: lee2025undo
**Read for thread(s)**: 04
**Read on**: 2026-05-02

## TL;DR (3 sentences)

Argues current LLM unlearning is fragile — a few fine-tuning steps reverse the unlearning — and proposes UNDO (Unlearn–Noise–Distill-on-Outputs): distill the unlearned model into a noised copy of itself, which transfers the *behavior* without transferring the *latent capability*. UNDO claims to match retrain-from-scratch robustness on WMDP at 60-80% of compute and labels for only 0.01% of pretraining data. This is the most direct unlearning analog to "tamper-resistance" — the unlearning equivalent of trap-induction in that the goal is *fine-tuning-resistance* of the unlearned state.

## Why we read it

Thread 04 — UNDO is the strongest current "unlearning that survives fine-tuning" defense; it's an alternative mechanism for the same end-goal (fine-tuning-resistant capability removal) and a natural compare-and-contrast for trap-method positioning.

## Key claims (with location)

1. Diagnosis: unlearning leaves latent capability that is recoverable by relearning attacks (§1, §3).
2. Method: distill unlearned-teacher into noised-student, transferring behavior but not the latent capability (§4).
3. Claim: UNDO matches "retrain from scratch with perfect data filtering" robustness, at 60-80% of compute and using only 0.01% of pretraining data labeled (§5).
4. Evaluated on WMDP (§5).

## Methods we could borrow / discard

- **Borrow**: distillation as a tamper-resistance amplifier. We could ask whether trap-induction's RFD survives distillation (i.e., distill the trapped model and see if the trap geometry transfers).
- **Borrow**: their "robustness against relearning attacks" metric as a complement to RFD. Specifically the protocol of fine-tuning the defended model on a small forget-set sample.
- **Discard**: the distillation pipeline itself is heavy and orthogonal to our parameter-level intervention.

## Open questions / disagreements

- Does UNDO survive *LoRA* relearning attacks specifically, or just full-FT? If it has the same LoRA hole, our LoRA-defense angle is competitive against *both* trap-induction and unlearning lineages.
- UNDO and trap-induction are mechanistically very different but functionally aligned — there might be a hybrid where UNDO's distillation amplifies trap geometry. Speculative.

## Citation

arXiv:2506.06278. Lee, Foote, Infanger, Shor, Kamath, Goldman-Wetzler, Woodworth, Cloud, Turner.
