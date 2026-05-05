# Representation Noising: A Defence Mechanism Against Harmful Finetuning

**Authors / venue / year**: Domenic Rosati, Jan Wehner, Kai Williams, Łukasz Bartoszcze, Robie Gonzales, Carsten Maple, Subhabrata Majumdar, Hassan Sajjad, Frank Rudzicz / NeurIPS 2024 / 2024 (arXiv 2405.14577)
**Bib key**: rosati2024repnoise
**Read for thread(s)**: 04
**Read on**: 2026-05-02

## TL;DR (3 sentences)

RepNoise is the representative *alignment-stage* defense in the parameter-mod threat model: a three-part loss that (i) reduces predictive information about harmful outputs in the weights, (ii) preserves general capability, (iii) pushes harmful representations toward random noise. The paper claims the defense generalizes across unseen subsets of the same harm distribution and that the efficacy depends on "depth" — how many layers of harmful information have been removed. RepNoise is in the same lineage as TAR/Booster and reports resistance to harmful fine-tuning, but does not include a clean LoRA-rank ablation in the headline numbers, and Huang et al.'s survey notes the defense is "removable" once the attacker varies hyperparameters.

## Why we read it

Thread 04 — RepNoise is the canonical alignment-stage parameter-mod defense from outside the Hendrycks lineage; it's the methodological precursor that Booster ("RepNoise minus the representation-loss term plus harmful gradient attenuation") explicitly builds on. Knowing it sets the baseline for what alignment-stage parameter-mod defenses claim and demonstrate.

## Key claims (with location)

1. Threat model: open-weight, attacker has weights and fine-tunes (Sec. 1).
2. Loss: standard alignment loss + harmful-info noising on internal representations + capability preservation (Sec. 3).
3. Claim: generalizes to unseen harmful subsets drawn from the same distribution (Sec. 4).
4. Claim: efficacy correlates with how *deep* into the network the noising propagates (Sec. 4 ablations).
5. **Caveat (from follow-ups)**: subsequent work (Booster paper, Huang survey) reports RepNoise is sensitive to attack hyperparameters and partially recovered by sufficient fine-tuning steps.

## Methods we could borrow / discard

- **Borrow**: the depth-of-defense framing — argue that any *parameter-level* tamper-resistance defense should report a depth profile (which layers' tamper-resistance contributes how much). The trap method should adopt this analysis.
- **Borrow**: the generalization-across-harm-subsets evaluation. We should report whether trap-induction trained against attack distribution P generalizes to attack distribution Q ≠ P (this is also where the DRO angle in thread 03 lands).
- **Discard**: noise-the-representations is mechanistically distant from trap-induction; not a direct method to lift.

## Open questions / disagreements

- RepNoise's "generalization across harm distribution" claim is the natural place to test our hypothesis that operator-class generalization (not just data-distribution generalization) is the harder problem.
- No explicit LoRA breakdown in the original paper. Worth replicating their setup with LoRA to see if RepNoise has the same LoRA hole as TAR.

## Citation

arXiv:2405.14577. NeurIPS 2024. Rosati, Wehner, Williams, Bartoszcze, Gonzales, Maple, Majumdar, Sajjad, Rudzicz.
