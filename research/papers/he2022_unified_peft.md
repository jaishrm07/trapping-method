# Towards a Unified View of Parameter-Efficient Transfer Learning

**Authors / venue / year**: Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig / ICLR 2022 / 2021-2022
**Bib key**: he2022unifiedpeft
**Read for thread(s)**: 04, 01
**Read on**: 2026-05-02

## TL;DR (3 sentences)

The paper shows that adapters, prefix-tuning, LoRA, and BitFit can be expressed in a single template Δh = f(x) where f differs only in the *form* of the modification (parallel vs. sequential, applied to attention vs. FFN) and the *insertion location*. From this view, they design "Mix-and-Match" hybrids that combine the best of each. For a defender, the corollary is critical: PEFT methods occupy a structured, finite-dimensional family — not a discrete set of techniques — so a defense can in principle target the family jointly.

## Why we read it

Thread 04 — answers "is anyone treating LoRA / BitFit / prefix-tuning / IA³ as a structured family rather than separate attacks?" Yes, but only for *capability* analysis, not defense. Suggests an angle for our work.

## Key claims (with location)

1. Unified form: each PEFT inserts Δh = α · Wf(x) at a hidden state, parameterised by W (Sec. 3, Tab. 1).
2. LoRA = parallel adapter without nonlinearity, applied to attention (Sec. 3.3).
3. Prefix-tuning ≡ a specific form of parallel adapter on attention (Sec. 3.2).
4. Best designs put parallel adapters on FFN with the prefix-style attention augmentation (Sec. 5).

## Methods we could borrow / discard

- **Borrow conceptually**: define the *defense subspace* as the union of insertion points × parameterisation forms in their template. Defender protection should generalise across the whole family, not just LoRA.
- **Borrow as threat model**: evaluate trap against IA³ and prefix-tuning attacks, not only LoRA — same compute-budget regime, different parameter location/form.

## Open questions / disagreements

- They don't characterise the full *intrinsic dimension* of the unified family. If the union spans a higher-d subspace than any single PEFT, the defender's burden is that union — connects directly to aghajanyan2020.

## Citation

arXiv:2110.04366. He, Zhou, Ma, Berg-Kirkpatrick, Neubig (2022).
