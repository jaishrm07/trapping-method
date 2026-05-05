# Obfuscated Gradients Give a False Sense of Security

**Authors / venue / year**: Anish Athalye, Nicholas Carlini, David Wagner / ICML 2018 (best paper) / 2018
**Bib key**: athalye2018obfuscated
**Read for thread(s)**: 02
**Read on**: 2026-05-02

## TL;DR (3 sentences)

The paper identifies "obfuscated gradients" — defenses whose gradients become uninformative (shattered, stochastic, or vanishing/exploding) — as a common, deceptive failure mode that *appears* to defeat gradient-based attacks but doesn't. They show 7 of 9 ICLR 2018 defenses fall into this category and break with adapted attacks (BPDA, EOT). The taxonomy of failure modes (shattered, stochastic, vanishing/exploding) is the standard checklist for new defenses.

## Why we read it

Thread 02 — our trap defense could *easily* induce an exploding-gradient regime that looks like a strong defense but is actually gradient masking against the LoRA inner adversary. We need to sanity-check that our trap is *robust to gradient-free / black-box / surrogate-gradient* attacks, not merely the specific PGD/LoRA we trained against.

## Key claims (with location)

1. Three modes of obfuscation: shattered (non-differentiable preprocessing), stochastic (random masking), vanishing/exploding (deep iterative defenses) (Sec. 2).
2. **Vanishing/exploding gradients in iterative defenses** (Sec. 4.2) — directly relevant: any defense whose forward pass involves k iterations risks this. *Bilevel adversarial training is exactly the inverse: if the defender trains by k inner adversary steps, the resulting model can be in an exploding-gradient regime that breaks the attacker's gradient signal.*
3. Diagnostic checklist: (a) random noise > optimised attack ⇒ shattered; (b) increasing distortion budget doesn't increase attack success ⇒ vanishing; (c) unbounded distortion still fails ⇒ probably gradient masking (Sec. 5).

## Methods we could borrow / discard

- **Borrow as red-team**: evaluate our trap with (i) BPDA-style attacks on the LoRA branch, (ii) random LoRA initialisations to test if attacker just got unlucky finding the gradient direction, (iii) much higher attack budget than training-time.
- **Borrow checklist**: if our defense reports "trap holds against LoRA k=10" but breaks against k=100, that's classic gradient masking — the defense exploits the attacker's exact step count.
- **Borrow**: ensure the defender is *not* hiding behind the inner-loop unrolling depth used at training; sweep test-time k separately.

## Open questions / disagreements

- Most obfuscated-gradient examples are about *input perturbations*. Our setting (parameter perturbations via LoRA) is different — but the failure mode "trained against k-step adversary, breaks against (10k)-step adversary" is structurally identical.

## Citation

arXiv:1802.00420. Athalye, Carlini, Wagner (2018).
