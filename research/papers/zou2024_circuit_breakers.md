# Improving Alignment and Robustness with Circuit Breakers

**Authors / venue / year**: Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks / NeurIPS 2024 / 2024
**Bib key**: zou2024circuitbreakers
**Read for thread(s)**: 04
**Read on**: 2026-05-02

## TL;DR (3 sentences)

Circuit Breakers ("Representation Rerouting", RR) is the dominant *representation-level* defense from the Zou/Hendrycks lineage: at inference, harmful internal representations are pushed off-distribution so the model cannot complete the harmful generation. The defense is **not** a parameter-mod defense — it targets jailbreaks, adversarial-input attacks, and image hijacks, all of which assume the attacker cannot modify weights. Multiple follow-ups (Crescendo multi-turn jailbreak, "breaking circuit breakers" line of work) show even the inference-time threat model is fragile, and the defense increases over-refusal on benign prompts (4% → 38.5%).

## Why we read it

Thread 04 — circuit breakers are the most-cited Zou-lineage defense and frequently confused with tamper-resistance work (TAR). Need to nail down that they live in the *activation/representation* threat model, not the parameter-modification threat model, so we don't mis-position our work against them.

## Key claims (with location)

1. Mechanism: train the model so harmful representations are *rerouted* — moved away from coherent generation directions (Sec. 3, "Representation Rerouting" loss).
2. Threat model: jailbreak prompts and adversarial inputs at inference time; image hijacks for multimodal (Sec. 1, 4).
3. Claim: large reduction in attack success on Mistral-7B and Llama-3-8B against unseen attacks (Sec. 4).
4. **Not evaluated**: parameter-mod attacks. The paper does not test whether RR survives fine-tuning.
5. Known fragility: Crescendo and related multi-turn jailbreaks bypass RR (cf. Russinovich et al. 2024; "A Representation Engineering Perspective on the Effectiveness of Multi-Turn Jailbreaks", arXiv 2507.02956).

## Methods we could borrow / discard

- **Borrow**: nothing methodologically — different threat model. But the RR loss structure (push representations to a designated direction) is a useful contrast for explaining what *parameter-level* tamper-resistance like ours is doing differently: we shape the *loss landscape under fine-tuning operators*, not the *representations under inference-time inputs*.
- **Discard**: do not let reviewers conflate our work with circuit breakers. The paper must explicitly state that RR is not a parameter-mod defense.

## Open questions / disagreements

- The RR loss has been shown to be removable by fine-tuning (the very threat model TAR/trap targets). This is a known piece of folklore — worth confirming with citations if we use it as motivation.
- Over-refusal regression (4% → 38.5%) is the canonical "capability cost" failure mode of representation-level defenses. We should report a comparable benign-capability number for the trap method to show parameter-level defenses don't have this failure mode.

## Citation

arXiv:2406.04313. NeurIPS 2024. Zou, Phan, Wang, Duenas, Lin, Andriushchenko, Wang, Kolter, Fredrikson, Hendrycks.
