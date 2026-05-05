# SaLoRA: Safety-Alignment Preserved Low-Rank Adaptation

**Authors / venue / year**: Mingjie Li, Wai Man Si, Michael Backes, Yang Zhang, Yisen Wang / arXiv 2501.01765 / January 2025; OpenReview venue indeterminate at fetch time
**Bib key**: li2025salora
**Read for thread(s)**: 04
**Read on**: 2026-05-02

## TL;DR (3 sentences)

SaLoRA inserts a fixed safety module (computed from safety data) and a task-specific initialization for trainable LoRA parameters so that *legitimate* downstream LoRA fine-tuning does not erode safety alignment. **Important threat-model distinction**: SaLoRA defends against *unintentional* safety degradation by the user — not against an adversarial parameter-mod attacker who deliberately fine-tunes on harmful data. So despite the name, this is not a tamper-resistance defense; it is a fine-tuning-stage hygiene tool.

## Why we read it

Thread 04 — SaLoRA shows up adjacent to "LoRA defense" in search results and the Huang survey lists it under fine-tuning-stage defenses; we need to be clear on the boundary between SaLoRA-class (preserve alignment under benign LoRA) vs. trap-class (resist adversarial LoRA).

## Key claims (with location)

1. Threat model: user fine-tunes via LoRA on a downstream task; safety alignment may erode unintentionally (§1, §2 — the framing is "preserved alignment", not "tamper resistance").
2. Method: fixed safety module + task-specific LoRA init, both derived from safety data (§3).
3. Claim: outperforms adapter-based baselines on alignment-preservation × downstream-task tradeoff (§5).
4. **Not evaluated**: adversarial harmful-data fine-tuning by an attacker. SaLoRA is not designed for that threat model.

## Methods we could borrow / discard

- **Borrow**: the technical idea that LoRA's safety properties depend on initialization and on which subspace is trainable. This is *exactly* the projector/operator-conditioning insight thread 01 needs. SaLoRA constrains the LoRA subspace to a "safety-aligned" projection — analogous to how we'd want the trap geometry to dominate the LoRA-reachable subspace.
- **Discard**: do not cite SaLoRA as a tamper-resistance defense — it is not. Cite it as evidence that LoRA fine-tuning has structural levers that can be controlled.

## Open questions / disagreements

- Could SaLoRA's fixed-safety-module design be combined with trap-induction so that the trap's geometry is enforced on the LoRA-reachable subspace specifically? This is a candidate operator-aware extension worth exploring.
- The Huang survey lists SaLoRA in the fine-tuning-stage column. Whether SaLoRA is run by the *defender* (alignment-stage) or the *user* (fine-tuning-stage) matters for how we cite it.

## Citation

arXiv:2501.01765. Li, Si, Backes, Zhang, Wang.
