# Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning

**Authors / venue / year**: Armen Aghajanyan, Luke Zettlemoyer, Sonal Gupta / ACL 2021 (arXiv 2020) / 2020
**Bib key**: aghajanyan2020intrinsic
**Read for thread(s)**: 01, 04
**Read on**: 2026-05-02

## TL;DR (3 sentences)

The paper measures the *intrinsic dimension* d_90 of a fine-tuning task — the smallest random subspace dimension that recovers 90% of full fine-tuning performance — and finds it shockingly small: ~200 for RoBERTa-base on MRPC, growing sublinearly with model size. Larger pretrained models have *lower* intrinsic dimension, which explains why PEFT (and LoRA in particular) works at all. This was the conceptual ancestor of LoRA: if real fine-tuning lives in a low-dim subspace, parameterise the update as one.

## Why we read it

Thread 04 (defense landscape, §5) and thread 01: the intrinsic dim of fine-tuning *constrains* the defender's design space. If a malicious task has d_90 ≈ k (small), the trap only needs to block a k-dim subspace of update directions per layer — but the *direction* of that subspace is unknown a priori, so the defender must defend against the union over tasks.

## Key claims (with location)

1. d_90(RoBERTa-base, MRPC) ≈ 200; d_90(BART-large, MRPC) ≈ 250 (Tab. 1, Fig. 2).
2. Intrinsic dim *decreases* monotonically through pre-training (Sec. 5).
3. d_90 ∝ task difficulty, weakly ∝ task data size (Sec. 4.3).
4. Generalisation bound that uses d_90 in place of parameter count (Thm. 1, Sec. 6).

## Methods we could borrow / discard

- **Borrow**: SAID (Structure-Aware Intrinsic Dimension) projection idea — could be inverted to define a "subspace covering" defense: if attacks span a manifold of d_90 dimensions, defender mass-trains against a basis of that manifold.
- **Borrow**: use d_90 as a *meta-quantity* to characterise harmful-vs-benign tasks. If harmful fine-tuning has higher intrinsic dim than benign, that asymmetry is exploitable.
- **Discard**: their projection is *random Fastfood*; for our defense we want *structured* (LoRA-shaped) projections.

## Open questions / disagreements

- They never measure d_90 for *safety-relevant* tasks (jailbreak fine-tuning). Open empirical question — could be a high-leverage measurement for our project.
- The "low intrinsic dim → defense is easy" intuition is backwards: low-dim means *less* information needed to attack, hence cheaper attack budgets. The defender's burden grows with the union over plausible task subspaces, not with d_90 of any one task.

## Citation

arXiv:2012.13255. Aghajanyan, Zettlemoyer, Gupta (2020/2021).
