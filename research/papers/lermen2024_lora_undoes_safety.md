# LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B

**Authors / venue / year**: Simon Lermen, Charlie Rogers-Smith, Jeffrey Ladish / arXiv preprint (revised May 2024) / 2023-2024
**Bib key**: lermen2024lora
**Read for thread(s)**: 04, 01
**Read on**: 2026-05-02

## TL;DR (3 sentences)

The paper shows that QLoRA with rank ~16-64 and a budget under $200 on a single GPU is enough to drive Llama 2-Chat 7B/13B/70B and Mixtral-Instruct refusal rates on AdvBench from ~100% to <1% while preserving general capability (MMLU). The attack is cheap precisely because it is parameter-efficient: only a tiny low-rank perturbation to a handful of attention projections is required to "unlock" already-latent harmful behaviour. The takeaway: open-weight safety training is not a durable barrier under LoRA-class adversaries.

## Why we read it

Thread 04 — this is the canonical attack we are defending against. Thread 01 — the attack budget defines the threat model the trap must dominate.

## Key claims (with location)

1. LoRA rank 16 (later 64), AdamW, lr=1e-4, ~100 steps suffices on 70B with QLoRA 4-bit (Sec. 3, Tab. 1).
2. Attack adapts only attention `q_proj`/`v_proj` (later all linear layers); MLP-only attacks also work but slightly less efficient (Sec. 3.2).
3. Refusal rate drops from ~100% to <1% on AdvBench while MMLU drops <1pp (Tab. 2-3).
4. Total compute cost <$200 on 70B with one A100 / consumer GPU (Sec. 1).

## Methods we could borrow / discard

- **Borrow**: their hyper-parameter grid as the default attack we red-team against. Use rank ∈ {8,16,32,64}, lr ∈ {1e-5..1e-3}, target attention projections.
- **Discard**: nothing — this is the attack, not a defense.

## Open questions / disagreements

- They use lr=1e-4 (Adam). Our defender training uses η=0.1 SGD on the LoRA branch, which is ~3 orders of magnitude larger relative to typical LoRA practice. This may explain Stage-5 instability (cf. thread 02).
- Does the attack budget scale with defense strength? We need to characterize how many *more* steps the attack needs vs. base after the trap is installed.

## Citation

arXiv:2310.20624. Lermen, Rogers-Smith, Ladish (2024).
