# Thread 02 — Bilevel adversarial training stability

**Status:** open
**Type:** lit review
**Owner:** —

## Question

What is known about stabilizing bilevel adversarial training when the
inner adversary is a non-trivial fine-tuner (not a perturbation, not a
linear probe)? Specifically: what tricks let us avoid the "inner loop
overshoots → outer-loop gradient explodes → collapse" pattern we saw
in Stage 5 v2 attempts 1–5?

## Why this matters

Our concrete failure mode:
- LoRA inner loop with η=0.1 / k=3 / B=0 produces `softplus(ΔL_act) ∈
  [0, 360]` — wildly variable per step.
- This wide-distribution gradient destabilizes the defender.
- We patched it with output-clamp + λ_trap=0.3 + grad_clip=1, but those
  are band-aids. Real solution lives in the inner-loop dynamics.

## Reading list (priority-ordered)

1. **Finn, Abbeel, Levine 2017 — MAML.** The OG of bilevel meta-learning.
   Read for: how they handle inner-loop instability, second-order vs
   first-order tradeoffs.
2. **Nichol et al. 2018 — Reptile / FOMAML.** First-order approximations
   to MAML. Relevant: avoiding `create_graph=True` which we use.
3. **Rajeswaran et al. 2019 — iMAML.** Implicit-gradient bilevel via
   IFT — avoids unrolling, more memory-efficient, often more stable.
4. **Antoniou et al. 2018 — How to Train Your MAML.** Tricks for MAML
   stability: per-layer-per-step inner LR, multi-step loss, batch
   normalization weirdnesses. Directly applicable.
5. **Liu et al. 2019 — DARTS** and follow-ups. NAS as bilevel — many
   stability tricks, including weight sharing + warm restarts.
6. **Madry et al. 2018 — adversarial training (PGD).** Different bilevel
   (perturbation, not parameter), but the gradient masking / robust
   overfitting literature is relevant when our trap could induce
   gradient masking.
7. **Rice, Wong, Kolter 2020 — robust overfitting.** What happens when
   you over-train against a fixed adversary. Trap might suffer from a
   variant.

## Open hypotheses to test (post-read)

- Is our η=0.1 inner LR way too high relative to MAML standards (which
  typically use 1e-2 to 1e-3 for inner)?
- Should we use **first-order approximation** for the LoRA branch (skip
  `create_graph=True`)? Cheaper memory, possibly more stable, at cost of
  ignoring through-LoRA gradient flow to defender.
- Does **per-step inner LR scheduling** (Antoniou) help — start tiny,
  ramp up?

## Acceptance criterion

Thread closes when we have:
- A short list (≤5) of stability tricks from the literature with
  pointers to which paper / equation, **AND**
- An assessment of which is plug-in-able for our setting in <1 day of
  work.

## Field findings (Agent B, 2026-05-02)

- **Inner LR diagnosis: η=0.1 is ~100x larger than the literature norm.** Finn et al. 2017 (MAML, arXiv:1703.03400) use α ~ 1e-2; Lermen et al. 2024 (LoRA attack, arXiv:2310.20624) use AdamW lr=1e-4 with rank 16-64. Hayou et al. 2024 (arXiv:2406.08447) prove the maximum *stable* LR for Init[A] (B=0) LoRA is bounded above and depends on width — η=0.1 likely violates it. Quick win: drop η to ~1e-3 SGD or 1e-4 Adam and see if instability resolves before adopting any structural fix.
- **Drop `create_graph=True` first; iMAML if the FOMAML signal is too weak.** Nichol et al. 2018 (Reptile, arXiv:1803.02999) show FOMAML matches second-order MAML within ~1pp at much lower memory and *much* higher stability. TAR (Tamirisa et al. 2024, arXiv:2408.00761) use exactly FOMAML for their k=64 inner adversary — converging evidence that second-order is unnecessary for this class of defense. If FOMAML loses signal, fall back to iMAML (Rajeswaran et al. 2019, arXiv:1909.04630): add (λ/2)‖φ-θ‖² to inner objective, use CG-HVP for outer gradient. Inner-step memory becomes O(1).
- **MAML++ "Multi-Step Loss" is plug-and-play.** Antoniou et al. 2019 (arXiv:1810.09502, Sec. 4.2) compute outer loss at every inner step φ_1..φ_k with annealed weights and sum. This *evens out* the gradient contribution per inner step; the "step-k gradient is huge, step-1 is tiny" pattern that produced our [0, 360] softplus(ΔL_act) range disappears. Estimated <1 day to implement.
- **Per-layer per-step inner LR (LSLR) gives the system a learned schedule.** Antoniou Sec. 4.4: meta-train a tensor α[layer, inner_step] alongside θ. For our adversarial setting, the natural variant is to *sample* η from a distribution at each outer step rather than learn it (we don't want to optimise the attacker), but the "ramped schedule (small early, larger later)" finding is robust and worth hard-coding.
- **Track λ_max(∇²_θ L_outer) as a canary.** Zela et al. 2020 (RobustDARTS, arXiv:1909.09656) show in NAS that the dominant outer-Hessian eigenvalue rises monotonically before bilevel collapse. Estimate it cheaply via 5 power-iteration HVP steps every N outer steps. Spike → snapshot weights, drop outer LR, or early-stop. Generalises beyond NAS.
- **iMAML-style proximal regulariser (λ/2)‖LoRA-params‖² on the inner objective.** Independent of whether we use implicit gradients, this regulariser bounds the inner Hessian spectrum and acts like weight decay on the adversary. Same trick that RobustDARTS calls "stronger inner regularisation" (Zela §4.2). Likely the single highest-leverage *single-line* change.
- **Madry-style "no graph through the inner" is also a viable architecture.** Madry et al. 2018 (arXiv:1706.06083) bilevel adversarial training uses k=7 PGD inner steps but never differentiates through them — the inner just produces a perturbed example used as ordinary outer training data. Translating to our setting: run k LoRA steps, take the post-attack model, compute outer trap loss on it as if it were a normal forward pass. No second-order, no exploding meta-gradient.
- **Robust overfitting will bite. Plan early stopping now.** Rice, Wong, Kolter 2020 (arXiv:2002.11569) show that in PGD-adversarial training, robust test accuracy peaks early and decays. Our analogue: held-out-attack trap effectiveness will peak then decay. Build a held-out attack distribution (different LoRA rank/lr/init/target-layer) *now* and early-stop on it.
- **LoRA-specific stabiliser: scale the inner step by α/√r (rsLoRA), not α/r.** rsLoRA (Kalajdzievski 2023) and Hayou 2024 (arXiv:2406.08447) show the standard α/r scaling under-trains B at high rank; α/√r maintains a constant feature-update magnitude across rank. For our defender training, sweep rank ∈ {8,16,32,64} with rsLoRA scaling so the attacker's effective step size is rank-independent — eliminates one nuisance variable.
- **Gradient masking sanity-check: the trap *will* tempt us into it.** Athalye, Carlini, Wagner 2018 (arXiv:1802.00420) catalogue exploding/vanishing gradients in iterative defenses. Our defender, trained against k=3-10 inner steps, may simply learn to push the attacker into a region of bad LoRA gradient flow that disappears at k=100. Diagnostics: (a) test-time attack with k=10*train_k and unbounded LR-grid; (b) random LoRA inits over 50 seeds; (c) attack with rsLoRA scaling and Adam (not just SGD). If any breaks the defense, we have gradient masking, not robustness.
