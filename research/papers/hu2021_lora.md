# LoRA: Low-Rank Adaptation of Large Language Models

**Authors / venue / year**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen / ICLR 2022 (arXiv 2021) / 2021
**Bib key**: hu2021lora
**Read for thread(s)**: 01, 02, 04
**Read on**: 2026-05-02

## TL;DR (3 sentences)

LoRA freezes pre-trained W and learns a low-rank update ΔW = BA where A ∈ ℝ^{r×k}, B ∈ ℝ^{d×r}, r ≪ min(d,k). Standard init: A is Kaiming-uniform (gain=√5), B is *zeros* — guarantees ΔW = 0 at step 0 so fine-tuning starts from the pre-trained model exactly. The forward pass uses W + (α/r) BA where α is a fixed scalar (default α = r so the multiplier is 1; rsLoRA later proposes α/√r).

## Why we read it

Thread 01 / 02 — we need the canonical LoRA spec to (a) match the attacker's parameterisation exactly and (b) understand why B=0 init creates the gradient pathology we see in Stage 5. Specifically, ∂L/∂A at step 0 is zero (because B=0 multiplies it), so ALL early signal flows through ∂L/∂B, which is amplified by A's Kaiming-scale norms — this is exactly the asymmetric gradient explosion we observe.

## Key claims (with location)

1. Parameterisation: h = Wx + (α/r) BAx with A∼𝒩(0, σ²), B = 0 (Sec. 4.1, Eq. 3).
2. Default targets q_proj and v_proj only — sufficient on GPT-3 (Sec. 7.1).
3. r ∈ {1,2,4,8} typically; performance saturates fast (Sec. 7.2 Tab. 6).
4. α/r scaling — α treated as another lr; "we set α to the first r we try and do not tune it" (Sec. 4.1).
5. ΔW has higher "amplification factor" of task-specific directions in W than W's own top singular subspace (Sec. 7.3).

## Methods we could borrow / discard

- **Borrow**: exact init recipe for our adversary so our threat model matches what attackers actually run.
- **Diagnose**: B=0 init *creates* an asymmetric loss landscape — at t=0, ∇_A L = 0 and ∇_B L = (α/r) (∂L/∂h) x A^⊤. This couples B-gradient magnitude to ‖A‖ ~ √(2/k), which for k=4096 is small but not tiny. With η=0.1 (our setting) and many inner steps with `create_graph=True`, this asymmetry can compound through second-order paths.

## Open questions / disagreements

- What if we use the *symmetric* "init A=0, B=Kaiming" variant (Hayou 2024 — see paper note hayou2024_lora_init)? It changes which matrix carries the early-step signal and may stabilise unrolled adversary gradients.
- What's the smallest r at which the attack still works? (Lermen uses r=16-64; Hu shows r=1 sometimes suffices on GLUE — our defense range should cover r ≥ 1.)

## Citation

arXiv:2106.09685. Hu et al. (2021/2022).
