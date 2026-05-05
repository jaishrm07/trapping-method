# Thread 04 — HFT defense landscape (post-2024)

**Status:** open
**Type:** lit review
**Owner:** —

## Question

Who else is working on defending open-weight models against harmful
fine-tuning, and what's their state of the art? Specifically: has
anyone tried to defend *LoRA* fine-tuning attacks (vs prompt-injection
or activation-steering)?

## Why this matters

We may be reinventing wheels. Conversely, we may be one of the few
groups taking the *parameter-defense* angle (vs guardrails / RLHF
hardening / mechanistic-interpretability circuit breaking). Need to
know the field map before pitching this direction.

## Known work to start from

- **Zheng et al. ICML 2025** — CN-immunization. Reproduced (Stage 4.5).
  No LoRA defense.
- **Sarker et al. NeurIPS 2025 (Lock-LLM Workshop)** — trapping. We
  reproduced. Linear-probing only.
- **Lermen, Rogers-Smith, Ladish 2024** — *LoRA Fine-tuning Efficiently
  Undoes Safety Training in Llama 2-Chat 70B*. The attack we're trying
  to defend against. Read for: attack budget, layer choice, rank
  effects.
- **Yang et al. 2023** — *Shadow Alignment* (similar attack with full
  FT). Setting context.
- **Zou et al. 2024 — circuit breakers**. Defends at the
  representation level. Different angle. Compare assumptions.
- **HFT survey** (early 2026, downloaded in `papers/`). Look for
  defense taxonomy section + open problems.

## Reading list (priority-ordered, 2024+)

1. The HFT survey we downloaded — defense taxonomy, open problems list.
2. Tamper-Resistant Safeguards papers (Zou and follow-ups).
3. Anything by **Andy Zou, Mantas Mazeika, Dan Hendrycks** post-2024.
4. **NeurIPS 2025 Lock-LLM workshop** — full proceedings. Other papers
   in the same workshop as Sarker et al. likely overlap heavily.
5. **Aghajanyan-track** intrinsic-dimensionality work — informs how
   small a LoRA adversary really is, and whether we can use that.
6. **MUSE / Tofu / WMDP unlearning benchmarks** — different problem
   (unlearn a fact) but methodology overlaps.

## Open hypotheses to test (post-read)

- Does the field already converge on "you can't defend LoRA fine-tuning,
  use guardrails instead"? If so, what's our contribution argument?
- Is there work treating fine-tuning attacks as a *distribution* over
  attack methods (DRO connection to thread 03)?
- What's the state of *evaluation*? Are RFD-on-Cars and similar the
  community's standard, or has something better emerged?

## Acceptance criterion

Thread closes when we have:
- A 1-page field map: who's doing what, what's defended, what isn't,
  **AND**
- A clear positioning statement for our line of work that's consistent
  with the field.

## LoRA-specific findings (Agent B, 2026-05-02)

- **Canonical attack budget (Lermen, Rogers-Smith, Ladish 2024, arXiv:2310.20624).** Rank 16-64, AdamW lr=1e-4, ~100 steps, target attention q_proj/v_proj (or all linears), <$200 single-GPU on Llama-2-Chat 70B. Refusal rate drops from ~100% to <1% on AdvBench, MMLU drops <1pp. This is the threat model — our defense must hold over rank ∈ {1,...,64}, lr ∈ {1e-5,...,1e-3}, AdamW vs SGD, attention-only vs all-linear.
- **LoRA parameterisation (Hu et al. 2021, arXiv:2106.09685).** h = Wx + (α/r)BAx. Default init: A ~ Kaiming-uniform, **B = 0** so ΔW=0 at step 0. α typically set to first r tried and not tuned. This is what Lermen attacks and what we reproduce as our adversary. Note: "init B=0" creates an asymmetric loss landscape — at t=0 ∇_A L = 0, only ∇_B L is nonzero. Hayou et al. 2024 (arXiv:2406.08447) prove this gives Init[A] a *smaller* maximum stable LR than the alternative Init[B] (A=0, B random). Implication: the attacker's stable-LR ceiling depends on init choice; defender must cover both.
- **Intrinsic dimensionality framing (Aghajanyan, Zettlemoyer, Gupta 2020, arXiv:2012.13255, ACL 2021).** Fine-tuning RoBERTa on MRPC has d_90 ≈ 200; GPT-3-class models have d_90 in the few-thousands. Larger pretrained models have *lower* intrinsic dim. For us this cuts both ways: (i) attacks live in a low-d subspace per task, so the defender's burden per attack is bounded; (ii) but the *union over plausible harmful tasks* may have much higher intrinsic dim. Open empirical question (we should measure): what is d_90 of jailbreak fine-tuning specifically?
- **PEFT methods are a structured family, not a discrete set (He et al. 2022, arXiv:2110.04366, ICLR 2022).** Adapters, prefix-tuning, LoRA, BitFit, IA³ all fit a single template Δh = α·W·f(x), differing only in insertion location and parameterisation form. Implication for defense: a LoRA-only defense is a special case; a defender targeting the full PEFT family is the principled threat model. We have not seen prior work formalise the *defender-side* union — potential contribution angle.
- **TAR (Tamirisa et al. 2024, arXiv:2408.00761, ICLR 2025) is the most directly comparable defense.** Bilevel adversarial training where inner = LoRA fine-tune (k=64), outer = maximise post-fine-tune harmful loss. Uses FOMAML (no second-order through the inner). Reports tamper-resistance over hundreds of fine-tune steps. Critique: Qi et al. 2024 ("On Evaluating the Durability of Safeguards", arXiv:2412.07097) showed several TAR-style defenses break under stronger / longer attacks — i.e. early gradient-masking signature. Our positioning: "trap-based parameter defense" is a different mechanism than TAR's outer-loss formulation, so even if TAR is brittle, the trap idea is independent.
- **Pruning + low-rank can break safety alignment without LoRA (Wei et al. 2024, "Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications").** Adds an attack mode where pruning ~3% of safety-critical weights or applying a low-rank perturbation outside the LoRA optimisation regime breaks alignment. Defense implication: any defense that protects only against gradient-based fine-tuning leaves a side door open. Our trap should be evaluated against weight pruning and untrained low-rank perturbations as well.
- **Init asymmetry has a defender-side implication.** Hayou 2024 shows Init[A] (B=0) is conservative; Init[B] (A=0) tolerates higher LR and learns faster. If our attacker is using Init[A] and we trap that regime, an attacker using Init[B] (or QLoRA's tied init, or LoRA+ with η_B/η_A ratio) may bypass us cleanly. Defender training should sample over init schemes.
- **Rank-stabilised scaling (rsLoRA, Kalajdzievski 2023).** Scales adapter as α/√r instead of α/r. Makes higher-rank attacks more effective without re-tuning lr — a *stronger* threat model. Defender needs to be robust to this.
- **The community has *not* converged on "you can't defend LoRA, use guardrails."** Active counter-evidence: TAR (ICLR 2025), Repnoise, "Deep Ignorance" (arXiv:2508.06601, filtering-pretraining-data approach), and our own trapping line. The skepticism (Qi et al. 2412.07097) targets durability under stronger attacks, not the existence of the research direction. Our positioning: tractable, testable parameter-level defense as a sibling to TAR with a different mechanism.
- **Standard evaluation suite is converging.** AdvBench refusal rate (Lermen) + MMLU retention (capability) + multiple LoRA configs swept (rank/lr/target). RFD-on-Cars (used in trapping work) is a different benchmark for ResNet linear-probing setting; LLM evaluations are essentially Lermen-style. Need both for our project.

## Field map (Agent A, 2026-05-02)

Skeptical synthesis of the post-2024 HFT-defense landscape, distinguishing
*claimed* defenses from *demonstrated*-against-LoRA defenses. Complements
Agent B's LoRA-attack-side findings above. Per-paper notes are in
`research/papers/`.

### (a) Who's working on this and where they publish

Five clusters dominate the parameter-modification threat model:

1. **Hendrycks / Mazeika / Zou (CAIS, Gray Swan, UIUC, Berkeley).** Tamper-
   resistance lineage for open-weight LLMs. Outputs: TAR (Tamirisa et al.,
   ICLR 2025; arXiv 2408.00761), Circuit Breakers (Zou et al., NeurIPS
   2024; arXiv 2406.04313 — but this is *representation-level*, not
   parameter-mod), WMDP (Li et al., ICML 2024; arXiv 2403.03218).
   Publication venues: ICLR / NeurIPS / ICML main tracks.

2. **Huang / git-disl (Georgia Tech, Liu group).** Alignment-stage
   perturbation defenses. Vaccine (NeurIPS 2024; 2402.01109), Booster
   (ICLR 2025 Oral; 2409.01586), the HFT survey (arXiv 2409.18169, v6
   April 2026), and Yi (co-author) on CTRAP (arXiv 2505.16559). Centroid
   of the field by publication volume.

3. **Sajjad / Rosati (Dalhousie / U Waterloo).** RepNoise (NeurIPS 2024;
   2405.14577). Lock-LLM 2025 entry on undistillable models with teacher
   scrambling.

4. **Zheng / Yeh (Purdue, Google).** Immunization through curvature.
   IMMA (ECCV 2024; 2311.18815) for diffusion. Condition-number
   immunization for LLMs (ICML 2025 Oral; 2505.23760).

5. **Thomas group (Virginia Tech).** Trap-induction (Sarker et al.,
   NeurIPS 2025 Lock-LLM Workshop; OpenReview gfAn827WAW). One workshop
   paper, conceptually adjacent to CTRAP and SOPHON.

**Adjacent clusters worth tracking:**
- AISI / Apollo / EleutherAI (Casper, Gleave, Cundy, Biderman): Deep
  Ignorance (2508.06601). Evaluation-axis: "Safety Gap Toolkit" at
  Lock-LLM 2025.
- Unlearning-as-defense (Turner / MATS, CAIS): UNDO (2506.06278); RMU
  (WMDP).
- Pre-LLM ancestors: Henderson MLAC (AIES 2023; 2211.14946); SOPHON
  (Deng et al., IEEE S&P 2024; 2404.12699).
- PEFT safety-hygiene (not tamper resistance): SaLoRA (2501.01765);
  Safe LoRA (2405.16833).

**Lock-LLM 2025 reality check.** Of ~23 confirmed accepted poster papers,
**only Sarker et al. is on the Un-Finetunable / parameter-defense lane**.
The other accepted papers cluster on fingerprinting (OML, Nasery), prompt
injection, multimodal red-teaming, agent security, undistillable models
(Rosati's Teacher Scrambling), and quantization-adversarial reparams.
Sarker is essentially uncontested in the workshop's parameter-defense
slot, which is favorable positioning but means the workshop alone is
not the field map.

### (b) Threat models — defended vs. claimed

| Threat model | Attacker capability | Canonical defenses |
|---|---|---|
| Inference-time / prompt | Adversarial prompts, multi-turn jailbreak (Crescendo), image hijack | Circuit Breakers (Zou 2024). **Not** parameter-mod. |
| Activation steering | Add steering vectors at inference | Adjacent literature; not the trap target. |
| **Parameter modification (HFT)** — our target | Download weights, fine-tune on harmful data | TAR, RepNoise, Vaccine/Booster, CTRAP, Sarker trapping, Zheng CN-immunization, Deep Ignorance (pre-train), UNDO (unlearn-then-distill) |

Within the HFT bucket, the literature collapses operator class
(full-FT vs LoRA vs prompt-tuning) into pipeline stage (alignment / FT /
post-FT). The Huang survey explicitly does this. The collapse hides the
LoRA gap because most "alignment-stage" defenses are headlined under
full-FT.

### (c) What's been *demonstrated* (vs claimed) for LoRA defense

**Demonstrated against LoRA in primary results:**
- **IMMA (Zheng & Yeh, ECCV 2024).** Diffusion T2I, evaluated against
  LoRA + Textual Inversion + DreamBooth. Cleanest operator-aware
  defense in the literature. Vision-only — does not yet transfer to LLMs.
- **Booster (Huang et al., ICLR 2025 Oral).** Headline numbers under
  LoRA (rank 32, α=4). Sweep over rank / lr / optimizer is narrow; not
  stress-tested against Lermen-style adversarial LoRA.
- **SaLoRA / Safe LoRA.** Demonstrated under LoRA, but threat model is
  *benign user fine-tuning*, not adversarial HFT. Misleading to count
  as tamper-resistance.

**Claimed / partial — known LoRA failures:**
- **TAR (Tamirisa et al., ICLR 2025).** "28 red-team adversaries" but
  only 2 use PEFT. Huang survey + 2025 MIT thesis (Zhang) report
  TAR's safeguard "largely breaks" under LoRA. Headline step counts
  are full-FT.
- **RepNoise (Rosati et al., NeurIPS 2024).** No clean LoRA-rank
  ablation in primary results; Huang survey notes hyperparameter
  sensitivity. Qi et al. 2024 (2412.07097) shows TAR-class defenses
  break under stronger/longer attacks — likely applies here.
- **Sarker trapping (NeurIPS 2025 Lock-LLM Workshop).** Reproduced:
  linear-probing RFD ≈ 47, **LoRA RFD ≈ 1.16**. LoRA effectively
  undefended — this is the gap.
- **Zheng CN-immunization (ICML 2025 Oral).** Local curvature only;
  no LoRA-specific operator conditioning. Multi-step LoRA trajectory
  unbounded.
- **CTRAP (Yi et al., 2505.16559).** No published LoRA-specific
  ablation. Mechanism (collapse trigger from gradient distinguishability)
  may not transfer across LoRA ranks.

**Pre-training-stage:**
- **Deep Ignorance (O'Brien/Casper et al., 2508.06601).** 10K-step /
  300M-token resistance budget on a 6.9B model. LoRA breakdown not
  foregrounded in abstract — needs full-paper read. Pre-training
  filtering is more durable conceptually but bypassable via in-context
  retrieval (authors acknowledge).

**Unlearning-as-defense:**
- **WMDP/RMU (Li et al., ICML 2024).** Vulnerable to relearning
  attacks, including LoRA-based. Sheshadri/Casper note benchmarks
  measure passive-attacker robustness only.
- **UNDO (Lee et al., 2506.06278).** Distill-after-unlearn matches
  retrain-from-scratch on WMDP. LoRA-specific evaluation against the
  distilled model is not foregrounded.

**Adjacent attack channels.** Wei et al. 2024 ("Assessing the Brittleness
of Safety Alignment via Pruning and Low-Rank Modifications") shows
pruning ~3% of safety-critical weights or applying *untrained* low-rank
perturbations breaks alignment without any fine-tuning. Defenses that
target only gradient-based attacks are blind to this side door.

**Net assessment.** As of mid-2026, **no LLM defense in the open
literature is robustly demonstrated against the full Lermen 2024
attack budget under LoRA across rank/lr/optimizer sweeps**. Booster
claims the closest but with narrow grids. The field's "best LoRA
defense" is currently aspirational rather than established.

### (d) Open problems / gaps the trap-method angle could land

The Huang survey §5 flags hyperparameter-robustness, cross-scenario
generalization, and mechanistic interpretability as the three open
problems. Cross-referencing with the LoRA gap and the operator-class
axis, the concrete openings are:

1. **Operator-conditioned defense for LLMs.** No published LLM defense
   targets a specific adaptation operator; IMMA does this for diffusion.
   Trap-induction conditioned on the LoRA-reachable subspace is the
   strongest unfilled cell.

2. **Trajectory-level guarantees under PEFT.** Zheng CN gives local
   bounds, Sarker trapping makes basin-level claims, but neither
   bounds the LoRA-class attacker's *trajectory*. A bilevel /
   meta-learning formulation taking LoRA hyperparameters as inner-loop
   parameters is missing.

3. **DRO over the attacker distribution.** No paper formalizes the
   defender's problem as min-max over a distribution of LoRA
   adversaries (rank ∈ {8,16,32,64}, lr ∈ [1e-5,1e-3], steps ∈
   [50,1000], init ∈ {Init[A], Init[B], rsLoRA}). Connects to thread 03.
   Wei-style untrained-low-rank attacks should also be in the support.

4. **Per-operator tamper-resistance budgets.** Field standard is
   single-headline numbers (TAR's "thousands of steps", Deep Ignorance's
   "10K steps"). No paper reports a 2D table (operator × budget).
   Adopting this as a reporting standard is itself a contribution.

5. **Mechanistic account of why defenses fail under LoRA.** Why does
   trapping linear-probing succeed (RFD 47) but LoRA fail (RFD 1.16)?
   Thread 01 needs to answer this; no published paper attempts it
   because the LoRA-specific evaluation is rarely done.

6. **Composition with unlearning / distillation.** UNDO shows
   distillation amplifies unlearning robustness. Whether trap geometry
   survives distillation, or whether distillation can be used to
   *amplify* trap geometry, is open.

7. **PEFT-family-level defense.** PEFT methods (LoRA, prefix-tuning,
   adapters, IA³) share a template (He et al. 2022). A defense
   formalised at the family level rather than operator-by-operator
   has no published instance.

### (e) Where our line of work sits relative to the field

We are in the **post-training, alignment-stage, geometric-trap**
sub-cluster, alongside Sarker's trapping (which we extend) and Zheng's
condition-number immunization (which we also extend). Distinct from:

- **TAR** (Hendrycks lineage) — same threat model, different mechanism.
  Trap commits to a basin geometrically; TAR simulates attacker
  trajectories via meta-learning. Even if Qi et al. 2412.07097's
  durability critique applies to TAR-style defenses, trap-induction is
  an independent mechanism.
- **CTRAP** (Yi et al.) — same word "trap", but mechanism is *collapse
  on attack-gradient detection*, not basin geometry. Need to contrast
  explicitly, not just with TAR/Zheng.
- **Booster / Vaccine / RepNoise** — alignment-stage, but representation
  /perturbation-based, not landscape-based.
- **Deep Ignorance** — orthogonal pre-training stage. Defense-in-depth
  partner, not competitor.
- **IMMA** — closest *conceptual* peer (operator-aware defense), but
  in diffusion. Transferring IMMA's operator-conditioned framing to
  LLM LoRA is the unfilled cell that defines our paper.

**Positioning statement (draft).** *We extend Sarker et al.'s
trap-induction objective and Zheng et al.'s condition-number
regularization to bound LoRA fine-tuning adversaries specifically. We
treat the problem as operator-conditioned defense over a distribution
of LoRA configurations (rank, lr, steps, init), and we report defense
efficacy per-operator rather than as a single tamper-resistance number.
Our angle on the LoRA gap is methodologically novel: no LLM-side
parameter-mod defense currently demonstrates LoRA resistance under a
non-trivial budget × hyperparameter sweep, and IMMA shows the
operator-aware approach works in vision but has not been transported
to LLMs.*

**Key risks (null hypotheses to test).**
- *Capacity tradeoff null.* If LoRA-resistance requires shrinking the
  LoRA-reachable subspace, benign-LoRA-FT utility collapses. The
  contribution then shrinks to "you should report per-operator RFD" —
  an evaluation contribution, not a defense one.
- *Operator-extrapolation null.* A defense conditioned on rank=r₀ may
  fail at rank r₁ ≠ r₀. The DRO operator-mixture in thread 03 is the
  hedge; we should explicitly target this in evaluation.
- *Side-door null.* Wei-style untrained low-rank perturbations and
  pruning attacks are not gradient-based; the trap may not hold against
  them. We should verify.

## PEFT-family operator-set test results (2026-05-03)

Per-thread Q5 (PEFT-family-as-defender-target) was tested as Stage 5 v5b:
- `trap_operators: [linear_probe, lora_r4, lora_r8, lora_r16, lora_r32]`
- Uniform sampling, otherwise identical to v4a (FOMAML, no predictor).

### Outcome: NaN cascade at step 1051 / 2500

```
step 1050: ill=0.0571 primary=1.484 trap=3.5817 well=0.1307
step 1051: ill=nan    primary=nan   trap=nan    well=nan
```

Same failure signature as the original v4 NaN attempt on Falcon. Killed
manually at step 1051. Not re-runnable without further intervention.

### Provisional diagnosis

LoRA-r32's per-step gradient magnitude (4× LoRA-r8's) appears to
exceed what `grad_clip=1.0` + the trap output clamp at 10 can absorb.
Per-step trap=3.58 was within the unclamped regime — the clamp didn't
bite. `grad_clip` rescales globally, preserving direction; if rank-32's
contribution dominates a single step, it skews the optimizer step.
`‖B@A‖_F` for randomly-initialized LoRA-r32 is √4× larger than r8 even
with the standard `α/r` scaling.

### What it would take to make Fork B work

| Fix | Cost |
|---|---|
| **Per-rank gradient clipping** in `trap_loss_multiop` | medium refactor |
| **rsLoRA-style `α/√r` scaling** to equalize per-step magnitude across ranks | low |
| **Drop LoRA-r32; test `[LP, r4, r8, r16]` first** | trivial |

Q5 (PEFT-family unification as defender's joint target) remains
unanswered empirically. The *field gap* (no one has formalized the PEFT
family as a joint defender target) is still open as a contribution
angle, but our naive uniform-sampling expansion is not the implementation
that closes it.

## Three null hypotheses — current state (2026-05-03)

Agent A's three null tests for any positive LoRA defense claim:

| Null hypothesis | Tested? | Status |
|---|---|---|
| **Capacity tradeoff** — bounding LoRA-reachable subspace destroys benign LoRA-FT utility | NO | We haven't run benign-task LoRA fine-tune on any immunized backbone |
| **Operator-extrapolation** — defense at rank=r₀ fails at r₁≠r₀ | partial | v4a probes against r8/r32 show the defense doesn't generalize (LoRA-r32 RFD = −1.18 in v4a, i.e. immunized backbone is *easier* for r32 than baseline) |
| **Side-door** — pruning + untrained low-rank perturbations bypass any gradient-based defense | NO | Need Wei et al. 2024-style pruning attack |

The capacity-tradeoff null is the most important next test even if we
abandon further trap-formulation work. Difference between "we defend
LoRA fine-tuning" and "we destroy LoRA fine-tuning ability generally."

