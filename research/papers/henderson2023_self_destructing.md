# Self-Destructing Models: Increasing the Costs of Harmful Dual Uses of Foundation Models (MLAC)

**Authors / venue / year**: Peter Henderson, Eric Mitchell, Christopher Manning, Dan Jurafsky, Chelsea Finn / AIES 2023 / 2023 (arXiv 2211.14946)
**Bib key**: henderson2023selfdestruct
**Read for thread(s)**: 04
**Read on**: 2026-05-02

## TL;DR (3 sentences)

The methodological ancestor of essentially all post-training tamper-resistance defenses (TAR, RepNoise, Sarker trapping, CTRAP). Introduces MLAC (Meta-Learned Adversarial Censoring) — a meta-training procedure that searches for parameter states that are easy to fine-tune for the desired task but represent low-utility local optima for the harmful task, demonstrated at small scale on a BERT-style model preventing repurposing for gender identification while preserving profession classification. The conceptual core — "find a parameter state where the harmful task is hard to fine-tune toward" — is the seed of the trap-induction framing.

## Why we read it

Thread 04 — establishes the lineage: Henderson 2023 (MLAC) → Tamirisa 2025 (TAR, refined MLAC for LLMs) → Sarker 2025 (geometric-trap variant of MLAC for LLMs). Citing this paper correctly is necessary to position trap-induction as a refinement, not reinvention.

## Key claims (with location)

1. Conceptual goal: parametric states amenable to fine-tuning for benign tasks but local optima for harmful tasks (§3).
2. Method: meta-learning-style training over benign-FT and harmful-FT trajectories (§4).
3. Empirical: small-scale BERT result, gender-identification task vs. profession classification (§5).
4. Acknowledges this is conceptual proof-of-concept; LLM scaling is left open.

## Methods we could borrow / discard

- **Borrow**: MLAC's framing of the parameter state as having dual properties (easy-to-FT for benign, hard-to-FT for harmful) is the cleanest articulation of what a trap-method paper is delivering. Cite directly.
- **Borrow**: the small-scale BERT validation pattern — useful to first prove the trap-induction extension at small scale before scaling to LLMs.
- **Discard**: the BERT setting is dated; method conceptually carries forward but specifics don't.

## Open questions / disagreements

- MLAC predates LoRA. The crucial question — how does this defense interact with LoRA-rank attacker — is not addressed because the technique didn't exist. Our paper inherits this question.

## Citation

arXiv:2211.14946. AIES 2023. Henderson, Mitchell, Manning, Jurafsky, Finn.
