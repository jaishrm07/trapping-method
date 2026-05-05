# SOPHON: Non-Fine-Tunable Learning to Restrain Task Transferability for Pre-trained Models

**Authors / venue / year**: Jiangyi Deng, Shengyuan Pang, Yanjiao Chen, Liangming Pan, Yang Xiang, Yating Yang, Wenyuan Xu / IEEE S&P 2024 / 2024 (arXiv 2404.12699)
**Bib key**: deng2024sophon
**Read for thread(s)**: 04
**Read on**: 2026-05-02

## TL;DR (3 sentences)

SOPHON proposes "non-fine-tunable learning" for pre-trained vision/language models: protect the model so that fine-tuning to a *restricted* domain incurs cost equivalent to (or greater than) training from scratch, while preserving performance on the *original* domain. The mechanism is meta-learning-style, with explicit attempts to entrap the model in a "hard-to-escape local optimum" for the restricted task — this is the direct conceptual ancestor of Sarker et al.'s trapping framing. Evaluated across 7 restricted domains, 6 architectures, 3 fine-tuning methods, 5 optimizers — relatively broad robustness sweep, though still not LLM-scale.

## Why we read it

Thread 04 — SOPHON predates the LLM-side trapping work and explicitly uses "entrap" language. Important to clarify whether Sarker et al.'s contribution over SOPHON is conceptual (new geometry-aware trap formulation) vs. empirical (LLM/LoRA setting).

## Key claims (with location)

1. Two objectives: intactness on original domain + non-fine-tunability on restricted domains (§3).
2. Mechanism: meta-learning-style fine-tuning simulation/evaluation, optimizing for hard-to-escape local optima (§4).
3. Evaluated across 7 restricted domains, 6 architectures, 3 FT methods, 5 optimizers, varied lr/batch sizes (§5).
4. **Caveat**: pre-LLM-era; not evaluated on LLMs or on LoRA at LLM scale.

## Methods we could borrow / discard

- **Borrow**: SOPHON's "fine-tuning simulation/evaluation" as the predecessor template that both TAR (LLM) and Sarker trapping (LLM) refine. Cite as the methodological ancestor.
- **Borrow**: their robustness sweep across optimizers/lrs/methods is more comprehensive than most LLM-side defense papers. Adopt as a template for our evaluation.
- **Discard**: vision-domain experiments; the conceptual structure is what carries over.

## Open questions / disagreements

- Sarker's trapping paper should be explicit about how it extends/differs from SOPHON. The candidate differentiators: (a) geometric trap-minimum formulation rather than pure meta-learning, (b) LLM/LoRA setting, (c) explicit RFD-style metric. Verify with Sarker text.

## Citation

arXiv:2404.12699. IEEE S&P 2024. Deng, Pang, Chen, Pan, Xiang, Yang, Xu.
