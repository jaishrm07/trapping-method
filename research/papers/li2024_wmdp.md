# The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning

**Authors / venue / year**: Nathaniel Li et al. (CAIS / Hendrycks lab, 80+ authors) / ICML 2024 / 2024 (arXiv 2403.03218)
**Bib key**: li2024wmdp
**Read for thread(s)**: 04
**Read on**: 2026-05-02

## TL;DR (3 sentences)

WMDP is a 4,157-question multiple-choice benchmark over biosecurity, cybersecurity, and chemical hazardous knowledge — it serves as both a measurement tool for residual hazardous capability *and* the reference benchmark for unlearning-as-defense. The paper introduces RMU (Representation Misdirection for Unlearning), a representation-control unlearning method that reduces WMDP performance while preserving general capability. Important threat-model caveat: WMDP and RMU are evaluated against *passive* attackers (output-only access); the relearning-attack literature (TOFU, MUSE, Sheshadri-Casper "Distillation Robustifies Unlearning") shows RMU-style unlearning is largely *reversible* by a few hundred fine-tuning steps on a small forget-set sample.

## Why we read it

Thread 04 — WMDP/RMU is the unlearning-as-defense baseline and the natural complement to trap-induction (different mechanism, same goal of "the attacker shouldn't get the harmful capability back"). The relearning-attack literature is also where the LoRA-fine-tuning attack lives in the unlearning sub-community.

## Key claims (with location)

1. WMDP: 4,157 MCQ items in bio/cyber/chem hazardous knowledge (Sec. 2).
2. RMU: representation-control unlearning method, applied at chosen layers (Sec. 4).
3. RMU reduces WMDP score while preserving MMLU-class capability (Sec. 5).
4. **Implicit caveat**: the paper does not stress-test against an attacker that fine-tunes the unlearned model on hazardous data to recover the capability. This is the relearning attack, addressed in follow-up work.

## Methods we could borrow / discard

- **Borrow**: WMDP as a benchmark of last resort if we want to demonstrate trap-induction on a *capability-removal* threat model rather than an alignment-removal one. WMDP's MCQ format is convenient for comparison.
- **Borrow**: the relearning-attack methodology as our adversarial-finetuning attack budget for capability-removal experiments.
- **Discard**: RMU itself — it's a different mechanism (representation control) than trap-induction. Cite as a different threat-model variant.

## Open questions / disagreements

- The WMDP benchmark is the standard for hazardous capability; if our trap method extends to *both* alignment-erosion and capability-recovery threat models, we should report on WMDP-style targets. If only the former, scope clearly.
- "Distillation Robustifies Unlearning" (Sheshadri-Casper et al. 2025) extends RMU to be relearning-resistant via distillation. This is the closest "tamper-resistant unlearning" peer to our trap-method approach for capability removal.

## Citation

arXiv:2403.03218. ICML 2024. Li, Pan, Gopal, Yue, Berrios, Gatti, Li, Dombrowski, Goel, Phan, Mukobi, Helm-Burger, Lababidi, Justen, Liu, Chen, Barrass, Zhang, Zhu, Tamirisa, Bharathi, Khoja, Zhao, Herbert-Voss, Breuer, Marks, Patel, Zou, Mazeika, Wang, Oswal, Lin, Hunt, Tienken-Harder, Shih, Talley, Guan, Kaplan, Steneker, Campbell, Jokubaitis, Levinson, Wang, Qian, Karmakar, Basart, Fitz, Levine, Kumaraguru, Tupakula, Varadharajan, Wang, Shoshitaishvili, Ba, Esvelt, Wang, Hendrycks.
