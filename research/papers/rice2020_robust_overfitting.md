# Overfitting in Adversarially Robust Deep Learning

**Authors / venue / year**: Leslie Rice, Eric Wong, Zico Kolter / ICML 2020 / 2020
**Bib key**: rice2020robustoverfitting
**Read for thread(s)**: 02
**Read on**: 2026-05-02

## TL;DR (3 sentences)

Adversarial training (PGD-style) suffers severe robust overfitting: robust test accuracy peaks early then decays, while clean test accuracy keeps improving. Best practical fix is plain *early stopping* on a robust validation set; tricks like double-descent regularisation, smoothing, etc. don't beat it. The phenomenon is universal across CIFAR, SVHN, ImageNet.

## Why we read it

Thread 02 — our trap defender will likely have an analogous failure: trap-effectiveness on a held-out attack (different LoRA seed / rank / lr) peaks early in defender training and then decays as the defender over-fits to the specific inner adversary instances seen during training.

## Key claims (with location)

1. Robust test accuracy peaks within first ~50% of training and decays thereafter (Sec. 3, Fig. 1).
2. Early stopping on a held-out robust accuracy is the single best mitigation (Sec. 4).
3. Various heuristics (label smoothing, weight decay tuning, semi-supervised) help marginally; none beats early stopping (Sec. 5).

## Methods we could borrow / discard

- **Borrow**: build a held-out *attack distribution* (LoRA configs not seen during training: different rank, target layers, init scheme) and early-stop the defender when its trap-effectiveness on that held-out set peaks.
- **Borrow**: during defender training, log trap-effectiveness on multiple held-out attacks at every checkpoint. Pick the best.
- **Discard**: their other regularisation tricks — early stopping is the high-leverage move.

## Open questions / disagreements

- Robust overfitting is documented for k=7 PGD; unclear how it scales with our k=3-10 LoRA inner adversary on much larger models.

## Citation

arXiv:2002.11569. Rice, Wong, Kolter (2020).
