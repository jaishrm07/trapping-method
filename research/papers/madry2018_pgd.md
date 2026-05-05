# Towards Deep Learning Models Resistant to Adversarial Attacks (PGD)

**Authors / venue / year**: Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu / ICLR 2018 / 2017-2018
**Bib key**: madry2018pgd
**Read for thread(s)**: 02
**Read on**: 2026-05-02

## TL;DR (3 sentences)

Frames adversarial robustness as a min-max saddle-point min_θ E max_‖δ‖≤ε L(x+δ; θ) and shows PGD is a strong "universal" first-order inner solver. The defender training is bilevel — solve inner with PGD for k steps (not differentiated through; just used as data augmentation), then take an outer step on the perturbed example. Notes empirically that more inner steps = more robust but with diminishing returns.

## Why we read it

Thread 02 — the canonical bilevel adversarial training. *Crucially, Madry-style training does NOT differentiate through the inner loop* — the inner produces a worst-case input which is then used as ordinary training data. This is a key contrast with our setup, which does use create_graph=True.

## Key claims (with location)

1. Saddle-point formulation (Sec. 2, Eq. 2.1).
2. PGD is approximately the strongest first-order adversary — multiple random restarts converge to similar loss values (Sec. 4).
3. Larger model capacity needed for adversarial training to converge well (Sec. 5.2).
4. Training uses k=7 inner steps for CIFAR-10; *no second-order through the inner*.

## Methods we could borrow / discard

- **Borrow critically**: ask whether we *need* `create_graph=True`. If our outer signal can be defined as "post-attack loss", we can drop the second-order graph entirely (Madry-style) and gain stability for free, at the cost of some specificity in the meta-gradient.
- **Borrow**: many random restarts of the inner adversary at evaluation; helps detect gradient masking.

## Open questions / disagreements

- Madry's inner adversary is a perturbation in *input* space (small, bounded). Our inner adversary is a *parameter* update (LoRA), unbounded except by step count × LR. The optimum-of-the-inner is much less well-defined; we may need iMAML-style proximal regularisation to make Madry-style "no-graph" training well-posed.

## Citation

arXiv:1706.06083. Madry et al. (2018).
