# Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs

**Authors / venue / year**: Kyle O'Brien, Stephen Casper, Quentin Anthony, Tomek Korbak, Robert Kirk, Xander Davies, Ishan Mishra, Geoffrey Irving, Yarin Gal, Stella Biderman / arXiv 2508.06601 (August 2025) / Oxford / EleutherAI / UK AISI
**Bib key**: obrien2025deepignorance
**Read for thread(s)**: 04
**Read on**: 2026-05-02

## TL;DR (3 sentences)

Deep Ignorance argues that the most durable form of tamper-resistance is *pre-training-stage* data filtering — never let the model learn the dangerous knowledge in the first place — and demonstrates resistance to **10,000 adversarial fine-tuning steps** and 300M tokens of biothreat-related text on a 6.9B model, claiming "an order of magnitude" better tamper-resistance than post-training defenses. Critical caveat the authors raise: filtered models can still leverage the knowledge if it is provided in-context (RAG / search-augmented), so the defense is *not* sufficient on its own. This shifts the field's frame: instead of "make the post-training safeguard tamper-resistant", "make the model genuinely ignorant of the unsafe content."

## Why we read it

Thread 04 — Deep Ignorance is the new tamper-resistance baseline-to-beat as of late 2025 and reframes what counts as a defense. If the field is moving toward pre-training filtering, post-training defenses (TAR, trap, condition-number) are pushed into a "defense in depth" role rather than a primary defense.

## Key claims (with location)

1. Pre-training data filtering, with a multi-stage pipeline targeting biothreat proxy knowledge (Sec. 3).
2. Tamper-resistance budget: **10,000 fine-tuning steps, 300M tokens of adversarial data** on a 6.9B model (Sec. 5). This is significantly more than post-training defenses claim.
3. State-of-the-art vs. unspecified post-training baselines (Sec. 5).
4. **Limitation (authors')**: "filtered models lack internalized dangerous knowledge, [but] can still leverage such information when it is provided in context (e.g., via search tool augmentation), demonstrating a need for a defense-in-depth approach."

## Methods we could borrow / discard

- **Borrow**: the framing that *tamper-resistance is an empirical resistance budget* (steps × tokens × adversaries). Adopt this as the primary metric for our paper, alongside RFD.
- **Borrow**: the defense-in-depth framing — argue trap-induction is a complementary post-training layer to data filtering, not a competitor.
- **Discard**: pre-training filtering is out of scope for our line of work (we work on already-pre-trained models).

## Open questions / disagreements

- Is Deep Ignorance evaluated under LoRA? The 10K-step number appears to be full-FT or unspecified; the LoRA-attack channel may behave differently because LoRA cannot change the parts of the model that simply don't contain the knowledge. Worth checking the paper directly.
- The "defense-in-depth" frame is the natural positioning for our work. Cite Deep Ignorance as the *upstream* layer and trap-induction as the *post-training* layer.

## Citation

arXiv:2508.06601. O'Brien, Casper, Anthony, Korbak, Kirk, Davies, Mishra, Irving, Gal, Biderman.
