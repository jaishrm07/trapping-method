# NeurIPS 2025 Workshop on Lock-LLM — accepted-paper landscape

**Authors / venue / year**: Various / NeurIPS 2025 Workshop / 2025
**Bib key**: lockllm2025workshop
**Read for thread(s)**: 04
**Read on**: 2026-05-02

## TL;DR (3 sentences)

The Lock-LLM workshop covers five "Un-" problems: Un-Distillable, Un-Finetunable, Un-Compressible, Un-Editable, Un-Usable. Of the ~23 confirmed accepted poster papers, **only one squarely targets the Un-Finetunable / parameter-mod-defense lane: Sarker et al. ("Model Immunization by Trapping Harmful Finetuning", id `gfAn827WAW`)**. The rest distribute across distillation/IP protection (fingerprinting, OML, "Undistillable Open Language Models with Teacher Scrambling" by Dionicio/Elahi/Rosati/Sajjad), prompt injection, multimodal red-teaming, agent security, and quantization-adversarial reparameterizations.

## Why we read it

Thread 04 — verify whether competitor *defense* papers exist in the same workshop. Result: Sarker is essentially uncontested in the Un-Finetunable lane within the workshop, which is favorable positioning but also means the workshop alone is not the field map.

## Key claims (with location)

1. Workshop scope (per call for papers): five "Un-" topic areas, of which Un-Finetunable is "techniques to prevent unauthorized parameter updates."
2. Confirmed accepted papers (poster, OpenReview submissions page 1):
   - Model Immunization by Trapping Harmful Finetuning — Sarker, Hakim, Ishmam, Tang, Thomas (`gfAn827WAW`)
   - Undistillable Open Language Models with Teacher Scrambling — Dionicio, Elahi, Rosati, Sajjad (`g9vFg3O8YY`) — Un-Distillable lane, Rosati = same author as RepNoise
   - The Safety Gap Toolkit: Evaluating Hidden Dangers of Open-Source Models — Dombrowski, Bowen, Gleave, Cundy (`0mID6YIwHe`) — eval/redteaming
   - Towards Quantization-Adversarial Reparameterizations — Ma (`XsM1ZMpaII`) — Un-Compressible lane
   - Scalable Fingerprinting of LLMs — Nasery et al. (`dCu5dh7h4v`)
   - OML: A Primitive for Reconciling Open Access with Owner Control in AI Model Distribution — Cheng et al. (`W3ryccayYs`)
   - Several prompt-injection / red-teaming / agent-security papers (out of scope for our threat model).
3. **No other parameter-mod-defense paper in the accepted list that competes directly with Sarker et al. in the LLM setting.**

## Methods we could borrow / discard

- **Borrow**: the workshop framing as a positioning frame. "Un-Finetunable LLMs" is now an explicit research target the community has named; cite the workshop and Sarker as the canonical entry.
- **Discard**: nothing methodological — this is a venue map, not a method paper.

## Open questions / disagreements

- Cundy/Gleave/Bowen ("Safety Gap Toolkit") are at the same workshop and are the same line of authors as Deep Ignorance (Cundy at AISI/Cohere/Apollo). This suggests the AISI–Apollo Research–EleutherAI cluster is the *evaluation* arm of the field, not the *defense* arm.
- Rosati shows up at this workshop on the *Un-Distillable* side (Teacher Scrambling), having previously authored RepNoise (Un-Finetunable). Suggests a multi-lane cross-pollination.

## Citation

NeurIPS 2025 Workshop on Lock-LLM. https://lock-llm.github.io/ , https://openreview.net/group?id=NeurIPS.cc/2025/Workshop/Lock-LLM
