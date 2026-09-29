# RAG vs. Parametric Generation for Factuality

Small-scale study comparing closed-book (parametric) generation, BM25-retrieval-augmented
generation, and an oracle (gold-passage) upper bound on SQuAD factual QA, with a
rule-based hallucination proxy on wrong answers.

Research questions:

- **RQ1**: Does augmenting a small LM with BM25-retrieved context improve factual QA
  accuracy (EM/F1) over closed-book generation?
- **RQ2**: Does retrieval reduce the rate of confidently-stated incorrect answers
  (a hallucination proxy) vs. closed-book generation?
- **RQ3**: How much of any RAG benefit is bounded by retrieval quality — how far
  below an oracle/gold-passage upper bound does realistic (BM25) retrieval fall?

## Method summary

| Component | Choice |
|---|---|
| Dataset | SQuAD v1.1 validation split, N randomly sampled questions (default 250, seed 42) |
| Retrieval | BM25 (`rank_bm25`) over **all unique passages** in the split (not just the sampled gold passages — see comment in `src/01_sample_squad.py`) |
| Generator | Small open-weight instruction-tuned LM, run locally (default: `Qwen/Qwen2.5-1.5B-Instruct`) |
| Conditions | (1) closed-book — question only; (2) BM25-RAG — top-k retrieved passages + question; (3) oracle — gold passage + question |
| Metrics | SQuAD Exact Match / F1 (official normalization, reimplemented in `src/utils.py`); rule-based hallucination-proxy classification (confident fabrication vs. explicit uncertainty) on wrong answers |

## Pipeline

```
src/01_sample_squad.py        -> data/sample.jsonl, data/context_pool.jsonl
src/02_build_bm25_index.py    -> data/retrieved.jsonl
src/03_generate_answers.py    -> outputs/predictions_{condition}.jsonl
src/04_score_squad.py         -> outputs/scores_summary.json, outputs/scores_per_question.csv
src/05_classify_hallucination.py -> outputs/hallucination_summary.json, outputs/hallucination_classification.csv
src/06_make_figures.py        -> outputs/figure_em_f1.png, outputs/figure_hallucination.png
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/01_sample_squad.py --n 250 --seed 42
python src/02_build_bm25_index.py --k 3
python src/03_generate_answers.py --model Qwen/Qwen2.5-1.5B-Instruct
python src/04_score_squad.py
python src/05_classify_hallucination.py
python src/06_make_figures.py
```

## Known limitations

- The hallucination-proxy classification is rule-based (keyword matching on hedging
  phrases), not human-verified. `src/05_classify_hallucination.py` produces a random
  spot-check sample (`outputs/hallucination_spotcheck_sample.csv`) for manual
  verification; the manual-agreement rate is reported alongside the automated numbers.
- BM25 is a lexical retriever; it cannot match paraphrases or synonyms the way a dense
  retriever (e.g., DPR) could. This is a deliberate scope decision for a small,
  reproducible project, not an oversight — see Methodology section of the poster.
- SQuAD v1.1 has no unanswerable questions; all questions have a gold answer in the
  passage, so this setup does not test abstention on genuinely unanswerable questions
  (SQuAD v2.0 would, at the cost of extra complexity).
- Single-seed, single-sample run (no bootstrap confidence intervals) given the project
  timeline; treat point estimates accordingly.

## Reproducibility

All generation uses greedy decoding (`do_sample=False`) for deterministic outputs.
Sampling seed is fixed (default 42) and passed explicitly to `01_sample_squad.py`.

## License / course context

Coursework project for the NLP module, University of Trier. Course reading grounding:
Lewis et al. 2020 (arXiv:2005.11401), Gao et al. 2024 (arXiv:2312.10997), Asai et al.
2023 (arXiv:2310.11511) — full reference list in the submission's appendix PDF.
