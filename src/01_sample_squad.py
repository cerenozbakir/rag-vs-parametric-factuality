"""
Step 1: Download SQuAD, sample N questions, and build the BM25 retrieval pool.

Run:
    python src/01_sample_squad.py --n 250 --seed 42 --squad_version v1

Outputs:
    data/sample.jsonl        -> the sampled questions (with gold passage + gold answers)
    data/context_pool.jsonl  -> ALL unique context paragraphs in the split (the BM25 search space)

Design note (important, read this):
We deliberately build the BM25 index over ALL unique paragraphs in the SQuAD
validation split (~2,000 paragraphs from ~35 Wikipedia articles), NOT just the
gold passages of the sampled questions. If we only indexed the ~250 gold
passages, BM25 would trivially find the right one almost every time (small
haystack), which would make BM25-RAG look artificially close to Oracle and
defeat the point of RQ3 (how much realistic retrieval falls short of the
oracle upper bound). Indexing the full paragraph pool makes retrieval a real
needle-in-haystack problem, which is the honest way to test this.
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=250, help="number of questions to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--squad_version",
        choices=["v1", "v2"],
        default="v1",
        help="v1 = squad (all questions answerable). v2 = squad_v2 (includes "
        "unanswerable questions -- more realistic but adds complexity; "
        "v1 is recommended for a first pass given the timeline).",
    )
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()

    hf_name = "rajpurkar/squad" if args.squad_version == "v1" else "rajpurkar/squad_v2"
    print(f"Loading {hf_name} validation split from Hugging Face...")
    ds = load_dataset(hf_name, split="validation")
    print(f"Loaded {len(ds)} question-answer pairs.")

    # For v2, drop unanswerable questions for this lean version of the study
    # (unanswerable questions need a 4th "abstain correctly" condition that's
    # out of scope here -- note this as a limitation in the Results section).
    if args.squad_version == "v2":
        before = len(ds)
        ds = ds.filter(lambda ex: len(ex["answers"]["text"]) > 0)
        print(f"Filtered out unanswerable questions: {before} -> {len(ds)}")

    random.seed(args.seed)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    sample_indices = sorted(indices[: args.n])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Write the sampled questions
    sample_path = out_dir / "sample.jsonl"
    with open(sample_path, "w", encoding="utf-8") as f:
        for idx in sample_indices:
            ex = ds[idx]
            record = {
                "id": ex["id"],
                "question": ex["question"],
                "gold_context": ex["context"],
                "gold_answers": ex["answers"]["text"],  # list of acceptable answer strings
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sample_indices)} sampled questions to {sample_path}")

    # 2) Write the full unique-context pool (the BM25 search space)
    unique_contexts = list(dict.fromkeys(ds["context"]))  # preserves order, dedups
    pool_path = out_dir / "context_pool.jsonl"
    with open(pool_path, "w", encoding="utf-8") as f:
        for i, ctx in enumerate(unique_contexts):
            f.write(json.dumps({"passage_id": i, "text": ctx}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(unique_contexts)} unique context passages to {pool_path}")
    print("\nDone. Sanity check: open data/sample.jsonl and skim a few rows.")


if __name__ == "__main__":
    main()
