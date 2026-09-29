"""
Step 2: Build a BM25 index over data/context_pool.jsonl and, for each sampled
question in data/sample.jsonl, retrieve the top-k passages.

Run:
    python src/02_build_bm25_index.py --k 3

Output:
    data/retrieved.jsonl
    Each line: {id, question, gold_context, gold_answers, retrieved_passages: [str, ...],
                gold_passage_rank: int or null}

`gold_passage_rank` tells you, for each question, where the *correct* passage
landed in BM25's ranking (0 = retrieved first, null = not in the top-k at all).
This number is directly useful evidence for RQ3 and for the Results section
("BM25 retrieved the gold passage in its top-3 for X% of questions").
"""
import argparse
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm import tqdm


def simple_tokenize(text: str):
    # Lowercase, alnum-only tokens. Good enough for BM25; no need for a
    # heavyweight tokenizer here.
    return re.findall(r"[a-z0-9]+", text.lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3, help="top-k passages to retrieve")
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pool_path = data_dir / "context_pool.jsonl"
    sample_path = data_dir / "sample.jsonl"

    passages = []
    with open(pool_path, encoding="utf-8") as f:
        for line in f:
            passages.append(json.loads(line))
    print(f"Loaded {len(passages)} candidate passages.")

    print("Tokenizing corpus and building BM25 index (this takes a few seconds)...")
    tokenized_corpus = [simple_tokenize(p["text"]) for p in passages]
    bm25 = BM25Okapi(tokenized_corpus)

    samples = []
    with open(sample_path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} sampled questions.")

    out_path = data_dir / "retrieved.jsonl"
    n_gold_in_topk = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in tqdm(samples, desc="Retrieving"):
            query_tokens = simple_tokenize(ex["question"])
            scores = bm25.get_scores(query_tokens)
            # top-k indices by score, descending
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: args.k]
            retrieved_texts = [passages[i]["text"] for i in top_idx]

            # For analysis: where does the true gold passage rank? (exact text match)
            gold_rank = None
            # get a full ranking to search for the gold passage's position
            full_ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for rank, i in enumerate(full_ranking):
                if passages[i]["text"] == ex["gold_context"]:
                    gold_rank = rank
                    break

            if gold_rank is not None and gold_rank < args.k:
                n_gold_in_topk += 1

            record = {
                "id": ex["id"],
                "question": ex["question"],
                "gold_context": ex["gold_context"],
                "gold_answers": ex["gold_answers"],
                "retrieved_passages": retrieved_texts,
                "gold_passage_rank": gold_rank,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    pct = 100 * n_gold_in_topk / len(samples)
    print(f"\nWrote {out_path}")
    print(
        f"Gold passage was inside BM25's top-{args.k} for {n_gold_in_topk}/{len(samples)} "
        f"questions ({pct:.1f}%). Keep this number -- it's a key RQ3 statistic for the poster."
    )


if __name__ == "__main__":
    main()
