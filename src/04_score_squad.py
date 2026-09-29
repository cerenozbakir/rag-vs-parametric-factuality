"""
Step 4: Score each condition's predictions with SQuAD EM/F1.

Run:
    python src/04_score_squad.py

Reads:
    outputs/predictions_closed_book.jsonl
    outputs/predictions_bm25_rag.jsonl
    outputs/predictions_oracle.jsonl

Writes:
    outputs/scores_summary.json   -> {condition: {em: float, f1: float, n: int}}
    outputs/scores_per_question.csv -> one row per (condition, question) with em/f1
"""
import csv
import json
from pathlib import Path

from utils import compute_exact, compute_f1, metric_max_over_ground_truths

CONDITIONS = ["closed_book", "bm25_rag", "oracle"]


def main():
    out_dir = Path("outputs")
    summary = {}
    per_question_rows = []

    for condition in CONDITIONS:
        pred_path = out_dir / f"predictions_{condition}.jsonl"
        if not pred_path.exists():
            print(f"[skip] {pred_path} not found yet.")
            continue

        em_total, f1_total, n = 0.0, 0.0, 0
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                pred = ex["prediction"]
                golds = ex["gold_answers"]
                em = metric_max_over_ground_truths(compute_exact, pred, golds)
                f1 = metric_max_over_ground_truths(compute_f1, pred, golds)
                em_total += em
                f1_total += f1
                n += 1
                per_question_rows.append(
                    {
                        "id": ex["id"],
                        "condition": condition,
                        "question": ex["question"],
                        "prediction": pred,
                        "gold_answers": " | ".join(golds),
                        "em": em,
                        "f1": round(f1, 4),
                    }
                )

        summary[condition] = {
            "em": round(100 * em_total / n, 2) if n else None,
            "f1": round(100 * f1_total / n, 2) if n else None,
            "n": n,
        }
        print(f"{condition:12s}  EM={summary[condition]['em']:6.2f}  F1={summary[condition]['f1']:6.2f}  (n={n})")

    with open(out_dir / "scores_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if per_question_rows:
        with open(out_dir / "scores_per_question.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_question_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_question_rows)

    # RQ3 helper: how much of the gap does BM25-RAG close relative to Oracle?
    if "closed_book" in summary and "bm25_rag" in summary and "oracle" in summary:
        cb_f1 = summary["closed_book"]["f1"]
        rag_f1 = summary["bm25_rag"]["f1"]
        oracle_f1 = summary["oracle"]["f1"]
        if oracle_f1 != cb_f1:
            pct_gap_closed = 100 * (rag_f1 - cb_f1) / (oracle_f1 - cb_f1)
            print(
                f"\nRQ3 headline number: BM25-RAG closes {pct_gap_closed:.1f}% of the "
                f"F1 gap between closed-book ({cb_f1}) and oracle ({oracle_f1})."
            )
            summary["rq3_pct_gap_closed_by_bm25"] = round(pct_gap_closed, 1)
            with open(out_dir / "scores_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    print(f"\nWrote {out_dir/'scores_summary.json'} and {out_dir/'scores_per_question.csv'}")


if __name__ == "__main__":
    main()
