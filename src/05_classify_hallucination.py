"""
Step 5: Hallucination-proxy classification (answers RQ2).

For every WRONG answer (EM=0) in a given condition, classify it as:
    - "explicit_uncertainty"    -> model hedged / said it didn't know
    - "confident_fabrication"   -> model stated a concrete but incorrect answer

This is a RULE-BASED proxy (keyword matching on hedge phrases -- see
UNCERTAINTY_MARKERS in utils.py), not a human-verified gold label. That is a
real limitation: say so explicitly in the poster's Results/Limitations text.
To mitigate it cheaply, this script also draws a random spot-check sample for
you to eyeball by hand (recommended: at least 30 rows) before trusting the
numbers.

Run:
    python src/05_classify_hallucination.py

Reads:
    outputs/predictions_closed_book.jsonl
    outputs/predictions_bm25_rag.jsonl   (optional, for the RQ2 comparison)

Writes:
    outputs/hallucination_classification.csv   (all wrong answers, both conditions)
    outputs/hallucination_summary.json         (% confident_fabrication per condition)
    outputs/hallucination_spotcheck_sample.csv (random subsample to manually verify)
"""
import csv
import json
import random
from pathlib import Path

from utils import compute_exact, classify_wrong_answer, metric_max_over_ground_truths

CONDITIONS_TO_CHECK = ["closed_book", "bm25_rag"]  # oracle is expected to rarely be wrong
SPOTCHECK_N = 30
SEED = 42


def main():
    out_dir = Path("outputs")
    rows = []

    for condition in CONDITIONS_TO_CHECK:
        pred_path = out_dir / f"predictions_{condition}.jsonl"
        if not pred_path.exists():
            print(f"[skip] {pred_path} not found yet.")
            continue

        n_total, n_wrong = 0, 0
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                n_total += 1
                em = metric_max_over_ground_truths(compute_exact, ex["prediction"], ex["gold_answers"])
                if em == 0:
                    n_wrong += 1
                    label = classify_wrong_answer(ex["prediction"])
                    rows.append(
                        {
                            "id": ex["id"],
                            "condition": condition,
                            "question": ex["question"],
                            "prediction": ex["prediction"],
                            "gold_answers": " | ".join(ex["gold_answers"]),
                            "label": label,
                        }
                    )
        print(f"{condition}: {n_wrong}/{n_total} answers were wrong (EM=0) and classified.")

    if not rows:
        print("Nothing to classify yet -- run 03 and make sure predictions exist.")
        return

    with open(out_dir / "hallucination_classification.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary stats per condition
    summary = {}
    for condition in CONDITIONS_TO_CHECK:
        cond_rows = [r for r in rows if r["condition"] == condition]
        if not cond_rows:
            continue
        n_fab = sum(1 for r in cond_rows if r["label"] == "confident_fabrication")
        n_unc = sum(1 for r in cond_rows if r["label"] == "explicit_uncertainty")
        total = len(cond_rows)
        summary[condition] = {
            "n_wrong_answers": total,
            "pct_confident_fabrication": round(100 * n_fab / total, 1),
            "pct_explicit_uncertainty": round(100 * n_unc / total, 1),
        }
    with open(out_dir / "hallucination_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary (this is your RQ2 headline table):")
    print(json.dumps(summary, indent=2))

    # Spot-check sample for manual verification of the rule-based labels
    random.seed(SEED)
    sample = rows if len(rows) <= SPOTCHECK_N else random.sample(rows, SPOTCHECK_N)
    spotcheck_path = out_dir / "hallucination_spotcheck_sample.csv"
    with open(spotcheck_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(sample[0].keys()) + ["manual_agree_y_n"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sample:
            r_copy = dict(r)
            r_copy["manual_agree_y_n"] = ""  # fill this in by hand
            writer.writerow(r_copy)

    print(
        f"\nWrote {spotcheck_path} with {len(sample)} rows.\n"
        "IMPORTANT (do this before writing Results): open this CSV and fill in "
        "'manual_agree_y_n' for each row (does the rule-based label look right to "
        "you?). Report the agreement rate on the poster as a sanity check on the "
        "hallucination-proxy metric -- this preempts the obvious grader question "
        "'how do you know your hallucination proxy is any good?'"
    )


if __name__ == "__main__":
    main()
