"""
Step 6: Turn outputs/scores_summary.json and outputs/hallucination_summary.json
into two poster-ready PNG figures at 300 PPI (comfortably above the 150 PPI
minimum in the submission requirements).

Run:
    python src/06_make_figures.py

Writes:
    outputs/figure_em_f1.png             -> grouped bar: EM & F1 per condition (RQ1, RQ3)
    outputs/figure_hallucination.png     -> stacked bar: confident fabrication vs
                                             explicit uncertainty per condition (RQ2)

Colors follow a validated colorblind-safe categorical palette (see the
dataviz reference this project used); do not swap in arbitrary colors
without re-checking contrast/CVD-safety.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

# --- palette (validated categorical slots 1/2/3, light-mode chart chrome) ---
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "text.color": INK_PRIMARY,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.size": 13,
    }
)

CONDITION_LABELS = {
    "closed_book": "Closed-book",
    "bm25_rag": "BM25-RAG",
    "oracle": "Oracle",
}


def make_em_f1_figure(out_dir: Path):
    with open(out_dir / "scores_summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    conditions = [c for c in ["closed_book", "bm25_rag", "oracle"] if c in summary]
    if not conditions:
        print("[skip] no scores_summary.json data found yet.")
        return

    labels = [CONDITION_LABELS[c] for c in conditions]
    em_vals = [summary[c]["em"] for c in conditions]
    f1_vals = [summary[c]["f1"] for c in conditions]

    x = range(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    bars_em = ax.bar([i - width / 2 for i in x], em_vals, width, label="Exact Match", color=BLUE)
    bars_f1 = ax.bar([i + width / 2 for i in x], f1_vals, width, label="F1", color=ORANGE)

    for bars in (bars_em, bars_f1):
        for b in bars:
            h = b.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                color=INK_PRIMARY,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("QA accuracy by condition (RQ1, RQ3)", fontsize=15, color=INK_PRIMARY, pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    # RQ3 annotation if available
    if "rq3_pct_gap_closed_by_bm25" in summary:
        pct = summary["rq3_pct_gap_closed_by_bm25"]
        ax.text(
            0.98,
            0.98,
            f"BM25-RAG closes {pct:.0f}% of the\nclosed-book -> oracle F1 gap",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            color=INK_SECONDARY,
        )

    fig.tight_layout()
    out_path = out_dir / "figure_em_f1.png"
    fig.savefig(out_path, dpi=300)
    print(f"Wrote {out_path}")


def make_hallucination_figure(out_dir: Path):
    path = out_dir / "hallucination_summary.json"
    if not path.exists():
        print("[skip] no hallucination_summary.json found yet.")
        return
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)

    conditions = [c for c in ["closed_book", "bm25_rag"] if c in summary]
    if not conditions:
        print("[skip] hallucination_summary.json has no matching conditions.")
        return

    labels = [CONDITION_LABELS[c] for c in conditions]
    fab = [summary[c]["pct_confident_fabrication"] for c in conditions]
    unc = [summary[c]["pct_explicit_uncertainty"] for c in conditions]

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    x = range(len(conditions))
    bars_fab = ax.bar(x, fab, width=0.5, label="Confident fabrication", color=BLUE)
    bars_unc = ax.bar(x, unc, width=0.5, bottom=fab, label="Explicit uncertainty", color=AQUA)

    for i, (f_val, u_val) in enumerate(zip(fab, unc)):
        if f_val > 0:
            ax.annotate(f"{f_val:.0f}%", xy=(i, f_val / 2), ha="center", va="center", color="white", fontsize=12)
        if u_val > 0:
            ax.annotate(
                f"{u_val:.0f}%", xy=(i, f_val + u_val / 2), ha="center", va="center", color="white", fontsize=12
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Share of wrong answers (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Hallucination proxy: wrong-answer type (RQ2)", fontsize=15, color=INK_PRIMARY, pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    fig.tight_layout()
    out_path = out_dir / "figure_hallucination.png"
    fig.savefig(out_path, dpi=300)
    print(f"Wrote {out_path}")


def main():
    out_dir = Path("outputs")
    make_em_f1_figure(out_dir)
    make_hallucination_figure(out_dir)


if __name__ == "__main__":
    main()
