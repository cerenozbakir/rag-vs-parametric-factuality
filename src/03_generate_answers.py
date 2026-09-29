"""
Step 3: Run all three conditions (closed-book, BM25-RAG, oracle) through a
small local instruction-tuned model and save predictions.

Run (default model, all 3 conditions):
    python src/03_generate_answers.py --model Qwen/Qwen2.5-1.5B-Instruct

Or restrict to one condition while testing:
    python src/03_generate_answers.py --model Qwen/Qwen2.5-1.5B-Instruct --condition closed_book

Output:
    outputs/predictions_closed_book.jsonl
    outputs/predictions_bm25_rag.jsonl
    outputs/predictions_oracle.jsonl

Each line: {id, question, prediction, gold_answers, condition}

Notes:
- Uses greedy decoding (do_sample=False) for reproducibility -- re-running
  should give (almost) identical outputs, which matters for a graded
  reproducibility requirement.
- The "If you are not sure, say 'I don't know'" instruction is deliberate:
  it gives the model an explicit way to abstain instead of guessing, which is
  what lets us separate "confident fabrication" from "explicit uncertainty"
  in the hallucination-proxy step (05_classify_hallucination.py / RQ2).
- max_new_tokens is small on purpose: SQuAD gold answers are short spans
  (a few words), so we don't want long, hard-to-score chatty responses.
"""
from __future__ import annotations  # allows `str | None` type hints on Python 3.9

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

CLOSED_BOOK_SYSTEM = (
    "You answer factual questions as briefly as possible, using only a short "
    "phrase (no full sentences, no explanations). Rely only on your own "
    "knowledge. If you do not know the answer, respond with exactly: "
    "I don't know."
)

CONTEXT_SYSTEM = (
    "You answer factual questions as briefly as possible, using only a short "
    "phrase (no full sentences, no explanations), based ONLY on the given "
    "context. If the context does not contain the answer, respond with "
    "exactly: I don't know."
)


def build_messages(condition: str, question: str, context: str | None):
    if condition == "closed_book":
        return [
            {"role": "system", "content": CLOSED_BOOK_SYSTEM},
            {"role": "user", "content": f"Question: {question}"},
        ]
    else:  # bm25_rag or oracle: both use a context, just different context source
        return [
            {"role": "system", "content": CONTEXT_SYSTEM},
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {question}",
            },
        ]


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HF model id. Ungated default: Qwen/Qwen2.5-1.5B-Instruct. "
        "Gated alternative (needs HF login + approved access): "
        "meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--condition",
        choices=["closed_book", "bm25_rag", "oracle", "all"],
        default="all",
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    args = parser.parse_args()

    device = pick_device()
    print(f"Using device: {device}")
    print(f"Loading model {args.model} ... (first run downloads the weights, be patient)")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    ).to(device)
    model.eval()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    retrieved = []
    with open(data_dir / "retrieved.jsonl", encoding="utf-8") as f:
        for line in f:
            retrieved.append(json.loads(line))
    print(f"Loaded {len(retrieved)} questions (with retrieval + gold context attached).")

    conditions = ["closed_book", "bm25_rag", "oracle"] if args.condition == "all" else [args.condition]

    @torch.no_grad()
    def generate(messages):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,  # greedy -> deterministic, reproducible
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    for condition in conditions:
        out_path = out_dir / f"predictions_{condition}.jsonl"
        print(f"\n=== Condition: {condition} -> {out_path} ===")
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in tqdm(retrieved, desc=condition):
                if condition == "closed_book":
                    context = None
                elif condition == "oracle":
                    context = ex["gold_context"]
                else:  # bm25_rag: concatenate retrieved passages
                    context = "\n\n".join(ex["retrieved_passages"])

                messages = build_messages(condition, ex["question"], context)
                prediction = generate(messages)

                record = {
                    "id": ex["id"],
                    "question": ex["question"],
                    "prediction": prediction,
                    "gold_answers": ex["gold_answers"],
                    "condition": condition,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Done: {out_path}")

    print("\nAll requested conditions finished. Next: python src/04_score_squad.py")


if __name__ == "__main__":
    main()
