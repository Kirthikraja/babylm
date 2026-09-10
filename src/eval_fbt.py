"""
src/eval_fbt.py

Evaluates a GPT-2 checkpoint on the False Belief Test (FBT) stimuli from
Trott et al. (2023), following the methodology in Kouwenhoven et al. (2026).

Each item is a short story ending with a cue of the form:
    "X thinks the book is on the [MASK]."   (Explicit)
    "X goes to get the book from the [MASK]."  (Implicit)

The model scores two candidate location completions (start vs end).
  - False Belief condition: correct answer = start location (where A put the object)
  - True Belief  condition: correct answer = end   location (where B moved it to)

Log-probabilities are normalised by token count (marginalisation for subword
tokenisation, following Pimentel & Meister 2024).

Dataset: https://github.com/seantrott/nlm-fb (data/stims/fb.csv)

Usage:
    python src/eval_fbt.py --model_path models/chunked/final --condition chunked
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

FBT_CSV_URL = "https://raw.githubusercontent.com/seantrott/nlm-fb/main/data/stims/fb.csv"
MASK_TOKEN  = "[MASK]"


def location_log_prob(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    context: str,
    location: str,
    device: torch.device,
) -> float:
    """Mean token log-probability of *location* given *context*.

    Normalising by token count makes scores comparable across locations
    that tokenise into different numbers of subword pieces.
    """
    ctx_ids  = tokenizer.encode(context, add_special_tokens=False)
    # GPT-2 tokeniser encodes a space-prefixed word differently from the bare word.
    # The context already ends with "the " so we encode the bare location word.
    loc_ids  = tokenizer.encode(" " + location.strip(), add_special_tokens=False)

    input_ids = torch.tensor([ctx_ids + loc_ids]).to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]   # (seq_len, vocab)

    log_probs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for i, tok_id in enumerate(loc_ids):
        pos = len(ctx_ids) + i - 1            # position of the preceding token
        total += log_probs[pos, tok_id].item()

    return total / len(loc_ids)               # mean per-token log-prob


def evaluate(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    df: pd.DataFrame,
    device: torch.device,
) -> dict:
    results: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="FBT", ncols=90):
        passage   = str(row["passage"])
        condition = str(row["condition"]).strip()   # "False belief" / "True belief"
        cue_type  = str(row.get("knowledge_cue", "Explicit")).strip()
        start_loc = str(row["start"]).strip()
        end_loc   = str(row["end"]).strip()

        # Strip [MASK] — context ends with "on the " or "from the "
        context = passage.replace(MASK_TOKEN, "").rstrip()

        lp_start = location_log_prob(model, tokenizer, context, start_loc, device)
        lp_end   = location_log_prob(model, tokenizer, context, end_loc,   device)

        # Correct answer depends on condition
        if "false" in condition.lower():
            correct = lp_start > lp_end    # should believe object is at start
        else:
            correct = lp_end > lp_start    # knows object moved to end

        results.append({
            "condition": condition,
            "knowledge_cue": cue_type,
            "correct": bool(correct),
            "lp_start": lp_start,
            "lp_end": lp_end,
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total   = len(results)
    overall = sum(r["correct"] for r in results) / total if total else 0.0

    # By condition
    by_condition: dict[str, float] = {}
    for cond in ("False Belief", "True Belief"):
        sub = [r for r in results if r["condition"] == cond]
        if sub:
            by_condition[cond] = sum(r["correct"] for r in sub) / len(sub)

    # By knowledge cue
    by_cue: dict[str, float] = {}
    for cue in set(r["knowledge_cue"] for r in results):
        sub = [r for r in results if r["knowledge_cue"] == cue]
        if sub:
            by_cue[cue] = sum(r["correct"] for r in sub) / len(sub)

    # By condition × cue (Tom's main analysis dimension)
    by_condition_cue: dict[str, float] = {}
    for cond in ("False Belief", "True Belief"):
        for cue in set(r["knowledge_cue"] for r in results):
            sub = [r for r in results if r["condition"] == cond and r["knowledge_cue"] == cue]
            if sub:
                key = f"{cond} × {cue}"
                by_condition_cue[key] = sum(r["correct"] for r in sub) / len(sub)

    return {
        "overall_accuracy": overall,
        "n_items": total,
        "by_condition": by_condition,
        "by_knowledge_cue": by_cue,
        "by_condition_x_cue": by_condition_cue,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_path",  type=str,  required=True)
    p.add_argument("--condition",   choices=["chunked", "flat"], required=True)
    p.add_argument("--output_dir",  type=Path, default=None)
    p.add_argument("--data_path",   type=Path, default=None,
                   help="Path to fb.csv (default: data/fbt/fb.csv)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    base   = Path(__file__).parent.parent
    out_dir = args.output_dir or base / "results" / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = args.data_path or base / "data" / "fbt" / "fb.csv"
    if not data_path.exists():
        log.error("FBT stimuli not found at %s", data_path)
        log.error("Download with: wget -P data/fbt/ %s", FBT_CSV_URL)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading model from %s ...", args.model_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_path)
    model     = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)
    model.eval()

    log.info("Loading FBT stimuli from %s ...", data_path)
    df = pd.read_csv(data_path)
    log.info("%d total rows in CSV", len(df))

    # Filter to the balanced 192-item subset used by Kouwenhoven et al. (2026):
    # first_mention=Start, recent_mention=End (start mentioned first, end most recently)
    # This controls for recency and primacy biases while keeping the dataset balanced.
    if "first_mention" in df.columns and "recent_mention" in df.columns:
        df = df[(df["first_mention"] == "Start") & (df["recent_mention"] == "End")]
        log.info("%d items after filtering to first=Start, recent=End", len(df))
    else:
        log.warning("Mention columns not found — using all %d rows", len(df))

    results = evaluate(model, tokenizer, df, device)

    log.info("FBT overall accuracy: %.4f  (%d items)", results["overall_accuracy"], results["n_items"])
    log.info("By condition:")
    for k, v in results["by_condition"].items():
        log.info("  %-20s  %.4f", k, v)
    log.info("By knowledge cue:")
    for k, v in results["by_knowledge_cue"].items():
        log.info("  %-20s  %.4f", k, v)
    log.info("By condition × cue (key analysis):")
    for k, v in results["by_condition_x_cue"].items():
        log.info("  %-35s  %.4f", k, v)

    out_path = out_dir / "fbt.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
