"""
src/eval_checkpoints.py

Evaluates FBT (+ optionally BLiMP) on every saved training checkpoint,
producing results/<condition>/checkpoint_evals.json for Figure-10-style plots.

Usage:
    python src/eval_checkpoints.py --condition chunked
    python src/eval_checkpoints.py --condition flat --fbt_only
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO, stream=sys.stdout,
)
log = logging.getLogger(__name__)

MASK_TOKEN = "[MASK]"


# ── FBT helpers ───────────────────────────────────────────────────────────────

def location_log_prob(model, tokenizer, context, location, device):
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    loc_ids = tokenizer.encode(" " + location.strip(), add_special_tokens=False)
    input_ids = torch.tensor([ctx_ids + loc_ids]).to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    total = sum(log_probs[len(ctx_ids) + i - 1, tok_id].item()
                for i, tok_id in enumerate(loc_ids))
    return total / len(loc_ids)


def eval_fbt(model, tokenizer, df, device):
    results = []
    for _, row in df.iterrows():
        passage   = str(row["passage"])
        condition = str(row["condition"]).strip()
        start_loc = str(row["start"]).strip()
        end_loc   = str(row["end"]).strip()
        context   = passage.replace(MASK_TOKEN, "").rstrip()
        lp_start  = location_log_prob(model, tokenizer, context, start_loc, device)
        lp_end    = location_log_prob(model, tokenizer, context, end_loc,   device)
        correct   = (lp_start > lp_end) if "false" in condition.lower() else (lp_end > lp_start)
        results.append({"condition": condition, "correct": bool(correct)})

    total    = len(results)
    overall  = sum(r["correct"] for r in results) / total if total else 0.0
    fb_items = [r for r in results if "false" in r["condition"].lower()]
    tb_items = [r for r in results if "true"  in r["condition"].lower()]
    return {
        "overall":       overall,
        "false_belief":  sum(r["correct"] for r in fb_items) / len(fb_items) if fb_items else 0.0,
        "true_belief":   sum(r["correct"] for r in tb_items) / len(tb_items) if tb_items else 0.0,
    }


# ── BLiMP helpers ─────────────────────────────────────────────────────────────

def sentence_mean_lp(model, tokenizer, sentence, device):
    ids = tokenizer.encode(sentence, add_special_tokens=False)
    if len(ids) < 2:
        return float("-inf")
    input_ids = torch.tensor([ids]).to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    total = sum(log_probs[i - 1, ids[i]].item() for i in range(1, len(ids)))
    return total / (len(ids) - 1)


def eval_blimp(model, tokenizer, blimp_items, device):
    correct = sum(
        sentence_mean_lp(model, tokenizer, item["sentence_good"], device) >
        sentence_mean_lp(model, tokenizer, item["sentence_bad"],  device)
        for item in blimp_items
    )
    return correct / len(blimp_items) if blimp_items else 0.0


def load_blimp_local(cache_path: Path) -> list[dict]:
    items = []
    if not cache_path.exists():
        log.warning("BLiMP cache not found at %s — skipping BLiMP", cache_path)
        return items
    for f in sorted(cache_path.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                obj = json.loads(line)
                if "sentence_good" in obj and "sentence_bad" in obj:
                    items.append({"sentence_good": obj["sentence_good"],
                                  "sentence_bad":  obj["sentence_bad"]})
    log.info("Loaded %d BLiMP items from %s", len(items), cache_path)
    return items


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--condition", choices=["chunked", "flat", "balanced"], required=True)
    p.add_argument("--fbt_only",  action="store_true",
                   help="Skip BLiMP evaluation (faster)")
    p.add_argument("--data_path", type=Path, default=None,
                   help="Path to fb.csv (default: data/fbt/fb.csv)")
    p.add_argument("--blimp_cache", type=Path, default=None,
                   help="Directory with BLiMP *.jsonl files")
    return p.parse_args()


def main():
    args = parse_args()
    base        = Path(__file__).parent.parent
    models_dir  = base / "models"  / args.condition
    results_dir = base / "results" / args.condition
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load words_seen mapping from training curves
    curves_path = results_dir / "training_curves.json"
    step_to_words: dict[int, int] = {}
    if curves_path.exists():
        curves = json.loads(curves_path.read_text(encoding="utf-8"))
        step_to_words = {int(s): int(w)
                         for s, w in zip(curves.get("step", []),
                                         curves.get("words_seen", []))}
        log.info("Loaded %d step→words mappings from training curves", len(step_to_words))
    else:
        log.warning("training_curves.json not found — words_seen will be null")

    # Find all checkpoints sorted by step number
    checkpoints = sorted(
        [d for d in models_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not checkpoints:
        log.error("No checkpoint-* directories found in %s", models_dir)
        sys.exit(1)
    log.info("Found %d checkpoints for condition=%s", len(checkpoints), args.condition)

    # Load FBT data
    data_path = args.data_path or base / "data" / "fbt" / "fb.csv"
    if not data_path.exists():
        log.error("FBT stimuli not found at %s", data_path)
        sys.exit(1)
    df = pd.read_csv(data_path)
    if "first_mention" in df.columns and "recent_mention" in df.columns:
        df = df[(df["first_mention"] == "Start") & (df["recent_mention"] == "End")]
    log.info("FBT items: %d", len(df))

    # Load BLiMP data (optional)
    blimp_items: list[dict] = []
    if not args.fbt_only:
        blimp_cache = args.blimp_cache or base / "data" / "blimp_cache"
        blimp_items = load_blimp_local(blimp_cache)
        if not blimp_items:
            log.warning("No BLiMP items found — running FBT only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    checkpoint_results = []

    for ckpt in checkpoints:
        step       = int(ckpt.name.split("-")[1])
        words_seen = step_to_words.get(step)

        log.info("── checkpoint-%d  (%.1fM words) ──",
                 step, (words_seen or 0) / 1_000_000)

        tokenizer = GPT2TokenizerFast.from_pretrained(str(ckpt))
        model     = GPT2LMHeadModel.from_pretrained(str(ckpt)).to(device)
        model.eval()

        entry = {"step": step, "words_seen": words_seen}

        fbt = eval_fbt(model, tokenizer, df, device)
        entry["fbt_overall"]      = round(fbt["overall"],       4)
        entry["fbt_false_belief"] = round(fbt["false_belief"],  4)
        entry["fbt_true_belief"]  = round(fbt["true_belief"],   4)

        if blimp_items:
            blimp_acc = eval_blimp(model, tokenizer, blimp_items, device)
            entry["blimp"] = round(blimp_acc, 4)

        blimp_str = f"  BLiMP={entry['blimp']:.3f}" if "blimp" in entry else ""
        log.info("  FBT=%.3f  (FB=%.3f  TB=%.3f)%s",
                 fbt["overall"], fbt["false_belief"], fbt["true_belief"], blimp_str)

        checkpoint_results.append(entry)

        del model
        torch.cuda.empty_cache()

    out = {
        "condition":   args.condition,
        "checkpoints": checkpoint_results,
    }
    out_path = results_dir / "checkpoint_evals.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Saved %d entries → %s", len(checkpoint_results), out_path)


if __name__ == "__main__":
    main()
