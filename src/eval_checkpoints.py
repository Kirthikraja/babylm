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
    if not ctx_ids or not loc_ids:
        return float("-inf")
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
        cue       = str(row.get("knowledge_cue", "")).strip().lower()
        context   = passage.replace(MASK_TOKEN, "").rstrip()
        lp_start  = location_log_prob(model, tokenizer, context, start_loc, device)
        lp_end    = location_log_prob(model, tokenizer, context, end_loc,   device)
        correct   = (lp_start > lp_end) if "false" in condition.lower() else (lp_end > lp_start)
        results.append({"condition": condition, "correct": bool(correct), "cue": cue})

    total    = len(results)
    overall  = sum(r["correct"] for r in results) / total if total else 0.0
    fb_items = [r for r in results if "false" in r["condition"].lower()]
    tb_items = [r for r in results if "true"  in r["condition"].lower()]

    def _acc(items):
        return sum(r["correct"] for r in items) / len(items) if items else None

    return {
        "overall":               overall,
        "false_belief":          _acc(fb_items) or 0.0,
        "true_belief":           _acc(tb_items) or 0.0,
        "explicit_false_belief": _acc([r for r in fb_items if r["cue"] == "explicit"]),
        "implicit_false_belief": _acc([r for r in fb_items if r["cue"] == "implicit"]),
        "explicit_true_belief":  _acc([r for r in tb_items if r["cue"] == "explicit"]),
        "implicit_true_belief":  _acc([r for r in tb_items if r["cue"] == "implicit"]),
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


# ── EWoK helpers ──────────────────────────────────────────────────────────────

def _conditional_mean_lp(model, tokenizer, context: str, target: str, device) -> float:
    """Mean log-prob of target tokens conditioned on context."""
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    tgt_ids = tokenizer.encode(" " + target.strip(), add_special_tokens=False)
    if not ctx_ids or not tgt_ids:
        return float("-inf")
    input_ids = torch.tensor([ctx_ids + tgt_ids]).to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    total = sum(log_probs[len(ctx_ids) + i - 1, tok_id].item()
                for i, tok_id in enumerate(tgt_ids))
    return total / len(tgt_ids)


def eval_ewok(model, tokenizer, ewok_items: list[dict], device) -> dict:
    """
    EWoK-Core accuracy: P(target_true | context) > P(target_false | context).
    Returns overall accuracy and per-domain breakdown.
    """
    correct_by_domain: dict[str, list[bool]] = {}
    for item in ewok_items:
        context      = item.get("context", "") or ""
        target_true  = item.get("target_true",  "")
        target_false = item.get("target_false", "")
        domain       = item.get("domain", "unknown")
        if not target_true or not target_false:
            continue
        if context:
            lp_true  = _conditional_mean_lp(model, tokenizer, context, target_true,  device)
            lp_false = _conditional_mean_lp(model, tokenizer, context, target_false, device)
        else:
            lp_true  = sentence_mean_lp(model, tokenizer, target_true,  device)
            lp_false = sentence_mean_lp(model, tokenizer, target_false, device)
        correct_by_domain.setdefault(domain, []).append(lp_true > lp_false)

    all_results = [v for vals in correct_by_domain.values() for v in vals]
    overall     = sum(all_results) / len(all_results) if all_results else 0.0
    by_domain   = {d: sum(vs) / len(vs) for d, vs in correct_by_domain.items() if vs}
    return {"overall": overall, "by_domain": by_domain, "n_items": len(all_results)}


def load_ewok_local(cache_path: Path) -> list[dict]:
    items = []
    if not cache_path.exists():
        log.warning("EWoK cache not found at %s — skipping EWoK", cache_path)
        log.warning("  Run: python scripts/download_ewok.py to fetch the data")
        return items
    for f in sorted(cache_path.glob("*.jsonl")):
        domain = f.stem  # filename = domain name
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Support multiple column-name conventions across dataset versions
            true_key  = next((k for k in ("target_true",  "sentence_good", "correct_sentence")   if k in obj), None)
            false_key = next((k for k in ("target_false", "sentence_bad",  "incorrect_sentence") if k in obj), None)
            if true_key and false_key:
                items.append({
                    "context":      obj.get("context", ""),
                    "target_true":  obj[true_key],
                    "target_false": obj[false_key],
                    "domain":       obj.get("domain", domain),
                })
    log.info("Loaded %d EWoK items from %s", len(items), cache_path)
    return items


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--condition",  choices=["chunked", "flat", "balanced"], required=True)
    p.add_argument("--fbt_only",   action="store_true",
                   help="Skip BLiMP and EWoK evaluation (faster)")
    p.add_argument("--data_path",  type=Path, default=None,
                   help="Path to fb.csv (default: data/fbt/fb.csv)")
    p.add_argument("--blimp_cache", type=Path, default=None,
                   help="Directory with BLiMP *.jsonl files")
    p.add_argument("--ewok_cache",  type=Path, default=None,
                   help="Directory with EWoK *.jsonl files (default: data/ewok_cache)")
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

    # Load BLiMP + EWoK data (optional)
    blimp_items: list[dict] = []
    ewok_items:  list[dict] = []
    if not args.fbt_only:
        blimp_cache = args.blimp_cache or base / "data" / "blimp_cache"
        blimp_items = load_blimp_local(blimp_cache)
        if not blimp_items:
            log.warning("No BLiMP items found — skipping BLiMP")

        ewok_cache = args.ewok_cache or base / "data" / "ewok_cache"
        ewok_items = load_ewok_local(ewok_cache)
        if not ewok_items:
            log.warning("No EWoK items found — skipping EWoK")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Load tokenizer once — it doesn't change between checkpoints. Early checkpoints
    # often don't save tokenizer files, so loading per-checkpoint silently falls back
    # to a broken/default tokenizer, making encode() return empty lists → all -inf → 0.0 scores.
    first_ckpt_with_tokenizer = next(
        (d for d in checkpoints if (d / "tokenizer.json").exists()
         or (d / "vocab.json").exists()),
        None,
    )
    tokenizer_path = str(first_ckpt_with_tokenizer) if first_ckpt_with_tokenizer else "gpt2"
    log.info("Loading tokenizer from: %s", tokenizer_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)

    out_path = results_dir / "checkpoint_evals.json"

    # Resume from existing results if present
    done_steps: set[int] = set()
    checkpoint_results: list[dict] = []
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            checkpoint_results = prev.get("checkpoints", [])
            done_steps = {e["step"] for e in checkpoint_results}
            log.info("Resuming: %d checkpoints already done, %d remaining",
                     len(done_steps), len(checkpoints) - len(done_steps))
        except Exception as exc:
            log.warning("Could not load existing results (%s) — starting fresh", exc)

    for ckpt in checkpoints:
        step       = int(ckpt.name.split("-")[1])
        words_seen = step_to_words.get(step)

        if step in done_steps:
            log.info("Skipping checkpoint-%d (already done)", step)
            continue

        log.info("── checkpoint-%d  (%.1fM words) ──",
                 step, (words_seen or 0) / 1_000_000)

        model     = GPT2LMHeadModel.from_pretrained(str(ckpt)).to(device)
        model.eval()

        entry = {"step": step, "words_seen": words_seen}

        fbt = eval_fbt(model, tokenizer, df, device)
        entry["fbt_overall"]      = round(fbt["overall"],      4)
        entry["fbt_false_belief"] = round(fbt["false_belief"], 4)
        entry["fbt_true_belief"]  = round(fbt["true_belief"],  4)
        for key in ("explicit_false_belief", "implicit_false_belief",
                    "explicit_true_belief",  "implicit_true_belief"):
            v = fbt.get(key)
            entry[f"fbt_{key}"] = round(v, 4) if v is not None else None

        if blimp_items:
            blimp_acc = eval_blimp(model, tokenizer, blimp_items, device)
            entry["blimp"] = round(blimp_acc, 4)

        if ewok_items:
            ewok_result = eval_ewok(model, tokenizer, ewok_items, device)
            entry["ewok"]            = round(ewok_result["overall"], 4)
            entry["ewok_by_domain"]  = {d: round(v, 4) for d, v in ewok_result["by_domain"].items()}
            entry["ewok_n_items"]    = ewok_result["n_items"]

        blimp_str = f"  BLiMP={entry['blimp']:.3f}" if "blimp" in entry else ""
        ewok_str  = f"  EWoK={entry['ewok']:.3f}"   if "ewok"  in entry else ""
        log.info("  FBT=%.3f  (FB=%.3f  TB=%.3f)  [exp-FB=%.3f  imp-FB=%.3f]%s%s",
                 fbt["overall"], fbt["false_belief"], fbt["true_belief"],
                 fbt.get("explicit_false_belief") or 0.0,
                 fbt.get("implicit_false_belief") or 0.0,
                 blimp_str, ewok_str)

        checkpoint_results.append(entry)

        # Save incrementally so a job timeout doesn't lose completed work
        sorted_results = sorted(checkpoint_results, key=lambda e: e["step"])
        out_path.write_text(
            json.dumps({"condition": args.condition, "checkpoints": sorted_results}, indent=2),
            encoding="utf-8",
        )

        del model
        torch.cuda.empty_cache()

    log.info("Done. %d total entries in %s", len(checkpoint_results), out_path)


if __name__ == "__main__":
    main()
