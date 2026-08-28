"""
src/eval_ewok.py

Evaluates a GPT-2 checkpoint on EWoK-Core-1.0.

EWoK tests world knowledge via minimal pairs: for each item the model sees two
sentences that differ in one target word; it should assign higher probability
to the plausible sentence. Accuracy = fraction of items where model prefers
the correct sentence.

Dataset: ewok-core/ewok-core-1.0 on HuggingFace

Usage:
    python src/eval_ewok.py --model_path models/chunked/final --condition chunked
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

EWOK_DATASET = "ewok-core/ewok-core-1.0"


def sentence_log_prob(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    sentence: str,
    device: torch.device,
) -> float:
    """Sum of log-probabilities of all tokens in *sentence* under *model*."""
    ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(ids, labels=ids)
    # out.loss is mean NLL; multiply by n_tokens to get total NLL, negate for log-prob
    n_tokens = ids.size(1)
    return -out.loss.item() * n_tokens


def evaluate(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    dataset,
    device: torch.device,
    batch_size: int = 1,
) -> dict:
    correct = total = 0
    by_domain: dict[str, list[bool]] = {}

    for item in tqdm(dataset, desc="EWoK", ncols=90):
        sent_good = item["sentence1"]  # plausible sentence
        sent_bad  = item["sentence2"]  # implausible sentence
        domain    = item.get("domain", "unknown")

        lp_good = sentence_log_prob(model, tokenizer, sent_good, device)
        lp_bad  = sentence_log_prob(model, tokenizer, sent_bad,  device)

        hit = lp_good > lp_bad
        correct += int(hit)
        total += 1
        by_domain.setdefault(domain, []).append(hit)

    overall_acc = correct / total if total > 0 else 0.0
    domain_acc = {d: sum(hits) / len(hits) for d, hits in by_domain.items()}

    return {
        "overall_accuracy": overall_acc,
        "n_items": total,
        "by_domain": domain_acc,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--condition", choices=["chunked", "flat"], required=True)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--cache_dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(__file__).parent.parent
    out_dir = args.output_dir or base / "results" / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading model from %s ...", args.model_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)
    model.eval()

    log.info("Loading EWoK dataset (%s split) ...", args.split)
    ds = load_dataset(
        EWOK_DATASET,
        split=args.split,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        trust_remote_code=True,
    )
    log.info("%d items", len(ds))

    results = evaluate(model, tokenizer, ds, device)
    log.info("EWoK overall accuracy: %.4f  (%d items)", results["overall_accuracy"], results["n_items"])
    for domain, acc in sorted(results["by_domain"].items()):
        log.info("  %-30s  %.4f", domain, acc)

    out_path = out_dir / "ewok.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
