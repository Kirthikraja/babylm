"""
scripts/download_ewok.py

Download EWoK-Core-1.0 from HuggingFace and save as per-domain JSONL files
in data/ewok_cache/ for use by eval_checkpoints.py.

Usage (run once, from repo root):
    python scripts/download_ewok.py
    python scripts/download_ewok.py --out_dir data/ewok_cache
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO, stream=sys.stdout,
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out_dir", type=Path,
                   default=Path(__file__).parent.parent / "data" / "ewok_cache")
    p.add_argument("--dataset", default="ewok-core/ewok-core-1.0",
                   help="HuggingFace dataset ID")
    p.add_argument("--split",   default="test",
                   help="Dataset split to download (test / validation / train)")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets library not installed — run: pip install datasets")
        sys.exit(1)

    log.info("Loading %s (split=%s) ...", args.dataset, args.split)
    try:
        ds = load_dataset(args.dataset, split=args.split)
    except Exception as exc:
        log.error("Failed to load dataset: %s", exc)
        sys.exit(1)

    log.info("Dataset size: %d items", len(ds))
    log.info("Columns: %s", ds.column_names)

    cols = ds.column_names

    # EWoK-Core-1.0 uses a 2×2 design: two contexts × two targets.
    # Each row expands into two evaluation items:
    #   Item A — Context1 should predict Target1 over Target2
    #   Item B — Context2 should predict Target2 over Target1
    if "Context1" in cols and "Context2" in cols and "Target1" in cols and "Target2" in cols:
        log.info("Detected EWoK-Core 2×2 format (Context1/2 × Target1/2)")
        dom_key = next((k for k in ("Domain", "domain", "type", "category") if k in cols), None)

        by_domain: dict[str, list[dict]] = {}
        for row in ds:
            domain = str(row[dom_key]) if dom_key else "unknown"
            # Item A: Context1 → Target1 preferred over Target2
            by_domain.setdefault(domain, []).append({
                "context":      row["Context1"],
                "target_true":  row["Target1"],
                "target_false": row["Target2"],
                "domain":       domain,
                "pair":         "A",
            })
            # Item B: Context2 → Target2 preferred over Target1
            by_domain.setdefault(domain, []).append({
                "context":      row["Context2"],
                "target_true":  row["Target2"],
                "target_false": row["Target1"],
                "domain":       domain,
                "pair":         "B",
            })

    else:
        # Generic format fallback
        true_key  = next((k for k in ("target_true",  "sentence_good", "correct_sentence")   if k in cols), None)
        false_key = next((k for k in ("target_false", "sentence_bad",  "incorrect_sentence") if k in cols), None)
        ctx_key   = next((k for k in ("context", "Context") if k in cols), None)
        dom_key   = next((k for k in ("domain", "Domain", "type", "category") if k in cols), None)

        if not true_key or not false_key:
            log.error(
                "Cannot find target columns. Available: %s\n"
                "Expected: target_true/sentence_good  AND  target_false/sentence_bad\n"
                "     OR:  Context1 + Context2 + Target1 + Target2 (2×2 design)",
                cols,
            )
            sys.exit(1)

        log.info("Using columns — context: %s, true: %s, false: %s, domain: %s",
                 ctx_key, true_key, false_key, dom_key)
        by_domain = {}
        for row in ds:
            domain = str(row[dom_key]) if dom_key else "unknown"
            by_domain.setdefault(domain, []).append({
                "context":      row.get(ctx_key, "") if ctx_key else "",
                "target_true":  row[true_key],
                "target_false": row[false_key],
                "domain":       domain,
            })

    total = 0
    for domain, items in sorted(by_domain.items()):
        out_path = args.out_dir / f"{domain}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        total += len(items)
        log.info("  %s: %d items → %s", domain, len(items), out_path.name)

    log.info("Done. %d total items across %d domains saved to %s",
             total, len(by_domain), args.out_dir)


if __name__ == "__main__":
    main()
