"""
src/download_blimp_cache.py

Downloads the BLiMP dataset from HuggingFace and saves each paradigm as a
.jsonl file under data/blimp_cache/, which eval_checkpoints.py reads for
BLiMP scoring at each checkpoint.

Usage:
    python src/download_blimp_cache.py
    python src/download_blimp_cache.py --out_dir data/blimp_cache
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path,
                   default=Path(__file__).parent.parent / "data" / "blimp_cache")
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Run: pip install datasets")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading BLiMP from HuggingFace (nyu-mll/blimp)…")
    ds = load_dataset("nyu-mll/blimp", "all", split="train")

    # Group by paradigm and write one .jsonl per paradigm
    by_paradigm: dict[str, list[dict]] = {}
    for row in ds:
        paradigm = row.get("linguistics_term") or row.get("field") or "unknown"
        by_paradigm.setdefault(paradigm, []).append({
            "sentence_good": row["sentence_good"],
            "sentence_bad":  row["sentence_bad"],
        })

    for paradigm, items in sorted(by_paradigm.items()):
        out_file = args.out_dir / f"{paradigm}.jsonl"
        out_file.write_text(
            "\n".join(json.dumps(item) for item in items),
            encoding="utf-8",
        )

    total = sum(len(v) for v in by_paradigm.values())
    log.info("Saved %d items across %d paradigms → %s",
             total, len(by_paradigm), args.out_dir)


if __name__ == "__main__":
    main()
