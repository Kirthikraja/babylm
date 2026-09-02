"""
src/download_data.py

Downloads BabyLM-2026-Strict from HuggingFace and saves each sub-corpus
as a plain .txt file under data/raw/babylm_100M/.

Usage:
    python src/download_data.py --split train_100M
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

HF_DATASETS = {
    "100M": "BabyLM-community/BabyLM-2026-Strict",
    "10M":  "BabyLM-community/BabyLM-2026-Strict-Small",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="100M", choices=["10M", "100M"],
                   help="Corpus scale to download")
    p.add_argument("--split", default="train",
                   help="Dataset split (default: train)")
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--cache_dir", type=Path, default=None,
                   help="HuggingFace cache directory (useful on ALICE scratch)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(__file__).parent.parent
    out_dir = args.output_dir or base / "data" / "raw" / f"babylm_{args.scale}"
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_name = HF_DATASETS[args.scale]
    log.info("Downloading %s  split=%s ...", hf_name, args.split)
    ds = load_dataset(
        hf_name,
        split=args.split,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        streaming=True,
    )
    log.info("Streaming and writing corpus ...")
    out_path = out_dir / "corpus.train.txt"
    n_lines = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in ds:
            fh.write(row["text"] + "\n")
            n_lines += 1
            if n_lines % 500_000 == 0:
                log.info("  %d lines written ...", n_lines)
    log.info("Wrote %d lines -> %s", n_lines, out_path)

    log.info("Done. Data saved to %s", out_dir)


if __name__ == "__main__":
    main()
