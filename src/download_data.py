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

HF_DATASET = "BabyLM-community/BabyLM-2026-Strict"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train_100M",
                   choices=["train_10M", "train_100M", "dev", "test"],
                   help="Dataset split to download")
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--cache_dir", type=Path, default=None,
                   help="HuggingFace cache directory (useful on ALICE scratch)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(__file__).parent.parent
    scale = "100M" if "100M" in args.split else "10M"
    out_dir = args.output_dir or base / "data" / "raw" / f"babylm_{scale}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %s / %s ...", HF_DATASET, args.split)
    ds = load_dataset(
        HF_DATASET,
        split=args.split,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        trust_remote_code=True,
    )
    log.info("Downloaded %d examples", len(ds))

    # Dataset has columns: text, domain (sub-corpus name)
    # Group by domain and write one .txt per sub-corpus
    domains = set(ds["domain"]) if "domain" in ds.column_names else {"all"}
    log.info("Sub-corpora found: %s", sorted(domains))

    if "domain" in ds.column_names:
        from collections import defaultdict
        by_domain: dict[str, list[str]] = defaultdict(list)
        for row in ds:
            by_domain[row["domain"]].append(row["text"])

        for domain, texts in by_domain.items():
            out_path = out_dir / f"{domain}.train.txt"
            with out_path.open("w", encoding="utf-8") as fh:
                fh.write("\n".join(texts))
            log.info("Wrote  %-25s  %d lines  ->  %s", domain, len(texts), out_path)
    else:
        # Fallback: single file
        out_path = out_dir / "corpus.train.txt"
        with out_path.open("w", encoding="utf-8") as fh:
            for row in ds:
                fh.write(row["text"] + "\n")
        log.info("Wrote single corpus file -> %s", out_path)

    log.info("Done. Data saved to %s", out_dir)


if __name__ == "__main__":
    main()
