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


SOURCE_FIELDS = ("sub_corpus", "source", "dataset", "domain", "split_name")


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
    source_field: str | None = None
    file_handles: dict[str, object] = {}
    n_lines = 0

    try:
        for row in ds:
            # Detect source field on first row
            if n_lines == 0:
                for f in SOURCE_FIELDS:
                    if f in row:
                        source_field = f
                        break
                if source_field:
                    log.info("  Detected sub-corpus field: '%s' — writing per-subcorpus files", source_field)
                else:
                    log.warning("  No sub-corpus field found (%s) — writing combined corpus.train.txt", SOURCE_FIELDS)

            text = row["text"]
            if source_field and (sub := row.get(source_field)):
                key = str(sub).lower().replace(" ", "_").replace("-", "_")
                if key not in file_handles:
                    p = out_dir / f"{key}.train.txt"
                    file_handles[key] = p.open("w", encoding="utf-8")
                    log.info("  Opened %s", p)
                file_handles[key].write(text + "\n")
            else:
                if "corpus" not in file_handles:
                    file_handles["corpus"] = (out_dir / "corpus.train.txt").open("w", encoding="utf-8")
                file_handles["corpus"].write(text + "\n")

            n_lines += 1
            if n_lines % 500_000 == 0:
                log.info("  %d lines written ...", n_lines)
    finally:
        for fh in file_handles.values():
            fh.close()

    log.info("Wrote %d lines across %d file(s):", n_lines, len(file_handles))
    for key in sorted(file_handles):
        p = out_dir / f"{key}.train.txt"
        log.info("  %s", p)
    log.info("Done. Data saved to %s", out_dir)


if __name__ == "__main__":
    main()
