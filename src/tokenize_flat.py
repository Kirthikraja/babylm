"""
src/tokenize_flat.py

Builds the WITHOUT-CHUNKING baseline dataset.

Tokenizes all sub-corpus .txt files using GPT2TokenizerFast and packs
tokens into fixed-length windows of 1024 with 50% stride (512 tokens).
No EOS tokens are inserted at sentence boundaries — this is a standard
sliding-window next-token-prediction setup.

Output: data/flat_100M/*.jsonl
Each line: {"sub_corpus": str, "chunk_id": int, "token_ids": [int, ...], "n_tokens": int}
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm
from transformers import GPT2TokenizerFast

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
    encoding="utf-8" if sys.version_info >= (3, 9) else None,
)
log = logging.getLogger(__name__)

MAX_TOKENS = 1024
STRIDE = 512


def process_file(
    src_path: Path,
    out_dir: Path,
    tokenizer: GPT2TokenizerFast,
    max_tokens: int = MAX_TOKENS,
    stride: int = STRIDE,
    log_every: int = 100_000,
) -> dict:
    sub_corpus = src_path.stem.replace(".train", "")
    out_path = out_dir / f"{sub_corpus}.jsonl"
    log.info("Processing  %s  ->  %s", src_path.name, out_path)

    n_lines = sum(1 for _ in src_path.open("r", encoding="utf-8", errors="replace"))
    log.info("  %d lines to tokenize", n_lines)

    out_dir.mkdir(parents=True, exist_ok=True)
    buffer: list[int] = []
    chunk_id = 0
    n_chunks = 0

    with src_path.open("r", encoding="utf-8", errors="replace") as fh, \
         out_path.open("w", encoding="utf-8") as out_fh:

        for line_no, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            buffer.extend(tokenizer.encode(line, add_special_tokens=False))

            # Flush complete windows from the buffer
            while len(buffer) >= max_tokens:
                window = buffer[:max_tokens]
                out_fh.write(json.dumps({
                    "sub_corpus": sub_corpus,
                    "chunk_id": chunk_id,
                    "token_ids": window,
                    "n_tokens": max_tokens,
                }) + "\n")
                chunk_id += 1
                n_chunks += 1
                buffer = buffer[stride:]   # advance by stride (overlap = max_tokens - stride)

            if line_no % log_every == 0:
                log.info("  line %d / %d  |  chunks so far: %d  |  buffer: %d tokens",
                         line_no, n_lines, n_chunks, len(buffer))

    log.info("  -> %d windows written for %s", n_chunks, sub_corpus)
    return {"sub_corpus": sub_corpus, "n_chunks": n_chunks}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build flat (no-EOS) sliding-window dataset from BabyLM sub-corpora.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--corpus_scale", choices=["10M", "100M"], default="100M")
    p.add_argument("--input_dir", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--stride", type=int, default=STRIDE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(__file__).parent.parent
    in_dir = args.input_dir or base / "data" / "raw" / f"babylm_{args.corpus_scale}"
    out_dir = args.output_dir or base / "data" / f"flat_{args.corpus_scale}"

    if not in_dir.exists():
        log.error("Input directory not found: %s", in_dir)
        sys.exit(1)

    files = sorted(in_dir.glob("*.train.txt"))
    if not files:
        log.error("No *.train.txt files in %s", in_dir)
        sys.exit(1)

    # Load from local cache if available, otherwise download once
    tokenizer_path = base / "tokenizer"
    src = str(tokenizer_path) if tokenizer_path.exists() else "gpt2"
    log.info("Loading GPT2TokenizerFast from %s ...", src)
    tokenizer = GPT2TokenizerFast.from_pretrained(src)

    for fpath in files:
        process_file(fpath, out_dir, tokenizer, args.max_tokens, args.stride)

    log.info("Done. Flat dataset written to %s", out_dir)


if __name__ == "__main__":
    main()
