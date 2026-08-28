"""
src/chunk_corpus.py

Splits BabyLM sub-corpus text files into fixed-length GPT-2 token chunks,
inserting the EOS token (id 50256) at every sentence boundary.

Rationale for EOS-at-sentence-boundaries
-----------------------------------------
Inserting <|endoftext|> after *every* sentence — not just at the end of a
chunk — teaches the model that sentences are genuine completion units.
Without this, the model sees sentence boundaries as arbitrary continuation
points, which degrades perplexity on sentence-final positions and undermines
any downstream probe that relies on end-of-sentence representations (including
the think-vector extraction used in our thesis).

Chunk construction rules
------------------------
1. Tokenise each sentence with GPT2TokenizerFast.
2. Append EOS after the sentence's token ids.
3. Accumulate sentence units into a chunk until the next unit would push us
   past --max_tokens (default 1024, matching GPT-2 Small's context window).
4. At that point, close the current chunk (it already ends with an EOS from
   the last sentence) and open a new one.
5. If a *single* sentence exceeds --max_tokens on its own, truncate it to
   (max_tokens - 1) tokens and append EOS, then log a WARNING.

Output format (one JSON object per line)
-----------------------------------------
{"sub_corpus": str, "chunk_id": int, "token_ids": [int, ...],
 "n_tokens": int, "n_sentences": int}

Usage
-----
  # 10M validation run (default)
  python src/chunk_corpus.py --corpus_scale 10M

  # 100M full run (same code, different paths)
  python src/chunk_corpus.py --corpus_scale 100M
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterator

import nltk
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

EOS_ID: int = 50256          
MAX_TOKENS: int = 1024       



def ensure_punkt() -> None:
    """Download the Punkt sentence tokeniser if it is not already present."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        log.info("Downloading NLTK Punkt tokeniser …")
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)



def stream_lines(path: Path, chunk_bytes: int = 10 * 1024 * 1024) -> Iterator[str]:
    """
    Yield individual text lines from *path* without loading the entire file.

    Reads in ~10 MB blocks to stay memory-efficient on large sub-corpora.
    """
    leftover = ""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            block = fh.read(chunk_bytes)
            if not block:
                if leftover.strip():
                    yield leftover
                break
            block = leftover + block
            lines = block.split("\n")
            leftover = lines[-1]          
            for line in lines[:-1]:
                if line.strip():
                    yield line



def build_chunks(
    lines: list[str],
    tokenizer: GPT2TokenizerFast,
    sub_corpus: str,
    max_tokens: int = MAX_TOKENS,
    use_turn_boundaries: bool = False,
) -> tuple[list[dict], int]:
    """
    Convert a list of raw text lines into EOS-delimited token chunks.

    Segmentation mode
    -----------------
    use_turn_boundaries=False (default, written text)
        Each line is sentence-split with NLTK Punkt; EOS is inserted after
        every sentence so the model learns sentence-completion signals.

    use_turn_boundaries=True (dialogue corpora, e.g. Switchboard)
        Each non-empty line is treated as one conversational turn and is
        followed by EOS.  NLTK Punkt is NOT applied because it is calibrated
        for written text and draws spurious boundaries inside disfluent speech
        (false starts, backchannels, mid-utterance pauses).  The natural
        completion unit in telephone conversation is the speaker turn, not the
        grammatical sentence — so EOS at turn boundaries is the theoretically
        correct signal for this register.

    Returns
    -------
    chunks : list[dict]
        Each dict has keys: sub_corpus, chunk_id, token_ids, n_tokens,
        n_sentences.  In turn-boundary mode, n_sentences counts turns.
    n_truncated : int
        Number of units that exceeded *max_tokens* and had to be truncated.
    """
    chunks: list[dict] = []
    current_ids: list[int] = []
    current_n_sents: int = 0
    chunk_id: int = 0
    n_truncated: int = 0

    def close_chunk() -> None:
        nonlocal chunk_id, current_ids, current_n_sents
        if not current_ids:
            return
        # Ensure the chunk ends with EOS (defensive; normally guaranteed by
        # the sentence/turn unit appending EOS before entering the buffer).
        if current_ids[-1] != EOS_ID:
            current_ids.append(EOS_ID)
        chunks.append({
            "sub_corpus":  sub_corpus,
            "chunk_id":    chunk_id,
            "token_ids":   current_ids,
            "n_tokens":    len(current_ids),
            "n_sentences": current_n_sents,
        })
        chunk_id += 1
        current_ids = []
        current_n_sents = 0

    for line in lines:
        # ── Choose segmentation unit ──────────────────────────────────────
        if use_turn_boundaries:
            # Each line is one conversational turn; skip empty lines.
            sentences = [line.strip()] if line.strip() else []
        else:
            sentences = nltk.sent_tokenize(line)
        for sent in sentences:
            sent_ids: list[int] = tokenizer.encode(sent, add_special_tokens=False)

           
            if len(sent_ids) + 1 > max_tokens:          # +1 for EOS
                n_truncated += 1
                log.warning(
                    "Unit truncated: %d -> %d tokens  |  '%s ...'",
                    len(sent_ids), max_tokens - 1, sent[:80],
                )
               
                close_chunk()
                truncated_unit = sent_ids[: max_tokens - 1] + [EOS_ID]
                chunks.append({
                    "sub_corpus":  sub_corpus,
                    "chunk_id":    chunk_id,
                    "token_ids":   truncated_unit,
                    "n_tokens":    len(truncated_unit),
                    "n_sentences": 1,
                })
                chunk_id += 1
                continue

            unit = sent_ids + [EOS_ID]  

           
            if len(current_ids) + len(unit) <= max_tokens:
                current_ids.extend(unit)
                current_n_sents += 1

            
            else:
                close_chunk()
                current_ids = list(unit)
                current_n_sents = 1

    close_chunk()   
    return chunks, n_truncated



def process_file(
    src_path: Path,
    out_dir: Path,
    tokenizer: GPT2TokenizerFast,
    max_tokens: int = MAX_TOKENS,
    use_turn_boundaries: bool = False,
) -> dict:
    """
    Chunk one sub-corpus file and write the output .jsonl.

    Returns a summary dict with keys: sub_corpus, n_chunks, n_tokens,
    avg_tokens_per_chunk, n_truncated_sentences.
    """
    sub_corpus = src_path.stem.replace(".train", "")
    out_path = out_dir / f"{sub_corpus}.jsonl"
    mode_label = "turn-boundary" if use_turn_boundaries else "sentence-boundary"
    log.info("Processing  %s  ->  %s  [%s]", src_path.name, out_path, mode_label)

  
    n_lines = sum(1 for _ in src_path.open("r", encoding="utf-8", errors="replace")
                  if _.strip())

    
    batch: list[str] = []
    all_chunks: list[dict] = []
    total_truncated = 0
    BATCH_LINES = 5_000

    with tqdm(total=n_lines, desc=sub_corpus, unit="lines", ncols=90) as pbar:
        for line in stream_lines(src_path):
            batch.append(line)
            pbar.update(1)
            if len(batch) >= BATCH_LINES:
                new_chunks, n_trunc = build_chunks(batch, tokenizer, sub_corpus, max_tokens, use_turn_boundaries)
                all_chunks.extend(new_chunks)
                total_truncated += n_trunc
                batch = []
        
        if batch:
            new_chunks, n_trunc = build_chunks(batch, tokenizer, sub_corpus, max_tokens, use_turn_boundaries)
            all_chunks.extend(new_chunks)
            total_truncated += n_trunc

    
    for idx, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = idx

   
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk) + "\n")

    total_tokens = sum(c["n_tokens"] for c in all_chunks)
    avg = total_tokens / len(all_chunks) if all_chunks else 0.0

    log.info(
        "  -> %d chunks  |  %d tokens  |  %.1f avg tokens/chunk  |  %d truncated",
        len(all_chunks), total_tokens, avg, total_truncated,
    )

    return {
        "sub_corpus":           sub_corpus,
        "n_chunks":             len(all_chunks),
        "n_tokens":             total_tokens,
        "avg_tokens_per_chunk": avg,
        "n_truncated_sents":    total_truncated,
    }



def print_summary(summaries: list[dict]) -> None:
    """Print a formatted table of per-sub-corpus chunk statistics."""
    header = f"{'Sub-corpus':<22} {'Chunks':>8} {'Tokens':>12} {'Avg tok/chunk':>14} {'Truncated':>10}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    total_chunks = total_tokens = 0
    for s in summaries:
        print(
            f"{s['sub_corpus']:<22} {s['n_chunks']:>8,} {s['n_tokens']:>12,} "
            f"{s['avg_tokens_per_chunk']:>14.1f} {s['n_truncated_sents']:>10}"
        )
        total_chunks += s["n_chunks"]
        total_tokens += s["n_tokens"]
    print(sep)
    print(f"{'TOTAL':<22} {total_chunks:>8,} {total_tokens:>12,}")
    print(sep)
    print()



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chunk BabyLM sub-corpora into EOS-delimited GPT-2 token sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--corpus_scale",
        choices=["10M", "100M"],
        default="10M",
        help="Which corpus scale to process.",
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Override the default input directory (data/raw/babylm_<scale>/).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Override the default output directory (data/chunked_<scale>/).",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=MAX_TOKENS,
        help="Maximum tokens per chunk (should match model context window).",
    )
    p.add_argument(
        "--dialogue_corpora",
        nargs="+",
        default=["switchboard"],
        metavar="NAME",
        help=(
            "Sub-corpus names that use conversational turn boundaries instead of "
            "NLTK sentence boundaries for EOS insertion.  Matched against the "
            "filename stem (without .train.txt).  Default: switchboard.  "
            "Rationale: Punkt is calibrated for written text and produces "
            "spurious boundaries in disfluent telephone speech; the natural "
            "completion unit in dialogue is the speaker turn."
        ),
    )
    p.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run on the first 1000 lines of each file only (quick sanity check).",
    )
    return p.parse_args()


def resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return (input_dir, output_dir) based on corpus_scale and any overrides."""
    base = Path(__file__).parent.parent  
    if args.input_dir:
        in_dir = args.input_dir
    else:
        in_dir = base / "data" / "raw" / f"babylm_{args.corpus_scale}"
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = base / "data" / f"chunked_{args.corpus_scale}"
    return in_dir, out_dir



def main() -> None:
    args = parse_args()
    in_dir, out_dir = resolve_dirs(args)

    dialogue_set = {name.lower() for name in args.dialogue_corpora}

    log.info("corpus_scale     : %s", args.corpus_scale)
    log.info("input_dir        : %s", in_dir)
    log.info("output_dir       : %s", out_dir)
    log.info("max_tokens       : %d", args.max_tokens)
    log.info("dialogue_corpora : %s  (turn-boundary EOS mode)", sorted(dialogue_set))
    if args.smoke_test:
        log.info("SMOKE TEST MODE - processing first 1000 lines per file")

  
    if not in_dir.exists():
        log.error("Input directory not found: %s", in_dir)
        sys.exit(1)

    corpus_files = sorted(in_dir.glob("*.train.txt"))
    if not corpus_files:
        log.error("No *.train.txt files found in %s", in_dir)
        sys.exit(1)
    log.info("Found %d sub-corpus files: %s",
             len(corpus_files), [f.name for f in corpus_files])

   
    ensure_punkt()


    log.info("Loading GPT2TokenizerFast …")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    assert tokenizer.eos_token_id == EOS_ID, (
        f"Unexpected EOS id: {tokenizer.eos_token_id} (expected {EOS_ID})"
    )
    log.info("Tokeniser loaded  (vocab_size=%d, eos_token_id=%d)",
             tokenizer.vocab_size, tokenizer.eos_token_id)


    summaries: list[dict] = []
    for fpath in corpus_files:
        if args.smoke_test:
    
            import tempfile, itertools
            with fpath.open("r", encoding="utf-8", errors="replace") as src:
                head = list(itertools.islice(
                    (l for l in src if l.strip()), 1000
                ))
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.writelines(l + "\n" for l in head)
                tmp_path = Path(tmp.name)
            sub_name = fpath.stem.replace(".train", "").lower()
            use_turns = sub_name in dialogue_set
            summary = process_file(tmp_path, out_dir, tokenizer, args.max_tokens, use_turns)
            summary["sub_corpus"] = sub_name + "_smoke"
            tmp_path.unlink()
        else:
            sub_name = fpath.stem.replace(".train", "").lower()
            use_turns = sub_name in dialogue_set
            summary = process_file(fpath, out_dir, tokenizer, args.max_tokens, use_turns)
        summaries.append(summary)

    print_summary(summaries)
    log.info("All chunks written to %s", out_dir)


if __name__ == "__main__":
    main()
