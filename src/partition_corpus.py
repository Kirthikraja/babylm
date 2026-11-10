"""
src/partition_corpus.py

Splits the BabyLM 100M corpus into 5 ablation conditions, applies EOS-chunking,
and equalises each condition to the same total-token budget via DENSITY sampling
(Sachdeva et al., ICLR 2026 — k-means over chunk embeddings, keep one centroid-
nearest chunk per cluster).

Corpus structure (data/raw/babylm_100M/corpus.train.txt):
  · lines 0 – first_childes_marker  : BNC Spoken + Switchboard (unmarked)
  · = = = childes/... = = =         : CHILDES sections
  · = = = PGxxxx = = =              : Gutenberg sections
  · (unmarked text between PG blocks) : OpenSubtitles (interleaved)
  · = = = Article Title = = =        : Simple Wikipedia sections

Conditions:
  full_baseline   – all content
  no_fiction      – remove Gutenberg (PG-marked blocks)
  no_childes      – remove CHILDES (childes-marked blocks)
  no_dialogue     – remove pre_childes block + OpenSubtitles (unmarked in Gut section)
  no_wikipedia    – remove Wikipedia (article-title-marked blocks)

Usage:
    python src/partition_corpus.py                          # all conditions
    python src/partition_corpus.py --condition no_childes   # one condition
    python src/partition_corpus.py --no_density             # skip k-means step
    python src/partition_corpus.py --target_tokens 50000000
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Generator

import numpy as np

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
EOS = "<|endoftext|>"
EOS_ID = 50256
MAX_SEQ_LEN = 1024
SENT_END_RE = re.compile(r"(?<=[.!?])\s+")

CONDITIONS = ["full_baseline", "no_fiction", "no_childes", "no_dialogue", "no_wikipedia"]

MARKER_RE = re.compile(r"^= = = (.+?) = = =$")
CHILDES_RE = re.compile(r"^childes/")
GUTENBERG_RE = re.compile(r"^PG\d")


# ── Corpus section labelling ───────────────────────────────────────────────────

def label_corpus(corpus_path: Path) -> Generator[tuple[str, str], None, None]:
    """Yield (section_label, text_line) for every non-marker line.

    Section labels:
        pre_childes   – BNC Spoken + Switchboard at start of file
        childes       – CHILDES blocks
        gutenberg     – PG-marked Gutenberg blocks
        opensubs      – unmarked text in the Gut+OpenSubs region
        wikipedia     – article-title-marked Simple Wikipedia blocks
    """
    current = "pre_childes"
    childes_seen = False
    gut_region = False   # True once we have entered the Gutenberg/OpenSubs zone

    with corpus_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = MARKER_RE.match(line.rstrip())
            if m:
                tag = m.group(1)
                if CHILDES_RE.match(tag):
                    current = "childes"
                    childes_seen = True
                    gut_region = False
                elif GUTENBERG_RE.match(tag):
                    current = "gutenberg"
                    gut_region = True
                else:
                    # Wikipedia article-title marker
                    current = "wikipedia"
                    gut_region = False
                continue

            # Determine actual label for unmarked text
            if current == "gutenberg" and gut_region:
                # Text between PG books inside the Gut+OpenSubs region = OpenSubtitles
                # (We yield 'gutenberg' only while directly under a PG marker.)
                # We track this with a secondary flag set in the marker handler.
                pass
            yield current, line


def _find_last_pg_line(corpus_path: Path) -> int:
    """Single-pass scan to find the line number of the last PG marker."""
    last = -1
    with corpus_path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if MARKER_RE.match(line.rstrip()) and GUTENBERG_RE.match(
                MARKER_RE.match(line.rstrip()).group(1)
            ):
                last = i
    return last


def label_corpus_v2(corpus_path: Path) -> Generator[tuple[str, str], None, None]:
    """
    Labels each non-marker line with its sub-corpus.

    Labels:
        pre_childes  – BNC Spoken + Switchboard at start of file
        childes      – CHILDES blocks (marker-bounded)
        gutenberg    – Gutenberg books (PG-marked, up to last PG marker)
        opensubs     – OpenSubtitles: text after the last PG marker, before Wikipedia
        wikipedia    – Simple Wikipedia (article-title-marked blocks)

    Requires a preliminary scan to locate the last PG marker line.
    """
    log.info("Pre-scan: finding last Gutenberg marker ...")
    last_pg_lineno = _find_last_pg_line(corpus_path)
    log.info("Last PG marker at line %d", last_pg_lineno)

    current = "pre_childes"
    gut_region = False   # True once first PG marker seen

    with corpus_path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh):
            m = MARKER_RE.match(line.rstrip())
            if m:
                tag = m.group(1)
                if CHILDES_RE.match(tag):
                    current = "childes"
                    gut_region = False
                elif GUTENBERG_RE.match(tag):
                    current = "gutenberg"
                    gut_region = True
                else:
                    current = "wikipedia"
                    gut_region = False
                continue

            # Determine actual label: text after last PG marker but before Wikipedia
            # is OpenSubtitles (lineno > last_pg_lineno and still in gut_region)
            if gut_region and lineno > last_pg_lineno:
                label = "opensubs"
            else:
                label = current
            yield label, line


# ── Sentence-boundary EOS insertion ───────────────────────────────────────────

def eos_chunk_text(text: str) -> str:
    """Insert EOS token after each sentence boundary."""
    return SENT_END_RE.sub(f" {EOS} ", text)


# ── Tokenization and packing ───────────────────────────────────────────────────

def build_jsonl_chunks(
    text_blocks: list[str],
    tokenizer,
    max_len: int = MAX_SEQ_LEN,
) -> list[dict]:
    """
    EOS-chunk each block, tokenize, and pack into max_len windows.
    Returns list of {"input_ids": [...]} dicts.
    """
    all_ids: list[int] = []
    for block in text_blocks:
        chunked = eos_chunk_text(block)
        ids = tokenizer.encode(chunked, add_special_tokens=False)
        all_ids.extend(ids)
        all_ids.append(EOS_ID)

    chunks: list[dict] = []
    for start in range(0, len(all_ids) - max_len + 1, max_len):
        window = all_ids[start : start + max_len]
        chunks.append({"token_ids": window})
    return chunks


# ── DENSITY equalization (k-means over last-hidden-state embeddings) ───────────

def density_subsample(
    chunks: list[dict],
    target_n: int,
    tokenizer,
    model,
    device: str = "cpu",
    batch_size: int = 32,
) -> list[dict]:
    """
    Downsample `chunks` to `target_n` via k-means DENSITY sampling:
      1. Embed each chunk (mean of last-layer hidden states).
      2. Run k-means with k=target_n.
      3. Keep the chunk nearest each centroid.

    Falls back to random sampling if sklearn is unavailable or n > len(chunks).
    """
    if target_n >= len(chunks):
        log.info("DENSITY: target_n=%d >= available=%d — no sampling needed", target_n, len(chunks))
        return chunks

    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        log.warning("sklearn not available — using random subsampling instead")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(chunks), size=target_n, replace=False)
        return [chunks[i] for i in sorted(idx)]

    import torch

    log.info("DENSITY: embedding %d chunks ...", len(chunks))
    embeddings: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for b_start in range(0, len(chunks), batch_size):
            batch = chunks[b_start : b_start + batch_size]
            max_b_len = max(len(c["input_ids"]) for c in batch)
            input_ids = torch.zeros(len(batch), max_b_len, dtype=torch.long).to(device)
            attention_mask = torch.zeros_like(input_ids)
            for j, c in enumerate(batch):
                ids = c["token_ids"]
                input_ids[j, : len(ids)] = torch.tensor(ids)
                attention_mask[j, : len(ids)] = 1

            out = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # Mean of last-layer hidden states over non-padding tokens
            last = out.hidden_states[-1]  # (batch, seq, hidden)
            mask = attention_mask.unsqueeze(-1).float()
            emb = (last * mask).sum(1) / mask.sum(1)  # (batch, hidden)
            embeddings.extend(emb.float().cpu().numpy())

            if (b_start // batch_size) % 20 == 0:
                log.info("  embedded %d / %d", b_start + len(batch), len(chunks))

    X = np.stack(embeddings)  # (n_chunks, hidden)
    log.info("DENSITY: running MiniBatchKMeans k=%d ...", target_n)
    km = MiniBatchKMeans(n_clusters=target_n, random_state=42, batch_size=4096, n_init=3)
    km.fit(X)
    labels = km.labels_  # (n_chunks,)

    # For each cluster, pick the chunk closest to its centroid
    centroids = km.cluster_centers_  # (k, hidden)
    kept_idx: list[int] = []
    for k in range(target_n):
        members = np.where(labels == k)[0]
        if len(members) == 0:
            continue
        dists = np.linalg.norm(X[members] - centroids[k], axis=1)
        kept_idx.append(int(members[np.argmin(dists)]))

    log.info("DENSITY: kept %d chunks out of %d", len(kept_idx), len(chunks))
    kept_idx.sort()
    return [chunks[i] for i in kept_idx]


# ── Per-condition text extraction ──────────────────────────────────────────────

CONDITION_KEEP: dict[str, set[str]] = {
    "full_baseline": {"pre_childes", "childes", "gutenberg", "opensubs", "wikipedia"},
    "no_fiction":    {"pre_childes", "childes", "opensubs", "wikipedia"},
    "no_childes":    {"pre_childes", "gutenberg", "opensubs", "wikipedia"},
    "no_dialogue":   {"childes", "gutenberg", "wikipedia"},   # removes BNC/Swbd + OpenSubs
    "no_wikipedia":  {"pre_childes", "childes", "gutenberg", "opensubs"},
}


def extract_condition_text(
    corpus_path: Path,
    keep_labels: set[str],
) -> list[str]:
    """Return list of text blocks (each is a contiguous unmarked block) for a condition."""
    blocks: list[str] = []
    current_lines: list[str] = []
    current_ok: bool | None = None

    for label, line in label_corpus_v2(corpus_path):
        ok = label in keep_labels
        if ok != current_ok and current_lines:
            if current_ok:
                blocks.append("".join(current_lines))
            current_lines = []
        current_ok = ok
        if ok:
            current_lines.append(line)

    if current_ok and current_lines:
        blocks.append("".join(current_lines))

    return blocks


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--corpus", type=Path,
                   default=Path("data/raw/babylm_100M/corpus.train.txt"))
    p.add_argument("--out_dir", type=Path, default=Path("data/ablation_100M"))
    p.add_argument("--condition", choices=CONDITIONS + ["all"], default="all",
                   help="Which condition to build (default: all)")
    p.add_argument("--target_tokens", type=int, default=50_000_000,
                   help="Token budget after DENSITY equalization")
    p.add_argument("--no_density", action="store_true",
                   help="Skip DENSITY sampling (write all chunks)")
    p.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--density_batch", type=int, default=64,
                   help="Batch size for chunk embedding during DENSITY")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base = Path(__file__).parent.parent
    corpus_path = base / args.corpus if not args.corpus.is_absolute() else args.corpus
    if not corpus_path.exists():
        log.error("Corpus not found: %s", corpus_path)
        sys.exit(1)

    out_dir = base / args.out_dir if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer (use standard GPT-2 tokenizer)
    from transformers import GPT2TokenizerFast
    log.info("Loading GPT-2 tokenizer ...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"eos_token": EOS})

    conditions_to_run = CONDITIONS if args.condition == "all" else [args.condition]

    for cond in conditions_to_run:
        cond_dir = out_dir / cond
        cond_dir.mkdir(exist_ok=True)
        jsonl_path = cond_dir / "train.jsonl"

        if jsonl_path.exists():
            log.info("Skipping %s — already exists at %s", cond, jsonl_path)
            continue

        log.info("=" * 60)
        log.info("Building condition: %s", cond)
        log.info("  keep labels: %s", sorted(CONDITION_KEEP[cond]))

        # ── Step 1: extract text blocks ───────────────────────────────────────
        log.info("  Extracting text blocks from corpus ...")
        blocks = extract_condition_text(corpus_path, CONDITION_KEEP[cond])
        total_words = sum(len(b.split()) for b in blocks)
        log.info("  Extracted %d blocks, ~%.1fM words", len(blocks), total_words / 1e6)

        # ── Step 2: EOS-chunk and tokenize ────────────────────────────────────
        log.info("  EOS-chunking and tokenizing ...")
        chunks = build_jsonl_chunks(blocks, tokenizer, max_len=args.max_seq_len)
        total_toks = len(chunks) * args.max_seq_len
        log.info("  %d chunks  (~%.1fM tokens)", len(chunks), total_toks / 1e6)

        # ── Step 3: DENSITY equalization ──────────────────────────────────────
        if not args.no_density and len(chunks) * args.max_seq_len > args.target_tokens:
            target_n = args.target_tokens // args.max_seq_len
            log.info("  DENSITY: downsampling to %d chunks (~%.1fM tokens) ...",
                     target_n, args.target_tokens / 1e6)

            # Load a small embedding model (use the base GPT-2 — no training needed)
            import torch
            from transformers import GPT2Model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("  DENSITY device: %s", device)
            embed_model = GPT2Model.from_pretrained("gpt2").to(device)

            chunks = density_subsample(
                chunks, target_n, tokenizer, embed_model, device, args.density_batch
            )
            del embed_model
            if device == "cuda":
                torch.cuda.empty_cache()

        log.info("  Writing %d chunks to %s ...", len(chunks), jsonl_path)
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk) + "\n")

        # Save metadata
        meta = {
            "condition": cond,
            "keep_labels": sorted(CONDITION_KEEP[cond]),
            "n_chunks": len(chunks),
            "total_tokens": len(chunks) * args.max_seq_len,
            "max_seq_len": args.max_seq_len,
            "density_applied": not args.no_density,
            "target_tokens": args.target_tokens,
        }
        (cond_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.info("  Done: %s  (%d chunks, %.1fM tokens)",
                 cond, len(chunks), len(chunks) * args.max_seq_len / 1e6)

    log.info("All conditions written to %s", out_dir)


if __name__ == "__main__":
    main()
