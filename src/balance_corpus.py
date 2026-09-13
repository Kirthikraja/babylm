"""
src/balance_corpus.py

Diversity-based downsampling of chunked BabyLM sub-corpora using the
DENSITY-style sampler from Sachdeva et al. (2026, ICLR 2026).

For each sub-corpus:
  1. Load all chunks from data/chunked_<scale>/<subcorpus>.jsonl
  2. Embed each chunk with pretrained GPT-2 (mean last hidden state → 768-dim)
  3. Cluster with k-means (k = min(n_chunks, target_k))
  4. Keep the chunk closest to each cluster centroid (no duplicates)
  5. Write balanced JSONL to data/balanced_<scale>/<subcorpus>.jsonl

Rationale
---------
Sachdeva et al. show that naïve duplication at low sampling rates can hurt
rather than help model quality.  Their DENSITY sampler selects diverse,
spread-out examples; models trained on 10% of data this way can outperform
models trained on the full dataset.  We adopt the same principle here:
when balancing sub-corpora to a common size, we select semantically diverse
chunks rather than repeating any text.

Corpus decisions (from PPTX presentation 2026-09-13)
-----------------------------------------------------
- Switchboard (41 chunks) is dropped entirely — too small to be usable.
- Target k = 1 005 (BNC Spoken chunk count), applied to every sub-corpus.
- Sub-corpora already ≤ target_k keep ALL their chunks (no upsampling).

Usage
-----
    python src/balance_corpus.py                          # 10M, target_k=1005
    python src/balance_corpus.py --corpus_scale 100M --target_k 5000
    python src/balance_corpus.py --target_k 1005 --batch_size 4
    python src/balance_corpus.py --smoke_test             # 200 chunks per corpus
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm
from transformers import GPT2Model, GPT2TokenizerFast

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

DEFAULT_TARGET_K = 1_005          # BNC Spoken chunk count (Option B decision)
DEFAULT_DROP = {"switchboard"}    # Dropped: too small (41 chunks)
EMBED_DIM = 768                   # GPT-2 Small hidden size


# ── Embedding ──────────────────────────────────────────────────────────────────

def embed_chunks(
    token_id_lists: list[list[int]],
    model: GPT2Model,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Embed a list of token-id sequences with GPT-2.

    Each chunk is passed through GPT-2 as-is (already 1024 tokens).
    The last hidden state is averaged across all token positions to produce
    one 768-dim vector per chunk — the approach described in Sachdeva et al.

    Returns
    -------
    embeddings : np.ndarray, shape (n_chunks, 768)
    """
    model.eval()
    all_embeds: list[np.ndarray] = []

    for start in tqdm(range(0, len(token_id_lists), batch_size),
                      desc="  embedding", unit="batch", ncols=90, leave=False):
        batch_ids = token_id_lists[start : start + batch_size]

        # Pad to the longest sequence in this batch
        max_len = max(len(ids) for ids in batch_ids)
        padded = [ids + [0] * (max_len - len(ids)) for ids in batch_ids]
        attention_mask = [
            [1] * len(ids) + [0] * (max_len - len(ids)) for ids in batch_ids
        ]

        input_tensor = torch.tensor(padded, dtype=torch.long, device=device)
        mask_tensor  = torch.tensor(attention_mask, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_tensor, attention_mask=mask_tensor)
            hidden = outputs.last_hidden_state  # (B, T, 768)

        # Mean-pool over non-padding positions
        mask_f = mask_tensor.unsqueeze(-1).float()           # (B, T, 1)
        summed = (hidden * mask_f).sum(dim=1)                # (B, 768)
        lengths = mask_f.sum(dim=1).clamp(min=1)             # (B, 1)
        mean_embeds = (summed / lengths).cpu().float().numpy()
        all_embeds.append(mean_embeds)

    return np.vstack(all_embeds)


# ── Diversity sampling ─────────────────────────────────────────────────────────

def density_sample(
    chunks: list[dict],
    embeddings: np.ndarray,
    target_k: int,
    seed: int = 42,
) -> list[dict]:
    """
    Select *target_k* diverse chunks via k-means centroid proximity.

    If n_chunks <= target_k, returns all chunks unchanged (no upsampling).

    Algorithm
    ---------
    1. Run k-means with k = target_k on the chunk embeddings.
    2. For each cluster, find the chunk whose embedding is closest to the
       cluster centroid (Euclidean distance).
    3. Return those chunks — one per cluster, no duplicates.
    """
    n = len(chunks)
    if n <= target_k:
        log.info("  n_chunks=%d <= target_k=%d — keeping all", n, target_k)
        return chunks

    log.info("  Clustering %d chunks → %d via k-means …", n, target_k)

    # MiniBatchKMeans is substantially faster for large n (e.g. CHILDES ~6k)
    if n > 3_000:
        km = MiniBatchKMeans(
            n_clusters=target_k, random_state=seed, n_init=10, batch_size=1024
        )
    else:
        km = KMeans(n_clusters=target_k, random_state=seed, n_init=10)

    km.fit(embeddings)
    centroids = km.cluster_centers_        # (target_k, 768)
    labels    = km.labels_                 # (n,)

    selected_indices: list[int] = []
    for cluster_id in range(target_k):
        member_idx = np.where(labels == cluster_id)[0]
        if len(member_idx) == 0:
            continue
        member_embeds = embeddings[member_idx]                    # (m, 768)
        centroid      = centroids[cluster_id]                     # (768,)
        dists         = np.linalg.norm(member_embeds - centroid, axis=1)
        closest       = member_idx[np.argmin(dists)]
        selected_indices.append(int(closest))

    return [chunks[i] for i in sorted(selected_indices)]


# ── Per-corpus processing ──────────────────────────────────────────────────────

def process_corpus(
    jsonl_path: Path,
    out_path: Path,
    model: GPT2Model,
    device: torch.device,
    target_k: int,
    batch_size: int,
    seed: int,
) -> dict:
    sub_corpus = jsonl_path.stem

    log.info("Loading %s …", jsonl_path.name)
    chunks: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunks.append(json.loads(line))

    n_before = len(chunks)
    log.info("  %d chunks loaded", n_before)

    token_id_lists = [c["token_ids"] for c in chunks]
    embeddings = embed_chunks(token_id_lists, model, device, batch_size)

    selected = density_sample(chunks, embeddings, target_k, seed)
    n_after = len(selected)

    # Re-index chunk_id sequentially in the output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for new_id, chunk in enumerate(selected):
            chunk["chunk_id"] = new_id
            chunk["selected_by"] = "density"
            fh.write(json.dumps(chunk) + "\n")

    total_tokens = sum(c["n_tokens"] for c in selected)
    log.info(
        "  %s: %d → %d chunks  (%.1f%% kept)  |  %d tokens",
        sub_corpus, n_before, n_after,
        100.0 * n_after / n_before if n_before else 0,
        total_tokens,
    )
    return {
        "sub_corpus":    sub_corpus,
        "n_before":      n_before,
        "n_after":       n_after,
        "tokens_after":  total_tokens,
        "pct_kept":      round(100.0 * n_after / n_before, 1) if n_before else 0,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DENSITY-style diversity sampling for BabyLM sub-corpora.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--corpus_scale", choices=["10M", "100M"], default="10M")
    p.add_argument(
        "--input_dir", type=Path, default=None,
        help="Override input dir (default: data/chunked_<scale>/).",
    )
    p.add_argument(
        "--output_dir", type=Path, default=None,
        help="Override output dir (default: data/balanced_<scale>/).",
    )
    p.add_argument(
        "--target_k", type=int, default=DEFAULT_TARGET_K,
        help="Target chunks per sub-corpus (Sachdeva et al. DENSITY sampling).",
    )
    p.add_argument(
        "--drop", nargs="+", default=list(DEFAULT_DROP),
        metavar="NAME",
        help="Sub-corpus names to drop entirely (matched against JSONL stem).",
    )
    p.add_argument(
        "--model_name", default="gpt2",
        help="Pretrained HF model to use for chunk embeddings.",
    )
    p.add_argument(
        "--batch_size", type=int, default=8,
        help="Chunks per GPU/CPU forward pass during embedding.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for k-means.")
    p.add_argument(
        "--smoke_test", action="store_true",
        help="Cap each corpus at 200 chunks (quick sanity check).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base    = Path(__file__).parent.parent
    in_dir  = args.input_dir  or base / "data" / f"chunked_{args.corpus_scale}"
    out_dir = args.output_dir or base / "data" / f"balanced_{args.corpus_scale}"
    drop    = {d.lower() for d in args.drop}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device          : %s", device)
    log.info("Embedding model : %s", args.model_name)
    log.info("Target k        : %d", args.target_k)
    log.info("Drop            : %s", sorted(drop))
    log.info("Input dir       : %s", in_dir)
    log.info("Output dir      : %s", out_dir)

    log.info("Loading pretrained GPT-2 for embedding …")
    model = GPT2Model.from_pretrained(args.model_name).to(device)
    model.eval()

    jsonl_files = sorted(in_dir.glob("*.jsonl"))
    if not jsonl_files:
        log.error("No .jsonl files found in %s — run chunk_corpus.py first.", in_dir)
        sys.exit(1)

    summaries: list[dict] = []
    for jf in jsonl_files:
        sub_corpus = jf.stem.lower()
        if sub_corpus in drop:
            log.info("Dropping %s (in --drop list)", sub_corpus)
            continue

        out_path = out_dir / jf.name

        # Smoke-test: cap at 200 chunks
        if args.smoke_test:
            chunks = []
            with jf.open("r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= 200:
                        break
                    if line.strip():
                        chunks.append(json.loads(line))
            # write temp file so process_corpus can load it
            import tempfile, os
            tmp = Path(tempfile.mktemp(suffix=".jsonl"))
            with tmp.open("w") as fh:
                for c in chunks:
                    fh.write(json.dumps(c) + "\n")
            target = min(args.target_k, 20)
            summary = process_corpus(tmp, out_path, model, device, target, args.batch_size, args.seed)
            tmp.unlink(missing_ok=True)
        else:
            summary = process_corpus(jf, out_path, model, device, args.target_k, args.batch_size, args.seed)

        summaries.append(summary)

    # Summary table
    print()
    header = f"{'Sub-corpus':<22} {'Before':>8} {'After':>8} {'% kept':>8} {'Tokens':>12}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for s in summaries:
        print(f"{s['sub_corpus']:<22} {s['n_before']:>8,} {s['n_after']:>8,} "
              f"{s['pct_kept']:>7.1f}% {s['tokens_after']:>12,}")
    print(sep)
    total_chunks = sum(s["n_after"] for s in summaries)
    total_tokens = sum(s["tokens_after"] for s in summaries)
    print(f"{'TOTAL':<22} {'':>8} {total_chunks:>8,} {'':>8} {total_tokens:>12,}")
    print(sep)
    log.info("Balanced corpus written to %s", out_dir)


if __name__ == "__main__":
    main()
