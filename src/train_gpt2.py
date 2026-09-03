"""
src/train_gpt2.py

Trains GPT-2 Small from scratch on a tokenized BabyLM dataset.

Two conditions:
  --condition chunked   : uses data/chunked_10M/*.jsonl (EOS at sentence boundaries)
  --condition flat      : uses data/flat_10M/*.jsonl    (sliding-window, no EOS)

Checkpointing: saves 1 checkpoint per million words processed (10 for 10M corpus).
Eval loss is computed on a 10% held-out split at every checkpoint.
Training curves saved to results/<condition>/training_curves.json.

Usage:
    python src/train_gpt2.py --condition chunked --corpus_scale 10M
    python src/train_gpt2.py --condition flat    --corpus_scale 10M
"""

from __future__ import annotations
import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    get_cosine_schedule_with_warmup,
)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── GPT-2 Small architecture ───────────────────────────────────────────────────
GPT2_SMALL = dict(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=3072,
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
)

# ── Training hyperparameters ────────────────────────────────────────────────────
LR = 6e-4
BATCH_SIZE = 4
GRAD_ACCUM = 32            # effective batch = 128
WARMUP_RATIO = 0.01
EVAL_SPLIT = 0.1           # 10% held-out for eval loss
WORDS_PER_CHECKPOINT = 1_000_000   # 1 checkpoint per 1M words


# ── Dataset ───────────────────────────────────────────────────────────────────

class ChunkDataset(Dataset):
    def __init__(self, jsonl_paths: list[Path], max_len: int = 1024):
        self.samples: list[list[int]] = []
        for path in jsonl_paths:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    obj = json.loads(line)
                    ids = obj["token_ids"]
                    if len(ids) > max_len:
                        ids = ids[:max_len]
                    self.samples.append(ids)
        log.info("Loaded %d chunks from %d files", len(self.samples), len(jsonl_paths))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ids = self.samples[idx]
        t = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": t, "labels": t.clone()}


def collate_fn(batch: list[dict], pad_id: int = 50256) -> dict:
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
    attention_mask = (input_ids != pad_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ── Eval loss ─────────────────────────────────────────────────────────────────

def compute_eval_loss(
    model: GPT2LMHeadModel,
    eval_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            total_loss += out.loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    base = Path(__file__).parent.parent
    data_dir = args.data_dir or base / "data" / f"{args.condition}_{args.corpus_scale}"
    out_dir = args.output_dir or base / "models" / args.condition
    results_dir = base / "results" / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        log.error("No .jsonl files in %s", data_dir)
        sys.exit(1)
    log.info("Data dir: %s  (%d files)", data_dir, len(jsonl_files))

    # ── Train / eval split ────────────────────────────────────────────────────
    full_dataset = ChunkDataset(jsonl_files)
    n_eval = max(1, int(len(full_dataset) * EVAL_SPLIT))
    n_train = len(full_dataset) - n_eval
    train_dataset, eval_dataset = random_split(
        full_dataset, [n_train, n_eval],
        generator=torch.Generator().manual_seed(42),
    )
    log.info("Train: %d chunks  |  Eval: %d chunks", n_train, n_eval)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=lambda b: collate_fn(b),
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=lambda b: collate_fn(b),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    config = GPT2Config(**GPT2_SMALL)
    model = GPT2LMHeadModel(config).to(device)
    log.info("GPT-2 Small: %d parameters", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    steps_per_epoch = math.ceil(n_train / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    # Checkpoint every 1M words — convert to steps
    # avg tokens per chunk ≈ 1024; avg words per token ≈ 0.75
    tokens_per_step = args.batch_size * args.grad_accum * 1024
    words_per_step = tokens_per_step * 0.75
    steps_per_checkpoint = max(1, int(WORDS_PER_CHECKPOINT / words_per_step))
    n_checkpoints = math.ceil(total_steps / steps_per_checkpoint)
    log.info("Total steps: %d  |  Checkpoint every %d steps  |  ~%d checkpoints",
             total_steps, steps_per_checkpoint, n_checkpoints)

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training curves storage ───────────────────────────────────────────────
    curves = {"condition": args.condition, "train_loss": [], "eval_loss": [], "step": [], "words_seen": []}
    global_step = 0
    words_seen = 0
    accum_loss = 0.0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()
            words_seen += args.batch_size * 1024 * 0.75  # approx words in this batch

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                train_loss = accum_loss * args.grad_accum / args.grad_accum
                accum_loss = 0.0

                log.info("epoch=%d  step=%d/%d  train_loss=%.4f  lr=%.2e  words=%.1fM",
                         epoch + 1, global_step, total_steps, train_loss,
                         scheduler.get_last_lr()[0], words_seen / 1_000_000)

                # ── Checkpoint + eval every N steps ──────────────────────────
                if global_step % steps_per_checkpoint == 0 or global_step == total_steps:
                    eval_loss = compute_eval_loss(model, eval_loader, device)
                    log.info("  CHECKPOINT  step=%d  eval_loss=%.4f  eval_ppl=%.2f",
                             global_step, eval_loss, math.exp(eval_loss))

                    ckpt_path = out_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt_path)

                    curves["step"].append(global_step)
                    curves["words_seen"].append(round(words_seen))
                    curves["train_loss"].append(round(train_loss, 4))
                    curves["eval_loss"].append(round(eval_loss, 4))

                    curves_path = results_dir / "training_curves.json"
                    curves_path.write_text(json.dumps(curves, indent=2), encoding="utf-8")
                    log.info("  Curves saved -> %s", curves_path)

    # Final model
    model.save_pretrained(out_dir / "final")
    tokenizer_path = base / "tokenizer"
    src = str(tokenizer_path) if tokenizer_path.exists() else "gpt2"
    GPT2TokenizerFast.from_pretrained(src).save_pretrained(out_dir / "final")
    log.info("Training complete. Final model -> %s/final", out_dir)
    log.info("Training curves -> %s/training_curves.json", results_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--condition", choices=["chunked", "flat"], required=True)
    p.add_argument("--corpus_scale", choices=["10M", "100M"], default="10M")
    p.add_argument("--data_dir", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=LR)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
