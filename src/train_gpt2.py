"""
src/train_gpt2.py

Trains GPT-2 Small from scratch on a tokenized BabyLM dataset.

Two conditions:
  --condition chunked   : uses data/chunked_100M/*.jsonl (EOS at sentence boundaries)
  --condition flat      : uses data/flat_100M/*.jsonl    (sliding-window, no EOS)

Saves 19 log-spaced checkpoints to models/<condition>/.

Usage:
    python src/train_gpt2.py --condition chunked --corpus_scale 100M
    python src/train_gpt2.py --condition flat    --corpus_scale 100M
"""

from __future__ import annotations
import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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
BATCH_SIZE = 32            # per-device
GRAD_ACCUM = 4             # effective batch = 128
WARMUP_RATIO = 0.01        # 1% of total steps
N_CHECKPOINTS = 19         # log-spaced


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


# ── Checkpoint schedule ────────────────────────────────────────────────────────

def log_spaced_steps(total_steps: int, n: int = N_CHECKPOINTS) -> list[int]:
    steps = np.logspace(0, np.log10(total_steps), n, dtype=int)
    return sorted(set(steps.tolist()))


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    base = Path(__file__).parent.parent
    data_dir = args.data_dir or base / "data" / f"{args.condition}_{args.corpus_scale}"
    out_dir = args.output_dir or base / "models" / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        log.error("No .jsonl files in %s", data_dir)
        sys.exit(1)
    log.info("Data dir: %s  (%d files)", data_dir, len(jsonl_files))

    dataset = ChunkDataset(jsonl_files)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    config = GPT2Config(**GPT2_SMALL)
    model = GPT2LMHeadModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("GPT-2 Small: %d parameters", n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    steps_per_epoch = math.ceil(len(dataset) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    log.info("Total steps: %d  |  Warmup: %d  |  Checkpoints: %d",
             total_steps, warmup_steps, N_CHECKPOINTS)

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    checkpoint_steps = set(log_spaced_steps(total_steps))

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        model.train()
        accum_loss = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 100 == 0:
                    log.info("epoch=%d  step=%d/%d  loss=%.4f  lr=%.2e",
                             epoch + 1, global_step, total_steps,
                             accum_loss * args.grad_accum / 100,
                             scheduler.get_last_lr()[0])
                    accum_loss = 0.0

                if global_step in checkpoint_steps:
                    ckpt_path = out_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt_path)
                    log.info("Saved checkpoint -> %s", ckpt_path)

    # Always save final model
    model.save_pretrained(out_dir / "final")
    GPT2TokenizerFast.from_pretrained("gpt2").save_pretrained(out_dir / "final")
    log.info("Training complete. Final model -> %s/final", out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--condition", choices=["chunked", "flat"], required=True)
    p.add_argument("--corpus_scale", choices=["10M", "100M"], default="100M")
    p.add_argument("--data_dir", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=LR)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
