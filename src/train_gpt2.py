"""
src/train_gpt2.py

Trains GPT-2 Small from scratch on a tokenized BabyLM dataset.

Conditions:
  --condition chunked   : uses data/chunked_10M/*.jsonl
  --condition flat      : uses data/flat_10M/*.jsonl
  --condition balanced  : uses data/balanced_10M/*.jsonl

Checkpointing: saves model + optimizer + scheduler + training state every 1M words.
Resume: automatically detects the latest checkpoint and continues from there.
        Just resubmit the same sbatch command — no extra flags needed.

Usage:
    python src/train_gpt2.py --condition flat --corpus_scale 100M
    python src/train_gpt2.py --condition chunked --corpus_scale 100M
"""

from __future__ import annotations
import argparse
import hashlib
import itertools
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
EVAL_SPLIT = 0.1
WORDS_PER_CHECKPOINT = 1_000_000

# ── Early stopping ──────────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE  = 3
EARLY_STOP_MIN_DELTA = 1e-4


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
    labels    = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L]    = b["labels"]
    attention_mask = (input_ids != pad_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ── Eval ──────────────────────────────────────────────────────────────────────

def compute_eval_loss(model, eval_loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            total += model(**batch).loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


# ── Data hash ─────────────────────────────────────────────────────────────────

def compute_data_hash(data_dir: Path) -> str:
    """MD5 of sorted JSONL filenames + line counts — detects data dir changes."""
    files = sorted(data_dir.glob("*.jsonl"))
    parts = []
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            n = sum(1 for _ in fh)
        parts.append(f"{f.name}:{n}")
    return hashlib.md5("\n".join(parts).encode()).hexdigest()


# ── Resume helpers ────────────────────────────────────────────────────────────

def find_latest_checkpoint(out_dir: Path) -> tuple[Path | None, dict | None]:
    """Return (checkpoint_path, state_dict) for the latest resumable checkpoint."""
    if not out_dir.exists():
        return None, None
    ckpts = sorted(
        [d for d in out_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")
         and (d / "training_state.pt").exists()],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not ckpts:
        return None, None
    latest = ckpts[-1]
    state = torch.load(latest / "training_state.pt", map_location="cpu",
                       weights_only=False)
    return latest, state


def save_checkpoint(
    out_dir: Path,
    global_step: int,
    model,
    optimizer,
    scheduler,
    epoch: int,
    batches_in_epoch: int,
    words_seen: float,
    best_eval_loss: float,
    patience_counter: int,
    curves: dict,
    data_hash: str,
) -> None:
    ckpt_path = out_dir / f"checkpoint-{global_step}"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_path)
    torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt_path / "scheduler.pt")
    torch.save({
        "global_step":       global_step,
        "epoch":             epoch,
        "batches_in_epoch":  batches_in_epoch,
        "words_seen":        words_seen,
        "best_eval_loss":    best_eval_loss,
        "patience_counter":  patience_counter,
        "curves":            curves,
        "data_hash":         data_hash,
    }, ckpt_path / "training_state.pt")
    log.info("  Checkpoint saved → %s", ckpt_path)


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    base        = Path(__file__).parent.parent
    data_dir    = args.data_dir or base / "data" / f"{args.condition}_{args.corpus_scale}"
    out_dir     = args.output_dir or base / "models" / args.condition
    results_dir = base / "results" / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        log.error("No .jsonl files in %s", data_dir)
        sys.exit(1)
    log.info("Data dir: %s  (%d files)", data_dir, len(jsonl_files))
    data_hash = compute_data_hash(data_dir)
    log.info("Data hash: %s", data_hash)

    # ── Dataset split (fixed seed so resume sees same split) ──────────────────
    full_dataset = ChunkDataset(jsonl_files)
    n_eval  = max(1, int(len(full_dataset) * EVAL_SPLIT))
    n_train = len(full_dataset) - n_eval
    train_dataset, eval_dataset = random_split(
        full_dataset, [n_train, n_eval],
        generator=torch.Generator().manual_seed(42),
    )
    log.info("Train: %d chunks  |  Eval: %d chunks", n_train, n_eval)

    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=lambda b: collate_fn(b),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Step / checkpoint math (needed before resume load) ────────────────────
    tokens_per_step    = args.batch_size * args.grad_accum * 1024
    words_per_step     = tokens_per_step * 0.75
    steps_per_epoch    = math.ceil(n_train / (args.batch_size * args.grad_accum))
    total_steps        = steps_per_epoch * args.epochs
    warmup_steps       = max(1, int(total_steps * WARMUP_RATIO))
    steps_per_ckpt     = max(1, int(WORDS_PER_CHECKPOINT / words_per_step))
    log.info("Total steps: %d  |  Steps/epoch: %d  |  Checkpoint every: %d steps",
             total_steps, steps_per_epoch, steps_per_ckpt)

    # ── Resume detection ──────────────────────────────────────────────────────
    resume_ckpt, resume_state = find_latest_checkpoint(out_dir)

    if resume_ckpt is not None:
        log.info("=" * 60)
        log.info("RESUMING from %s", resume_ckpt)
        log.info("=" * 60)
        stored_hash = resume_state.get("data_hash")
        if stored_hash is None:
            log.warning(
                "Checkpoint predates data-hash safety check — cannot verify data "
                "integrity. Proceeding, but consider retraining if source .jsonl "
                "files may have changed since this checkpoint was saved."
            )
        elif stored_hash != data_hash:
            log.error("=" * 60)
            log.error("DATA MISMATCH DETECTED — REFUSING TO RESUME")
            log.error("  Checkpoint trained on data_hash : %s", stored_hash)
            log.error("  Current data directory hash     : %s", data_hash)
            log.error("  The source .jsonl files in %s appear to have changed.", data_dir)
            log.error("  Resuming would silently corrupt training. Either restore the")
            log.error("  original data or delete this checkpoint and retrain from scratch.")
            log.error("=" * 60)
            sys.exit(1)
        else:
            log.info("Data hash verified — source data unchanged since checkpoint saved.")
        model = GPT2LMHeadModel.from_pretrained(str(resume_ckpt)).to(device)
        global_step      = resume_state["global_step"]
        words_seen       = resume_state["words_seen"]
        best_eval_loss   = resume_state["best_eval_loss"]
        patience_counter = resume_state["patience_counter"]
        curves           = resume_state["curves"]
        resume_epoch     = resume_state["epoch"]
        resume_batches   = resume_state["batches_in_epoch"]
        log.info("Resuming at global_step=%d  epoch=%d  batches_in_epoch=%d  words=%.1fM",
                 global_step, resume_epoch + 1, resume_batches, words_seen / 1_000_000)
    else:
        log.info("No resumable checkpoint found — starting fresh")
        model = GPT2LMHeadModel(GPT2Config(**GPT2_SMALL)).to(device)
        log.info("GPT-2 Small: %d parameters", sum(p.numel() for p in model.parameters()))
        global_step      = 0
        words_seen       = 0.0
        best_eval_loss   = float("inf")
        patience_counter = 0
        curves           = {
            "condition":        args.condition,
            "train_loss":       [], "eval_loss":        [],
            "step":             [], "words_seen":        [],
            "epoch_num":        [], "epoch_train_loss":  [],
            "epoch_eval_loss":  [], "early_stopped_at":  None,
        }
        resume_epoch   = -1
        resume_batches = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if resume_ckpt is not None:
        optimizer.load_state_dict(
            torch.load(resume_ckpt / "optimizer.pt", map_location=device,
                       weights_only=False))
        scheduler.load_state_dict(
            torch.load(resume_ckpt / "scheduler.pt", map_location="cpu",
                       weights_only=False))

    curves_path = results_dir / "training_curves.json"
    optimizer.zero_grad()

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(args.epochs):

        if epoch < resume_epoch:
            log.info("Skipping epoch %d (already completed)", epoch + 1)
            continue

        # Per-epoch shuffle seed → same order whether fresh or resumed
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(42 + epoch),
            num_workers=4, pin_memory=True, collate_fn=lambda b: collate_fn(b),
        )

        # Skip batches already processed in the resume epoch
        skip_batches    = resume_batches if epoch == resume_epoch else 0
        batches_in_epoch = skip_batches

        if skip_batches > 0:
            log.info("Epoch %d: skipping first %d batches (already done in previous run)",
                     epoch + 1, skip_batches)

        model.train()
        accum_loss = 0.0
        epoch_step_losses: list[float] = []

        loader = itertools.islice(train_loader, skip_batches, None)

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            loss  = out.loss / args.grad_accum
            loss.backward()
            accum_loss   += loss.item()
            batches_in_epoch += 1
            words_seen   += args.batch_size * 1024 * 0.75

            if batches_in_epoch % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                train_loss = accum_loss * args.grad_accum / args.grad_accum
                accum_loss = 0.0
                epoch_step_losses.append(train_loss)

                log.info("epoch=%d  step=%d/%d  train_loss=%.4f  lr=%.2e  words=%.1fM",
                         epoch + 1, global_step, total_steps, train_loss,
                         scheduler.get_last_lr()[0], words_seen / 1_000_000)

                if global_step % steps_per_ckpt == 0 or global_step == total_steps:
                    eval_loss = compute_eval_loss(model, eval_loader, device)
                    log.info("  CHECKPOINT  step=%d  eval_loss=%.4f  ppl=%.2f",
                             global_step, eval_loss, math.exp(eval_loss))

                    curves["step"].append(global_step)
                    curves["words_seen"].append(round(words_seen))
                    curves["train_loss"].append(round(train_loss, 4))
                    curves["eval_loss"].append(round(eval_loss, 4))
                    curves_path.write_text(json.dumps(curves, indent=2), encoding="utf-8")

                    save_checkpoint(
                        out_dir, global_step, model, optimizer, scheduler,
                        epoch, batches_in_epoch, words_seen,
                        best_eval_loss, patience_counter, curves,
                        data_hash=data_hash,
                    )

        # ── End of epoch ──────────────────────────────────────────────────────
        epoch_train_loss = (sum(epoch_step_losses) / len(epoch_step_losses)
                            if epoch_step_losses else float("inf"))
        epoch_eval_loss  = compute_eval_loss(model, eval_loader, device)

        curves["epoch_num"].append(epoch + 1)
        curves["epoch_train_loss"].append(round(epoch_train_loss, 4))
        curves["epoch_eval_loss"].append(round(epoch_eval_loss, 4))
        curves_path.write_text(json.dumps(curves, indent=2), encoding="utf-8")

        log.info("EPOCH %d/%d  train=%.4f  val=%.4f  ppl=%.2f",
                 epoch + 1, args.epochs, epoch_train_loss, epoch_eval_loss,
                 math.exp(epoch_eval_loss))

        if epoch_eval_loss < best_eval_loss - EARLY_STOP_MIN_DELTA:
            best_eval_loss   = epoch_eval_loss
            patience_counter = 0
            model.save_pretrained(out_dir / "best")
            log.info("  New best val loss %.4f → saved best/", best_eval_loss)
        else:
            patience_counter += 1
            log.info("  No improvement (%d/%d patience)", patience_counter, EARLY_STOP_PATIENCE)
            if patience_counter >= EARLY_STOP_PATIENCE:
                log.info("EARLY STOPPING at epoch %d", epoch + 1)
                curves["early_stopped_at"] = epoch + 1
                curves_path.write_text(json.dumps(curves, indent=2), encoding="utf-8")
                break

        # After completing the resume epoch, clear skip so next epochs run fully
        resume_batches = 0

    # ── Final model ───────────────────────────────────────────────────────────
    model.save_pretrained(out_dir / "final")
    tokenizer_path = base / "tokenizer"
    src = str(tokenizer_path) if tokenizer_path.exists() else "gpt2"
    GPT2TokenizerFast.from_pretrained(src).save_pretrained(out_dir / "final")
    log.info("Training complete. Final model → %s/final", out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--condition",    choices=["chunked", "flat", "balanced"], required=True)
    p.add_argument("--corpus_scale", choices=["10M", "100M"], default="10M")
    p.add_argument("--data_dir",     type=Path, default=None)
    p.add_argument("--output_dir",   type=Path, default=None)
    p.add_argument("--epochs",       type=int,  default=1)
    p.add_argument("--batch_size",   type=int,  default=BATCH_SIZE)
    p.add_argument("--grad_accum",   type=int,  default=GRAD_ACCUM)
    p.add_argument("--lr",           type=float, default=LR)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
