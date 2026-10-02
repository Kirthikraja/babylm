"""
src/diagnose_training_stop.py

Reports why each training condition stopped and what to do next.
Run this on ALICE where models/ and data/ directories are accessible.

Usage:
    python src/diagnose_training_stop.py
    python src/diagnose_training_stop.py --conditions flat chunked
    python src/diagnose_training_stop.py --configured_epochs 5 --corpus_scale 100M
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO, stream=sys.stdout,
)
log = logging.getLogger(__name__)

ALL_CONDITIONS = ["chunked", "flat", "balanced"]


def count_data_tokens(data_dir: Path) -> tuple[int, int]:
    """Returns (total_tokens, total_chunks) by scanning all .jsonl files."""
    if not data_dir.exists():
        return 0, 0
    total_tokens = 0
    total_chunks = 0
    for f in sorted(data_dir.glob("*.jsonl")):
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    total_tokens += obj.get("n_tokens", len(obj.get("token_ids", [])))
                    total_chunks += 1
                except json.JSONDecodeError:
                    continue
    return total_tokens, total_chunks


def find_latest_checkpoint_state(models_dir: Path):
    """Return (state_dict, checkpoint_path) for the highest-step resumable checkpoint."""
    if not models_dir.exists():
        return None, None
    ckpts = sorted(
        [d for d in models_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")
         and (d / "training_state.pt").exists()],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not ckpts:
        return None, None
    latest = ckpts[-1]
    state = torch.load(latest / "training_state.pt", map_location="cpu", weights_only=False)
    return state, latest


def count_all_checkpoints(models_dir: Path) -> int:
    if not models_dir.exists():
        return 0
    return len([d for d in models_dir.iterdir()
                if d.is_dir() and d.name.startswith("checkpoint-")])


def diagnose_condition(
    condition: str,
    base: Path,
    corpus_scale: str,
    configured_epochs: int,
) -> dict:
    models_dir  = base / "models"  / condition
    results_dir = base / "results" / condition
    data_dir    = base / "data"    / f"{condition}_{corpus_scale}"

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  CONDITION: {condition.upper()}")
    print(f"{sep}")

    # ── Training curves ───────────────────────────────────────────────────────
    curves_path = results_dir / "training_curves.json"
    if curves_path.exists():
        curves = json.loads(curves_path.read_text(encoding="utf-8"))
        print(f"  training_curves.json: found")
    else:
        print(f"  training_curves.json: NOT FOUND at {curves_path}")
        curves = {}

    early_stopped_at   = curves.get("early_stopped_at")
    epoch_nums         = curves.get("epoch_num", [])
    epochs_completed   = len(epoch_nums)
    eval_losses        = curves.get("epoch_eval_loss", [])
    train_losses       = curves.get("epoch_train_loss", [])

    # ── Latest checkpoint state ───────────────────────────────────────────────
    state, ckpt_path = find_latest_checkpoint_state(models_dir)
    n_ckpts = count_all_checkpoints(models_dir)

    if state is None:
        print(f"  Checkpoint state: NOT FOUND in {models_dir}")
        words_seen    = 0.0
        last_epoch    = 0
        last_batches  = 0
        global_step   = 0
    else:
        words_seen    = float(state.get("words_seen", 0))
        last_epoch    = int(state.get("epoch", 0))
        last_batches  = int(state.get("batches_in_epoch", 0))
        global_step   = int(state.get("global_step", 0))
        print(f"  Latest checkpoint : {ckpt_path.name}  (global_step={global_step})")
        print(f"  Total checkpoints : {n_ckpts}")

    # ── Available data ────────────────────────────────────────────────────────
    print(f"\n  Data directory: {data_dir}")
    total_tokens, total_chunks = count_data_tokens(data_dir)

    if total_tokens == 0:
        print(f"    WARNING: directory not found or empty")
        words_per_epoch = float("inf")
        n_train_chunks = 0
    else:
        n_eval_chunks  = max(1, int(total_chunks * 0.1))
        n_train_chunks = total_chunks - n_eval_chunks
        words_per_epoch = total_tokens * 0.75 * (n_train_chunks / total_chunks)
        print(f"    Chunks total    : {total_chunks:,}")
        print(f"    Tokens total    : {total_tokens:,}  ({total_tokens/1e6:.2f}M)")
        print(f"    Chunks in train : {n_train_chunks:,}  (90% split)")
        print(f"    Words/epoch ≈   : {words_per_epoch:,.0f}  ({words_per_epoch/1e6:.3f}M)")

    # ── Training progress ─────────────────────────────────────────────────────
    print(f"\n  Training progress:")
    print(f"    Words seen           : {words_seen:,.0f}  ({words_seen/1e6:.3f}M)")
    print(f"    global_step          : {global_step:,}")
    print(f"    Epochs completed     : {epochs_completed} of {configured_epochs} configured")
    print(f"    Last checkpoint epoch: {last_epoch} (0-indexed)")
    print(f"    Batches at last ckpt : {last_batches:,}")
    print(f"    early_stopped_at     : {early_stopped_at}")

    if words_per_epoch > 0 and words_per_epoch != float("inf"):
        epochs_equiv = words_seen / words_per_epoch
        print(f"    Effective exposure   : {epochs_equiv:.3f} epochs-worth of data")
    else:
        epochs_equiv = 0.0

    if eval_losses:
        best_loss = min(eval_losses)
        best_ep   = eval_losses.index(best_loss) + 1
        print(f"    Best epoch val loss  : {best_loss:.4f}  (epoch {best_ep})")
        print(f"    Final epoch val loss : {eval_losses[-1]:.4f}")
        if len(eval_losses) >= 2:
            delta = eval_losses[-1] - eval_losses[-2]
            print(f"    Val loss Δ last epoch: {delta:+.4f}")

    # ── Diagnosis ─────────────────────────────────────────────────────────────
    print(f"\n  DIAGNOSIS:")

    if early_stopped_at is not None:
        diagnosis = "early_stopping"
        print(f"    ✓ STOPPED VIA EARLY STOPPING at epoch {early_stopped_at}")
        print(f"      Validation loss stopped improving. This is a genuine convergence signal.")
        print(f"      Words seen ({words_seen/1e6:.2f}M) does NOT reflect a hardware kill.")
        recommendation = (
            "To continue past this for matched-exposure comparison, add "
            "--early_stop_patience 999 to the job script and resubmit."
        )

    elif epochs_completed >= configured_epochs:
        diagnosis = "completed_epochs"
        print(f"    ✓ COMPLETED ALL {configured_epochs} CONFIGURED EPOCHS normally")
        print(f"      Training ran {epochs_equiv:.2f}× worth of unique data.")
        print(f"      Further data would require repeating epochs (methodologically fine,")
        print(f"      but should be noted in the thesis as data replay).")
        recommendation = (
            "No further training needed unless the target exposure exceeds "
            f"{epochs_completed * words_per_epoch/1e6:.1f}M words. "
            "If you need more, increase --epochs and resubmit."
        )

    else:
        diagnosis = "walltime"
        epochs_partial = epochs_equiv - int(epochs_equiv)
        print(f"    ✗ STOPPED DUE TO SLURM WALLTIME")
        print(f"      Completed {epochs_completed} full epochs; killed ~{epochs_partial*100:.0f}% through epoch {epochs_completed+1}.")
        print(f"      This is NOT convergence — just a 4-hour job limit.")
        recommendation = (
            "Simply resubmit the same sbatch command. The resume logic picks up "
            "exactly where the job died. No flag changes needed."
        )

    print(f"\n  RECOMMENDATION: {recommendation}")
    print()

    return {
        "condition":         condition,
        "diagnosis":         diagnosis,
        "words_seen":        words_seen,
        "words_per_epoch":   words_per_epoch,
        "epochs_completed":  epochs_completed,
        "configured_epochs": configured_epochs,
        "epochs_equiv":      epochs_equiv,
        "early_stopped_at":  early_stopped_at,
        "total_tokens":      total_tokens,
        "total_chunks":      total_chunks,
        "n_train_chunks":    n_train_chunks,
        "global_step":       global_step,
    }


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--conditions",        nargs="+", default=ALL_CONDITIONS,
                   choices=ALL_CONDITIONS)
    p.add_argument("--corpus_scale",      default="100M", choices=["10M", "100M"])
    p.add_argument("--configured_epochs", type=int, default=5,
                   help="Epochs the job was configured to run (check job_train_*.sh)")
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(__file__).parent.parent

    print("=" * 64)
    print("  BabyLM Training Stop Diagnosis")
    print(f"  Configured epochs : {args.configured_epochs}")
    print(f"  Corpus scale      : {args.corpus_scale}")
    print("=" * 64)

    results = []
    for cond in args.conditions:
        r = diagnose_condition(cond, base, args.corpus_scale, args.configured_epochs)
        results.append(r)

    # ── Summary table ─────────────────────────────────────────────────────────
    sep = "=" * 64
    print(f"\n{sep}")
    print("  SUMMARY TABLE")
    print(sep)
    header = f"  {'Condition':<12}  {'Words Seen':>12}  {'Epochs':>8}  {'Equiv':>6}  Diagnosis"
    print(header)
    print(f"  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*6}  {'-'*20}")
    for r in results:
        ws  = f"{r['words_seen']/1e6:.2f}M"
        ep  = f"{r['epochs_completed']}/{r['configured_epochs']}"
        eq  = f"{r['epochs_equiv']:.2f}"
        print(f"  {r['condition']:<12}  {ws:>12}  {ep:>8}  {eq:>6}  {r['diagnosis']}")

    # ── Realistic ceiling per condition ───────────────────────────────────────
    print(f"\n{sep}")
    print("  REALISTIC TRAINING CEILING PER CONDITION")
    print(f"  (maximum words each could reach if trained to {args.configured_epochs} epochs)")
    print(sep)
    ceilings = {}
    for r in results:
        if r["words_per_epoch"] == float("inf") or r["words_per_epoch"] == 0:
            ceiling = float("inf")
            print(f"  {r['condition']:<12}  DATA NOT FOUND — cannot compute ceiling")
        else:
            ceiling = r["words_per_epoch"] * args.configured_epochs
            ceilings[r["condition"]] = ceiling
            print(f"  {r['condition']:<12}  {ceiling/1e6:.1f}M words  "
                  f"({r['words_per_epoch']/1e6:.2f}M/epoch × {args.configured_epochs} epochs)")

    if ceilings:
        lowest_cond = min(ceilings, key=ceilings.get)
        lowest_val  = ceilings[lowest_cond]
        print(f"\n  → Lowest ceiling: {lowest_cond} at {lowest_val/1e6:.1f}M words")
        print(f"    This is the SAFE TARGET for matched-exposure comparison.")
        print(f"    Any condition that has already exceeded {lowest_val/1e6:.1f}M is fine as-is.")
        print(f"    Conditions below {lowest_val/1e6:.1f}M need to train more.")

    print()

    # ── Action list ───────────────────────────────────────────────────────────
    walltime = [r["condition"] for r in results if r["diagnosis"] == "walltime"]
    es       = [r["condition"] for r in results if r["diagnosis"] == "early_stopping"]
    done     = [r["condition"] for r in results if r["diagnosis"] == "completed_epochs"]

    print(f"{sep}")
    print("  ACTION LIST")
    print(sep)
    if walltime:
        print(f"  RESUBMIT (walltime kill, can simply resume):")
        for c in walltime:
            print(f"    sbatch jobs/job_train_{c}.sh")
    if es:
        print(f"  RESUME WITH EARLY STOPPING DISABLED (stopped too soon for fair comparison):")
        for c in es:
            print(f"    # Add --early_stop_patience 999 to jobs/job_train_{c}.sh, then:")
            print(f"    sbatch jobs/job_train_{c}.sh")
    if done:
        print(f"  NO ACTION NEEDED (completed configured epochs):")
        for c in done:
            print(f"    {c}: already at {ceilings.get(c, 0)/1e6:.1f}M words maximum")
    print()


if __name__ == "__main__":
    main()
