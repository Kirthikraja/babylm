"""
src/extract_think_vectors.py

Extracts the "think vector" from each training checkpoint.

Definition (Marks & Tegmark 2023 difference-in-means):
    v_L = mean_hidden(Explicit contexts) - mean_hidden(Implicit contexts)
  where hidden = last-token residual-stream activation at transformer layer L.

Explicit contexts contain "thinks" as the knowledge cue; Implicit contain
"goes to get". The vector captures what the model encodes differently about
mental-state language vs action language.

This is the core representation to probe for the thesis claim:
  "GPT-2 BabyLMs develop an internal think vector that mediates FBT behaviour."

Outputs (in results/think_vectors/<condition>/):
  vectors_step_NNNNNN.npz    — think_vector, explicit_mean, implicit_mean
                               each shape (n_layers+1, hidden_dim)
  think_vector_summary.json  — per-checkpoint norms + cosine sims to final

Usage:
    python src/extract_think_vectors.py --condition flat
    python src/extract_think_vectors.py --condition flat --layer 8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO, stream=sys.stdout,
)
log = logging.getLogger(__name__)

MASK_TOKEN = "[MASK]"


# ── Activation extraction ─────────────────────────────────────────────────────

def get_last_token_hidden_states(
    model:     GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    context:   str,
    device:    torch.device,
) -> np.ndarray | None:
    """Last-token hidden state from every layer for one context string.

    Returns array of shape (n_layers+1, hidden_dim):
      index 0 = embedding output (before block 0)
      index L = output of transformer block L (L = 1 … n_layers)
    Returns None if the context tokenises to zero tokens.
    """
    ids = tokenizer.encode(context, add_special_tokens=False)
    if not ids:
        return None

    input_ids = torch.tensor([ids]).to(device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
    return np.stack(
        [hs[0, -1, :].float().cpu().numpy() for hs in outputs.hidden_states],
        axis=0,
    )  # (n_layers+1, hidden_dim)


def extract_think_vectors(
    model:     GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    df:        pd.DataFrame,
    device:    torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute think vector at all layers for one checkpoint.

    Returns:
        think_vector  (n_layers+1, hidden_dim)  — explicit_mean - implicit_mean
        explicit_mean (n_layers+1, hidden_dim)
        implicit_mean (n_layers+1, hidden_dim)
    """
    explicit_states: list[np.ndarray] = []
    implicit_states: list[np.ndarray] = []
    n_skipped = 0

    for _, row in df.iterrows():
        passage = str(row["passage"])
        cue     = str(row.get("knowledge_cue", "")).strip().lower()
        context = passage.replace(MASK_TOKEN, "").rstrip()

        states = get_last_token_hidden_states(model, tokenizer, context, device)
        if states is None:
            n_skipped += 1
            continue

        if cue == "explicit":
            explicit_states.append(states)
        elif cue == "implicit":
            implicit_states.append(states)
        # unknown cue value: skip silently

    if n_skipped:
        log.warning("Skipped %d items with empty tokenisation", n_skipped)
    if not explicit_states or not implicit_states:
        raise ValueError(
            f"Need both Explicit and Implicit items. Got "
            f"{len(explicit_states)} explicit, {len(implicit_states)} implicit. "
            "Check that fb.csv has a 'knowledge_cue' column."
        )

    exp_mean  = np.stack(explicit_states).mean(axis=0)   # (n_layers+1, hidden_dim)
    imp_mean  = np.stack(implicit_states).mean(axis=0)
    think_vec = exp_mean - imp_mean

    return think_vec, exp_mean, imp_mean


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--condition",  choices=["chunked", "flat", "balanced"], default="flat")
    p.add_argument("--data_path",  type=Path, default=None,
                   help="Path to fb.csv (default: data/fbt/fb.csv)")
    p.add_argument("--out_dir",    type=Path, default=None,
                   help="Where to save results (default: results/think_vectors/<condition>)")
    p.add_argument("--layer",      type=int,  default=None,
                   help="Layer to highlight in the printed summary (0=embed, 1-12=blocks). "
                        "All layers are always extracted.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base       = Path(__file__).parent.parent
    models_dir = base / "models" / args.condition
    out_dir    = args.out_dir or base / "results" / "think_vectors" / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── words_seen mapping ────────────────────────────────────────────────────
    curves_path   = base / "results" / args.condition / "training_curves.json"
    step_to_words: dict[int, int] = {}
    if curves_path.exists():
        curves = json.loads(curves_path.read_text(encoding="utf-8"))
        step_to_words = {
            int(s): int(w)
            for s, w in zip(curves.get("step", []), curves.get("words_seen", []))
        }
        log.info("Loaded %d step→words mappings", len(step_to_words))
    else:
        log.warning("training_curves.json not found — words_seen will be null")

    # ── Find checkpoints ──────────────────────────────────────────────────────
    checkpoints = sorted(
        [d for d in models_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not checkpoints:
        log.error("No checkpoint-* directories in %s", models_dir)
        sys.exit(1)
    log.info("Found %d checkpoints for condition=%s", len(checkpoints), args.condition)

    # ── FBT stimuli ───────────────────────────────────────────────────────────
    data_path = args.data_path or base / "data" / "fbt" / "fb.csv"
    if not data_path.exists():
        log.error("FBT stimuli not found at %s", data_path)
        log.error("Download: wget -P data/fbt/ https://raw.githubusercontent.com/seantrott/nlm-fb/main/data/stims/fb.csv")
        sys.exit(1)
    df = pd.read_csv(data_path)
    if "first_mention" in df.columns and "recent_mention" in df.columns:
        df = df[(df["first_mention"] == "Start") & (df["recent_mention"] == "End")]

    if "knowledge_cue" not in df.columns:
        log.error("fb.csv is missing the 'knowledge_cue' column — cannot split Explicit vs Implicit")
        sys.exit(1)

    n_exp = (df["knowledge_cue"].str.lower() == "explicit").sum()
    n_imp = (df["knowledge_cue"].str.lower() == "implicit").sum()
    log.info("FBT items: %d total  (%d explicit, %d implicit)", len(df), n_exp, n_imp)

    # ── Tokenizer (loaded once — same across checkpoints) ────────────────────
    first_ckpt_with_tok = next(
        (d for d in checkpoints
         if (d / "tokenizer.json").exists() or (d / "vocab.json").exists()),
        None,
    )
    tokenizer_path = str(first_ckpt_with_tok) if first_ckpt_with_tok else "gpt2"
    log.info("Loading tokenizer from: %s", tokenizer_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Resume support ────────────────────────────────────────────────────────
    summary_path    = out_dir / "think_vector_summary.json"
    done_steps:     set[int]   = set()
    summary_records: list[dict] = []
    if summary_path.exists():
        try:
            prev            = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_records = prev.get("checkpoints", [])
            done_steps      = {r["step"] for r in summary_records}
            log.info("Resuming: %d already done, %d remaining",
                     len(done_steps), len(checkpoints) - len(done_steps))
        except Exception as exc:
            log.warning("Could not load existing summary (%s) — starting fresh", exc)

    # ── Extract from each checkpoint ─────────────────────────────────────────
    for ckpt in checkpoints:
        step       = int(ckpt.name.split("-")[1])
        words_seen = step_to_words.get(step)

        if step in done_steps:
            log.info("Skipping checkpoint-%d (already done)", step)
            continue

        log.info("── checkpoint-%d  (%.2fM words) ──", step, (words_seen or 0) / 1e6)

        model = GPT2LMHeadModel.from_pretrained(str(ckpt)).to(device)
        model.eval()

        think_vec, exp_mean, imp_mean = extract_think_vectors(model, tokenizer, df, device)
        n_layers_plus_1 = think_vec.shape[0]

        npz_path = out_dir / f"vectors_step_{step:06d}.npz"
        np.savez_compressed(
            npz_path,
            think_vector=think_vec,
            explicit_mean=exp_mean,
            implicit_mean=imp_mean,
        )

        norms = [float(np.linalg.norm(think_vec[L])) for L in range(n_layers_plus_1)]
        record = {
            "step":       step,
            "words_seen": words_seen,
            "norms":      norms,
            "npz_file":   npz_path.name,
        }
        summary_records.append(record)

        # Save incrementally (safe against SLURM walltime kills)
        summary_path.write_text(
            json.dumps(
                {"condition": args.condition, "checkpoints": sorted(summary_records, key=lambda r: r["step"])},
                indent=2,
            ),
            encoding="utf-8",
        )

        log.info("  ‖v‖ per layer: %s",
                 "  ".join(f"L{i}={n:.3f}" for i, n in enumerate(norms)))

        del model
        torch.cuda.empty_cache()

    # ── Cosine similarity to final checkpoint ─────────────────────────────────
    log.info("Computing cosine similarities to final checkpoint ...")
    sorted_records = sorted(summary_records, key=lambda r: r["step"])

    final_rec = sorted_records[-1]
    final_npz = np.load(out_dir / final_rec["npz_file"])
    final_vec = final_npz["think_vector"]   # (n_layers+1, hidden_dim)

    for rec in sorted_records:
        npz_data = np.load(out_dir / rec["npz_file"])
        tvec     = npz_data["think_vector"]
        rec["cosine_sim_to_final"] = [
            cosine_sim(tvec[L], final_vec[L]) for L in range(tvec.shape[0])
        ]

    summary_path.write_text(
        json.dumps({"condition": args.condition, "checkpoints": sorted_records}, indent=2),
        encoding="utf-8",
    )
    log.info("Saved: %s", summary_path)

    # ── Summary table ─────────────────────────────────────────────────────────
    n_layers_plus_1 = final_vec.shape[0]
    focus_layer     = args.layer if args.layer is not None else (n_layers_plus_1 // 2)

    print("\n" + "=" * 72)
    print(f"  THINK VECTOR SUMMARY — condition={args.condition}  focus_layer=L{focus_layer}")
    print("=" * 72)
    print(f"  {'Step':>8}  {'Words(M)':>10}  {'‖v‖':>10}  {'CosSim→final':>14}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*14}")
    for rec in sorted_records:
        ws   = (rec["words_seen"] or 0) / 1e6
        norm = rec["norms"][focus_layer]
        csim = (rec.get("cosine_sim_to_final") or [None] * (focus_layer + 1))[focus_layer]
        csim_str = f"{csim:>14.3f}" if csim is not None else f"{'—':>14}"
        print(f"  {rec['step']:>8}  {ws:>10.2f}  {norm:>10.4f}  {csim_str}")
    print()
    print(f"  Vectors saved to: {out_dir}")
    print()


if __name__ == "__main__":
    main()
