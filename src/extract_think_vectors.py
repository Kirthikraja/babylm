"""
src/extract_think_vectors.py

Extracts "think vectors" from each training checkpoint using three formulations:

  F1 (difference-in-means):
      v = mean(explicit_states) - mean(implicit_states)   [Marks & Tegmark 2023]

  F2 (linear probe direction):
      Fit a linear SVM on explicit vs implicit states; take the weight vector as v.

  F3 (leave-one-out stability):
      For each item i, compute v_LOO_i = mean(explicit \ {i}) - mean(implicit \ {i})
      (where {i} is whichever class item i belongs to).
      Report the mean cosine similarity of LOO vectors to the full v (F1), as a
      stability metric.  The mean LOO vector itself is also saved as an alternative
      think vector.

All three are computed per layer (embedding + 12 transformer blocks = 13 layers).

Outputs per checkpoint (in results/think_vectors/<condition>/):
  vectors_step_NNNNNN.npz   keys: think_vector_f1, think_vector_f2,
                                  think_vector_f3,  explicit_mean, implicit_mean
                            each: (n_layers+1, hidden_dim)
  think_vector_summary.json — per-checkpoint norms + cosine-sims to final, all 3 formulations

Usage:
    python src/extract_think_vectors.py --condition flat
    python src/extract_think_vectors.py --condition full_baseline --all_conditions
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
ABLATION_CONDITIONS = [
    "full_baseline", "no_fiction", "no_childes", "no_dialogue", "no_wikipedia",
]


# ── Activation extraction ─────────────────────────────────────────────────────

def get_last_token_hidden_states(
    model:     GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    context:   str,
    device:    torch.device,
) -> np.ndarray | None:
    """Last-token hidden state from every layer for one context string.

    Returns (n_layers+1, hidden_dim) or None if the context tokenises to 0 tokens.
    """
    ids = tokenizer.encode(context, add_special_tokens=False)
    if not ids:
        return None

    input_ids = torch.tensor([ids]).to(device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    return np.stack(
        [hs[0, -1, :].float().cpu().numpy() for hs in outputs.hidden_states],
        axis=0,
    )


# ── Think vector formulations ─────────────────────────────────────────────────

def formulation_f1(
    explicit_states: list[np.ndarray],
    implicit_states: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Difference-in-means: v = mean(explicit) - mean(implicit).

    Returns (think_vector, explicit_mean, implicit_mean) each (n_layers+1, hidden_dim).
    """
    exp_stack = np.stack(explicit_states)   # (n_exp, n_layers+1, hidden)
    imp_stack = np.stack(implicit_states)
    exp_mean  = exp_stack.mean(axis=0)
    imp_mean  = imp_stack.mean(axis=0)
    return exp_mean - imp_mean, exp_mean, imp_mean


def formulation_f2(
    explicit_states: list[np.ndarray],
    implicit_states: list[np.ndarray],
) -> np.ndarray:
    """Linear probe direction via SVM.

    For each layer, fit a linear SVM (C=1) on the binary classification task
    explicit=+1 vs implicit=-1.  The weight vector (normalised) is the think vector.

    Returns (n_layers+1, hidden_dim).
    """
    try:
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        log.warning("F2: sklearn not available — returning zeros")
        n_layers = explicit_states[0].shape[0]
        hidden   = explicit_states[0].shape[1]
        return np.zeros((n_layers, hidden))

    exp_stack = np.stack(explicit_states)   # (n_exp, n_layers+1, hidden)
    imp_stack = np.stack(implicit_states)
    n_layers  = exp_stack.shape[1]
    hidden    = exp_stack.shape[2]

    all_states = np.concatenate([exp_stack, imp_stack], axis=0)  # (n, layers, hidden)
    labels     = np.array([1] * len(explicit_states) + [-1] * len(implicit_states))

    vectors = np.zeros((n_layers, hidden))
    for L in range(n_layers):
        X = all_states[:, L, :]        # (n, hidden)
        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)
        svm    = LinearSVC(C=1.0, max_iter=2000, random_state=42)
        try:
            svm.fit(X_s, labels)
            w = svm.coef_[0]           # (hidden,)
            norm = np.linalg.norm(w)
            vectors[L] = w / norm if norm > 1e-10 else w
        except Exception:
            pass   # keep zero if SVM fails to converge

    return vectors


def formulation_f3(
    explicit_states: list[np.ndarray],
    implicit_states: list[np.ndarray],
    f1_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out stability.

    For each item i, hold it out from its class and recompute F1.
    Returns:
        loo_mean_vector  (n_layers+1, hidden_dim) — mean of all LOO vectors
        loo_cosine_sims  (n_items, n_layers+1)    — cosine similarity of each LOO vector to F1
    """
    n_exp     = len(explicit_states)
    n_imp     = len(implicit_states)
    n_items   = n_exp + n_imp
    n_layers  = explicit_states[0].shape[0]
    hidden    = explicit_states[0].shape[1]

    exp_stack = np.stack(explicit_states)
    imp_stack = np.stack(implicit_states)
    exp_sum   = exp_stack.sum(axis=0)   # (n_layers+1, hidden)
    imp_sum   = imp_stack.sum(axis=0)

    loo_vectors: list[np.ndarray] = []
    cosine_sims: list[np.ndarray] = []

    for i in range(n_exp):
        exp_loo = (exp_sum - exp_stack[i]) / (n_exp - 1) if n_exp > 1 else np.zeros_like(exp_sum)
        imp_loo = imp_sum / n_imp
        v_loo   = exp_loo - imp_loo
        loo_vectors.append(v_loo)
        sims = np.array([cosine_sim(v_loo[L], f1_vector[L]) for L in range(n_layers)])
        cosine_sims.append(sims)

    for i in range(n_imp):
        exp_loo = exp_sum / n_exp
        imp_loo = (imp_sum - imp_stack[i]) / (n_imp - 1) if n_imp > 1 else np.zeros_like(imp_sum)
        v_loo   = exp_loo - imp_loo
        loo_vectors.append(v_loo)
        sims = np.array([cosine_sim(v_loo[L], f1_vector[L]) for L in range(n_layers)])
        cosine_sims.append(sims)

    loo_mean = np.stack(loo_vectors).mean(axis=0)   # (n_layers+1, hidden)
    loo_sims = np.stack(cosine_sims)                # (n_items, n_layers+1)
    return loo_mean, loo_sims


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def extract_all_formulations(
    model:     GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    df:        pd.DataFrame,
    device:    torch.device,
) -> dict:
    """Compute F1/F2/F3 think vectors at all layers for one checkpoint.

    Returns dict with keys:
        think_vector_f1, think_vector_f2, think_vector_f3,
        explicit_mean, implicit_mean,
        loo_cosine_sims  (mean per layer, for summary)
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

    if n_skipped:
        log.warning("Skipped %d items with empty tokenisation", n_skipped)
    if not explicit_states or not implicit_states:
        raise ValueError(
            f"Need both Explicit and Implicit items. Got "
            f"{len(explicit_states)} explicit, {len(implicit_states)} implicit."
        )

    tv_f1, exp_mean, imp_mean = formulation_f1(explicit_states, implicit_states)
    tv_f2 = formulation_f2(explicit_states, implicit_states)
    tv_f3, loo_sims = formulation_f3(explicit_states, implicit_states, tv_f1)

    # F2 may have one fewer layer if SVM skipped embedding layer (not for F2)
    # Pad F2 to match F1 shape if needed
    if tv_f2.shape[0] < tv_f1.shape[0]:
        pad = np.zeros((tv_f1.shape[0] - tv_f2.shape[0], tv_f1.shape[1]))
        tv_f2 = np.concatenate([pad, tv_f2], axis=0)

    return {
        "think_vector_f1": tv_f1,
        "think_vector_f2": tv_f2,
        "think_vector_f3": tv_f3,
        "explicit_mean":   exp_mean,
        "implicit_mean":   imp_mean,
        "loo_cosine_sims": loo_sims.mean(axis=0).tolist(),  # (n_layers+1,) mean stability
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--condition",    default="flat",
                   help="Condition name (directory under models/)")
    p.add_argument("--data_path",    type=Path, default=None,
                   help="Path to fb.csv (default: data/fbt/fb.csv)")
    p.add_argument("--out_dir",      type=Path, default=None,
                   help="Where to save results (default: results/think_vectors/<condition>)")
    p.add_argument("--layer",        type=int,  default=None,
                   help="Layer to highlight in summary (0=embed, 1-12=blocks)")
    p.add_argument("--all_conditions", action="store_true",
                   help="Run on all ablation conditions (ignores --condition)")
    return p.parse_args()


def run_condition(condition: str, base: Path, data_path: Path, out_dir_root: Path,
                  focus_layer: int | None = None) -> None:
    models_dir = base / "models" / condition
    out_dir    = out_dir_root / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Find checkpoints ──────────────────────────────────────────────────────
    checkpoints = sorted(
        [d for d in models_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not checkpoints:
        log.error("No checkpoint-* directories in %s", models_dir)
        return
    log.info("Found %d checkpoints for condition=%s", len(checkpoints), condition)

    # ── FBT stimuli ───────────────────────────────────────────────────────────
    if not data_path.exists():
        log.error("FBT stimuli not found at %s", data_path)
        return
    df = pd.read_csv(data_path)
    if "first_mention" in df.columns and "recent_mention" in df.columns:
        df = df[(df["first_mention"] == "Start") & (df["recent_mention"] == "End")]
    if "knowledge_cue" not in df.columns:
        log.error("fb.csv missing 'knowledge_cue' column")
        return
    n_exp = (df["knowledge_cue"].str.lower() == "explicit").sum()
    n_imp = (df["knowledge_cue"].str.lower() == "implicit").sum()
    log.info("FBT items: %d total  (%d explicit, %d implicit)", len(df), n_exp, n_imp)

    # ── Words-seen mapping ────────────────────────────────────────────────────
    curves_path   = base / "results" / condition / "training_curves.json"
    step_to_words: dict[int, int] = {}
    if curves_path.exists():
        curves        = json.loads(curves_path.read_text(encoding="utf-8"))
        step_to_words = {
            int(s): int(w)
            for s, w in zip(curves.get("step", []), curves.get("words_seen", []))
        }

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    final_dir = models_dir / "final"
    if final_dir.exists():
        tokenizer_path = str(final_dir)
    else:
        first_with_tok = next(
            (d for d in checkpoints
             if (d / "tokenizer.json").exists() or (d / "vocab.json").exists()),
            None,
        )
        tokenizer_path = str(first_with_tok) if first_with_tok else "gpt2"
    log.info("Tokenizer: %s", tokenizer_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Resume ────────────────────────────────────────────────────────────────
    summary_path     = out_dir / "think_vector_summary.json"
    done_steps:      set[int]   = set()
    summary_records: list[dict] = []
    if summary_path.exists():
        try:
            prev            = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_records = prev.get("checkpoints", [])
            done_steps      = {r["step"] for r in summary_records}
            log.info("Resuming: %d done, %d remaining",
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

        result    = extract_all_formulations(model, tokenizer, df, device)
        n_layers  = result["think_vector_f1"].shape[0]

        npz_path  = out_dir / f"vectors_step_{step:06d}.npz"
        np.savez_compressed(
            npz_path,
            think_vector_f1 = result["think_vector_f1"],
            think_vector_f2 = result["think_vector_f2"],
            think_vector_f3 = result["think_vector_f3"],
            explicit_mean   = result["explicit_mean"],
            implicit_mean   = result["implicit_mean"],
        )

        record = {
            "step":          step,
            "words_seen":    words_seen,
            "norms_f1":      [float(np.linalg.norm(result["think_vector_f1"][L]))
                              for L in range(n_layers)],
            "norms_f2":      [float(np.linalg.norm(result["think_vector_f2"][L]))
                              for L in range(n_layers)],
            "norms_f3":      [float(np.linalg.norm(result["think_vector_f3"][L]))
                              for L in range(n_layers)],
            "loo_stability": result["loo_cosine_sims"],  # mean cosine F1_LOO → F1 per layer
            "npz_file":      npz_path.name,
        }
        summary_records.append(record)
        summary_path.write_text(
            json.dumps(
                {"condition": condition,
                 "checkpoints": sorted(summary_records, key=lambda r: r["step"])},
                indent=2,
            ),
            encoding="utf-8",
        )

        log.info("  F1 ‖v‖: %s",
                 "  ".join(f"L{i}={n:.3f}" for i, n in enumerate(record["norms_f1"])))
        del model
        torch.cuda.empty_cache()

    # ── Cosine similarity to final checkpoint ─────────────────────────────────
    if not summary_records:
        log.warning("No records to compute cosine similarity")
        return

    sorted_records = sorted(summary_records, key=lambda r: r["step"])
    final_npz      = np.load(out_dir / sorted_records[-1]["npz_file"])

    for rec in sorted_records:
        npz  = np.load(out_dir / rec["npz_file"])
        for form in ("f1", "f2", "f3"):
            final_v = final_npz[f"think_vector_{form}"]
            curr_v  = npz[f"think_vector_{form}"]
            rec[f"cosine_sim_to_final_{form}"] = [
                cosine_sim(curr_v[L], final_v[L]) for L in range(curr_v.shape[0])
            ]

    summary_path.write_text(
        json.dumps({"condition": condition, "checkpoints": sorted_records}, indent=2),
        encoding="utf-8",
    )
    log.info("Saved: %s", summary_path)

    # ── Summary table ─────────────────────────────────────────────────────────
    n_layers_plus_1 = final_npz["think_vector_f1"].shape[0]
    fl = focus_layer if focus_layer is not None else n_layers_plus_1 // 2

    print("\n" + "=" * 80)
    print(f"  THINK VECTOR SUMMARY  condition={condition}  layer=L{fl}")
    print("=" * 80)
    print(f"  {'Step':>8}  {'Words(M)':>10}  {'‖F1‖':>8}  {'‖F2‖':>8}  "
          f"{'CosSim F1':>10}  {'LOO stab':>10}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
    for rec in sorted_records:
        ws      = (rec["words_seen"] or 0) / 1e6
        n_f1    = rec["norms_f1"][fl]
        n_f2    = rec["norms_f2"][fl]
        cs_f1   = (rec.get("cosine_sim_to_final_f1") or [None]*(fl+1))[fl]
        loo     = (rec.get("loo_stability") or [None]*(fl+1))[fl]
        cs_str  = f"{cs_f1:.3f}" if cs_f1 is not None else "—"
        loo_str = f"{loo:.3f}"   if loo  is not None else "—"
        print(f"  {rec['step']:>8}  {ws:>10.2f}  {n_f1:>8.4f}  {n_f2:>8.4f}  "
              f"  {cs_str:>10}  {loo_str:>10}")
    print()
    print(f"  Vectors saved to: {out_dir}")
    print()


def main() -> None:
    args = parse_args()
    base      = Path(__file__).parent.parent
    data_path = args.data_path or base / "data" / "fbt" / "fb.csv"
    out_root  = base / "results" / "think_vectors"

    if args.all_conditions:
        conditions = ABLATION_CONDITIONS
    else:
        conditions = [args.condition]

    for cond in conditions:
        log.info("=" * 70)
        log.info("Processing condition: %s", cond)
        log.info("=" * 70)
        run_condition(cond, base, data_path, out_root, focus_layer=args.layer)


if __name__ == "__main__":
    main()
