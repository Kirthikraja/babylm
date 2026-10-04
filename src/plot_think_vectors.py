"""
src/plot_think_vectors.py

Visualises think vector emergence from results/think_vectors/<condition>/
think_vector_summary.json.

Figures produced:
  1. think_vector_emergence.png  — CosSim-to-final + norm across training
                                   for the focus layer (default: L6)
  2. think_vector_all_layers.png — CosSim heatmap: words × layer,
                                   shows which layer leads emergence
  3. think_vector_fbt_overlay.png — CosSim (L6) + FBT accuracy on same axis,
                                    shows whether the vector tracks behaviour

Usage:
    python src/plot_think_vectors.py
    python src/plot_think_vectors.py --condition flat --focus_layer 6
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO, stream=sys.stdout,
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.facecolor":     "#f2f2f2",
    "figure.facecolor":   "white",
    "axes.grid":          True,
    "grid.color":         "white",
    "grid.linewidth":     1.2,
    "grid.linestyle":     "-",
    "axes.axisbelow":     True,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
    "figure.dpi":         150,
    "xtick.bottom":       False,
    "ytick.left":         False,
})

CORAL = "#E8735A"
TEAL  = "#47A9A9"
GRAY  = "#555555"


# ── Figure 1: Emergence curve for one layer ───────────────────────────────────

def plot_emergence(records: list[dict], focus_layer: int, out_dir: Path) -> None:
    xs    = [r["words_seen"] / 1e6 for r in records if r.get("words_seen")]
    norms = [r["norms"][focus_layer] for r in records if r.get("words_seen")]
    cosims = [
        r["cosine_sim_to_final"][focus_layer]
        for r in records
        if r.get("words_seen") and r.get("cosine_sim_to_final")
    ]

    if len(xs) != len(cosims):
        xs_cos = [r["words_seen"] / 1e6 for r in records
                  if r.get("words_seen") and r.get("cosine_sim_to_final")]
    else:
        xs_cos = xs

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})

    # ── Top: CosSim ───────────────────────────────────────────────────────────
    ax1.axhline(0,   color="#aaaaaa", lw=1.0, ls="--", zorder=1)
    ax1.axhline(0.5, color="#aaaaaa", lw=0.8, ls=":",  zorder=1, alpha=0.7)
    ax1.axhline(1.0, color="#aaaaaa", lw=0.8, ls=":",  zorder=1, alpha=0.5)

    # Shade emergence zone
    emergence_start = next((x for x, c in zip(xs_cos, cosims) if c > 0.1), None)
    emergence_stable = next((x for x, c in zip(xs_cos, cosims) if c > 0.9), None)
    if emergence_start and emergence_stable:
        ax1.axvspan(emergence_start, emergence_stable, color=TEAL, alpha=0.07,
                    zorder=0, label=f"Emergence zone ({emergence_start:.0f}–{emergence_stable:.0f}M words)")

    ax1.plot(xs_cos, cosims, color=TEAL, lw=2.0, marker="o", markersize=3,
             zorder=3, label=f"Cosine sim to final (L{focus_layer})")

    ax1.set_ylabel("Cosine Similarity to\nFinal Checkpoint", fontsize=12)
    ax1.set_ylim(-0.15, 1.10)
    ax1.set_title(
        f"Think Vector Emergence During Training — Flat Model (Layer {focus_layer})",
        fontsize=13, fontweight="bold",
    )
    ax1.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="#cccccc",
               loc="lower right")

    # ── Bottom: Norm ──────────────────────────────────────────────────────────
    ax2.plot(xs, norms, color=CORAL, lw=1.8, marker="o", markersize=2.5,
             zorder=3, label=f"‖think vector‖ (L{focus_layer})")
    ax2.set_xlabel("Words Seen During Training (M)", fontsize=12)
    ax2.set_ylabel("L2 Norm", fontsize=12)
    ax2.set_ylim(0, max(norms) * 1.2)
    ax2.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="#cccccc",
               loc="upper right")

    fig.tight_layout()
    out = out_dir / "think_vector_emergence.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Figure 2: All-layer CosSim heatmap ───────────────────────────────────────

def plot_all_layers_heatmap(records: list[dict], out_dir: Path) -> None:
    records_with_cos = [r for r in records
                        if r.get("words_seen") and r.get("cosine_sim_to_final")]
    if not records_with_cos:
        log.warning("No cosine_sim_to_final data — skipping heatmap")
        return

    xs      = np.array([r["words_seen"] / 1e6 for r in records_with_cos])
    n_layers = len(records_with_cos[0]["cosine_sim_to_final"])

    # matrix: (n_layers, n_checkpoints)
    mat = np.array([r["cosine_sim_to_final"] for r in records_with_cos]).T

    # Thin to at most 200 columns for readability
    if mat.shape[1] > 200:
        idx = np.linspace(0, mat.shape[1] - 1, 200, dtype=int)
        mat = mat[:, idx]
        xs  = xs[idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        mat, aspect="auto", origin="lower",
        extent=[xs[0], xs[-1], -0.5, n_layers - 0.5],
        cmap="RdYlGn", vmin=-0.2, vmax=1.0,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Cosine Similarity to Final Checkpoint", fontsize=10)

    ax.set_xlabel("Words Seen During Training (M)", fontsize=12)
    ax.set_ylabel("Transformer Layer", fontsize=12)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=8)
    ax.set_title("Think Vector Direction Stability by Layer (Red = random, Green = converged)",
                 fontsize=12, fontweight="bold")

    # Remove spines for heatmap (they look wrong on imshow)
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.tight_layout()
    out = out_dir / "think_vector_all_layers.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Figure 3: CosSim overlaid with FBT accuracy ───────────────────────────────

def plot_fbt_overlay(
    records:    list[dict],
    fbt_path:   Path,
    focus_layer: int,
    out_dir:    Path,
) -> None:
    if not fbt_path.exists():
        log.warning("checkpoint_evals.json not found at %s — skipping overlay", fbt_path)
        return

    fbt_data = json.loads(fbt_path.read_text(encoding="utf-8"))
    fbt_ckpts = sorted(fbt_data.get("checkpoints", []),
                       key=lambda c: c["step"])

    cos_records = [r for r in records
                   if r.get("words_seen") and r.get("cosine_sim_to_final")]

    xs_cos  = [r["words_seen"] / 1e6 for r in cos_records]
    cosims  = [r["cosine_sim_to_final"][focus_layer] for r in cos_records]

    xs_fbt  = [c["words_seen"] / 1e6 for c in fbt_ckpts if c.get("words_seen")]
    fbt_fb  = [c["fbt_false_belief"] for c in fbt_ckpts if c.get("words_seen")]
    fbt_tb  = [c["fbt_true_belief"]  for c in fbt_ckpts if c.get("words_seen")]
    fbt_ov  = [c["fbt_overall"]      for c in fbt_ckpts if c.get("words_seen")]

    if not xs_fbt:
        log.warning("No words_seen in checkpoint_evals.json — skipping overlay")
        return

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    # Think vector (left axis)
    ax1.plot(xs_cos, cosims, color=TEAL, lw=2.0, marker="o", markersize=3,
             zorder=4, label=f"Think vector CosSim (L{focus_layer})")
    ax1.set_ylabel("Cosine Similarity to Final Think Vector", fontsize=11, color=TEAL)
    ax1.set_ylim(-0.15, 1.10)
    ax1.tick_params(axis="y", labelcolor=TEAL)

    # FBT accuracy (right axis)
    ax2.plot(xs_fbt, fbt_fb,  color=CORAL,  lw=1.8, marker="o", markersize=4,
             zorder=3, label="False Belief accuracy")
    ax2.plot(xs_fbt, fbt_tb,  color="#47A9A9", lw=1.8, marker="o", markersize=4,
             zorder=3, label="True Belief accuracy")
    ax2.axhline(0.5, color="#aaaaaa", lw=1.0, ls="--", zorder=1)
    ax2.set_ylabel("P(Correct)", fontsize=11)
    ax2.set_ylim(0.1, 1.0)

    ax1.set_xlabel("Words Seen During Training (M)", fontsize=12)
    ax1.set_title(
        f"Think Vector Emergence vs FBT Behaviour (Flat Model, L{focus_layer})",
        fontsize=12, fontweight="bold",
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=9, frameon=True, facecolor="white", edgecolor="#cccccc",
               loc="lower right")

    ax1.set_facecolor("#f2f2f2")
    ax1.grid(True, color="white", linewidth=1.2)

    fig.tight_layout()
    out = out_dir / "think_vector_fbt_overlay.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Print emergence stats ─────────────────────────────────────────────────────

def print_emergence_stats(records: list[dict], focus_layer: int) -> None:
    cos_records = [r for r in records
                   if r.get("words_seen") and r.get("cosine_sim_to_final")]
    if not cos_records:
        return

    def find_crossing(threshold: float) -> float | None:
        for r in cos_records:
            cs = r["cosine_sim_to_final"][focus_layer]
            if cs >= threshold:
                return r["words_seen"] / 1e6
        return None

    print("\n" + "=" * 60)
    print(f"  THINK VECTOR EMERGENCE STATS — Layer {focus_layer}")
    print("=" * 60)
    for thresh in [0.1, 0.5, 0.9, 0.99]:
        w = find_crossing(thresh)
        if w:
            print(f"  CosSim ≥ {thresh:.2f}  first at  {w:.1f}M words")
        else:
            print(f"  CosSim ≥ {thresh:.2f}  never reached")

    # Best layer by first crossing 0.5
    all_layers = len(cos_records[0]["cosine_sim_to_final"])
    print(f"\n  First CosSim ≥ 0.50 per layer:")
    for L in range(all_layers):
        w = next(
            (r["words_seen"] / 1e6 for r in cos_records
             if r["cosine_sim_to_final"][L] >= 0.5),
            None,
        )
        marker = " ← focus" if L == focus_layer else ""
        if w:
            print(f"    L{L:>2}:  {w:>6.1f}M words{marker}")
        else:
            print(f"    L{L:>2}:  never{marker}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--condition",   choices=["chunked", "flat", "balanced"], default="flat")
    p.add_argument("--focus_layer", type=int, default=6,
                   help="Layer to highlight in the emergence figure")
    p.add_argument("--out_dir",     type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    base    = Path(__file__).parent.parent
    tv_dir  = base / "results" / "think_vectors" / args.condition
    out_dir = args.out_dir or base / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = tv_dir / "think_vector_summary.json"
    if not summary_path.exists():
        log.error("Not found: %s — run extract_think_vectors.py first", summary_path)
        sys.exit(1)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = sorted(summary.get("checkpoints", []), key=lambda r: r["step"])

    if not records:
        log.error("No checkpoint records in %s", summary_path)
        sys.exit(1)

    log.info("Loaded %d checkpoint records", len(records))

    print_emergence_stats(records, args.focus_layer)

    plot_emergence(records, args.focus_layer, out_dir)
    plot_all_layers_heatmap(records, out_dir)

    fbt_path = base / "results" / args.condition / "checkpoint_evals.json"
    plot_fbt_overlay(records, fbt_path, args.focus_layer, out_dir)

    log.info("Done. Figures saved to %s", out_dir)


if __name__ == "__main__":
    main()
