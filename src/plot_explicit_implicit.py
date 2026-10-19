"""
src/plot_explicit_implicit.py

Produces a 3-panel figure (one panel per training condition) showing
P(Correct) for False Belief vs. True Belief across Explicit vs. Implicit
cues at the final training checkpoint — analogous to Figure 3 style plots
common in ToM / FBT literature.

Usage:
    python src/plot_explicit_implicit.py
    python src/plot_explicit_implicit.py --checkpoint last   # default
    python src/plot_explicit_implicit.py --checkpoint best   # highest fbt_overall
    python src/plot_explicit_implicit.py --out_dir results/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Palette matching the rest of the thesis plots ────────────────────────────
C_FLAT     = "#E64B35"   # red-orange
C_CHUNKED  = "#4DBBD5"   # steel blue
C_BALANCED = "#00A087"   # teal-green

FB_COLOR   = "#E8735A"   # salmon-red  for False Belief lines
TB_COLOR   = "#4DBBD5"   # steel blue  for True Belief lines
CHANCE     = "#999999"

CONDITIONS = [
    ("flat",     "Sliding-Window\n(Flat)",     C_FLAT),
    ("chunked",  "EOS-Chunked",                C_CHUNKED),
    ("balanced", "DENSITY-Balanced\n(control)", C_BALANCED),
]

# ── Possible field names the eval script might use ───────────────────────────
# We try each candidate and use the first one that exists in the checkpoint.
_FB_EXPLICIT_CANDIDATES  = ["fbt_false_belief_explicit", "fbt_fb_explicit",   "fbt_explicit_fb"]
_FB_IMPLICIT_CANDIDATES  = ["fbt_false_belief_implicit", "fbt_fb_implicit",   "fbt_implicit_fb"]
_TB_EXPLICIT_CANDIDATES  = ["fbt_true_belief_explicit",  "fbt_tb_explicit",   "fbt_explicit_tb"]
_TB_IMPLICIT_CANDIDATES  = ["fbt_true_belief_implicit",  "fbt_tb_implicit",   "fbt_implicit_tb"]

# Fallback: if no split-by-cue keys exist, use the overall FB/TB keys
# (produces a flat pair of points rather than crossing lines, but still informative)
_FB_OVERALL_CANDIDATES   = ["fbt_false_belief", "fbt_fb"]
_TB_OVERALL_CANDIDATES   = ["fbt_true_belief",  "fbt_tb"]


def _first(ckpt: dict, candidates: list[str]) -> float | None:
    for k in candidates:
        if k in ckpt and ckpt[k] is not None:
            return float(ckpt[k])
    return None


def load_final_checkpoint(results_dir: Path, cond: str, mode: str) -> dict | None:
    p = results_dir / cond / "checkpoint_evals.json"
    if not p.exists():
        print(f"  [WARN] not found: {p}", file=sys.stderr)
        return None
    data  = json.loads(p.read_text(encoding="utf-8"))
    ckpts = [c for c in data.get("checkpoints", []) if c.get("words_seen")]
    if not ckpts:
        return None
    if mode == "best":
        return max(ckpts, key=lambda c: c.get("fbt_overall", 0.0))
    return max(ckpts, key=lambda c: c["words_seen"])   # "last"


def extract_values(ckpt: dict) -> dict:
    """Return a dict with keys fb_explicit, fb_implicit, tb_explicit, tb_implicit.
    Falls back to overall FB/TB if per-cue keys are missing."""
    fb_exp = _first(ckpt, _FB_EXPLICIT_CANDIDATES)
    fb_imp = _first(ckpt, _FB_IMPLICIT_CANDIDATES)
    tb_exp = _first(ckpt, _TB_EXPLICIT_CANDIDATES)
    tb_imp = _first(ckpt, _TB_IMPLICIT_CANDIDATES)

    # If no explicit/implicit split, fall back to the overall value for both
    if fb_exp is None:
        fb_ovr = _first(ckpt, _FB_OVERALL_CANDIDATES)
        fb_exp = fb_imp = fb_ovr
    if tb_exp is None:
        tb_ovr = _first(ckpt, _TB_OVERALL_CANDIDATES)
        tb_exp = tb_imp = tb_ovr

    return {
        "fb_explicit": fb_exp,
        "fb_implicit": fb_imp,
        "tb_explicit": tb_exp,
        "tb_implicit": tb_imp,
        "words_seen":  ckpt.get("words_seen", 0),
        "step":        ckpt.get("step", 0),
    }


def print_available_keys(results_dir: Path) -> None:
    """Debug helper: print all FBT-related keys found in the last checkpoint."""
    for cond, _, _ in CONDITIONS:
        ckpt = load_final_checkpoint(results_dir, cond, "last")
        if ckpt is None:
            continue
        keys = sorted(k for k in ckpt if "fbt" in k.lower())
        print(f"\n{cond} FBT keys (step={ckpt.get('step')}, "
              f"{ckpt.get('words_seen', 0)/1e6:.1f}M words):")
        for k in keys:
            print(f"  {k}: {ckpt[k]}")


def plot(results_dir: Path, out_dir: Path, mode: str) -> None:
    fig, axes = plt.subplots(
        1, 3,
        figsize=(12, 4.5),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    x_pos    = [0, 1]
    x_labels = ["Explicit", "Implicit"]

    for ax, (cond, label, accent_col) in zip(axes, CONDITIONS):
        ckpt = load_final_checkpoint(results_dir, cond, mode)
        if ckpt is None:
            ax.set_visible(False)
            continue

        vals = extract_values(ckpt)
        words_m = vals["words_seen"] / 1e6

        # ── chance line ───────────────────────────────────────────────────
        ax.axhline(0.5, color=CHANCE, lw=1.2, ls="--", zorder=1, alpha=0.8)

        missing = any(v is None for v in [
            vals["fb_explicit"], vals["fb_implicit"],
            vals["tb_explicit"], vals["tb_implicit"],
        ])

        if not missing:
            fb_vals = [vals["fb_explicit"], vals["fb_implicit"]]
            tb_vals = [vals["tb_explicit"], vals["tb_implicit"]]

            # ── lines ─────────────────────────────────────────────────────
            ax.plot(x_pos, fb_vals, color=FB_COLOR, lw=2.0, zorder=3,
                    marker="o", markersize=8, markerfacecolor=FB_COLOR,
                    markeredgecolor="white", markeredgewidth=1.5,
                    label="False Belief")
            ax.plot(x_pos, tb_vals, color=TB_COLOR, lw=2.0, zorder=3,
                    marker="o", markersize=8, markerfacecolor=TB_COLOR,
                    markeredgecolor="white", markeredgewidth=1.5,
                    label="True Belief")

            # ── annotate values ───────────────────────────────────────────
            for xi, (fv, tv) in enumerate(zip(fb_vals, tb_vals)):
                ax.text(xi, fv + 0.025, f"{fv:.2f}", ha="center", va="bottom",
                        fontsize=8.5, color=FB_COLOR, fontweight="bold")
                ax.text(xi, tv - 0.035, f"{tv:.2f}", ha="center", va="top",
                        fontsize=8.5, color=TB_COLOR, fontweight="bold")
        else:
            # Fallback: only overall FB / TB available
            fb_ovr = _first(ckpt, _FB_OVERALL_CANDIDATES)
            tb_ovr = _first(ckpt, _TB_OVERALL_CANDIDATES)
            if fb_ovr is not None:
                ax.axhline(fb_ovr, color=FB_COLOR, lw=2.0, ls="-", label="False Belief (overall)")
            if tb_ovr is not None:
                ax.axhline(tb_ovr, color=TB_COLOR, lw=2.0, ls="-", label="True Belief (overall)")
            ax.text(0.5, 0.05, "No explicit/implicit split available",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=8, color="#888888", style="italic")

        # ── panel styling ─────────────────────────────────────────────────
        ax.set_title(label, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(0.10, 1.05)
        ax.set_facecolor("#F8F8F8")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")
        ax.spines["left"].set_color("#cccccc")
        ax.tick_params(colors="#555555")
        ax.grid(axis="y", color="#e0e0e0", lw=0.8, zorder=0)

        # Subtitle: words seen
        ax.text(0.5, -0.17, f"{words_m:.1f}M words",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color="#777777", style="italic")

        # Accent bar at top of panel
        ax.axhline(1.04, xmin=0.0, xmax=1.0, color=accent_col,
                   lw=4, solid_capstyle="butt", clip_on=False, zorder=4)

    axes[0].set_ylabel("P(Correct)", fontsize=12)

    # ── global legend ─────────────────────────────────────────────────────────
    fb_patch = mpatches.Patch(color=FB_COLOR, label="False Belief")
    tb_patch = mpatches.Patch(color=TB_COLOR, label="True Belief")
    ch_patch = mpatches.Patch(color=CHANCE,   label="Chance (0.50)")
    fig.legend(
        handles=[fb_patch, tb_patch, ch_patch],
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(
        "FBT Accuracy by Cue Type (Explicit vs. Implicit) — Final Checkpoint",
        fontsize=13, fontweight="bold", y=1.03,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"fbt_explicit_implicit_{mode}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", choices=["last", "best"], default="last",
                   help="Which checkpoint to use per condition")
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--debug_keys", action="store_true",
                   help="Print available FBT keys then exit")
    args = p.parse_args()

    base        = Path(__file__).parent.parent
    results_dir = base / "results"
    out_dir     = args.out_dir or base / "results" / "figures"

    if args.debug_keys:
        print_available_keys(results_dir)
        return

    plot(results_dir, out_dir, args.checkpoint)


if __name__ == "__main__":
    main()
