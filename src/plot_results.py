"""
src/plot_results.py

Publication-quality figures for the chunked vs flat GPT-2 BabyLM comparison.

Usage:
    python src/plot_results.py
    python src/plot_results.py --results_dir results --out_dir results/figures
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
import matplotlib.patches as mpatches
import numpy as np

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO, stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── Palette & labels ────────────────────────────────────────────────────────────
# Colorblind-safe, prints well in greyscale
C_FLAT     = "#E64B35"   # red-orange   — Sliding-Window
C_CHUNKED  = "#4DBBD5"   # steel blue   — EOS-Chunked
C_BALANCED = "#00A087"   # teal-green   — DENSITY-Balanced
C_FB       = "#E64B35"   # coral-red    — False Belief trajectory
C_TB       = "#4DBBD5"   # steel blue   — True Belief trajectory
C_BLIMP    = "#7E6EAB"   # muted purple — BLiMP
C_FBT_OV   = "#3C5488"   # navy         — FBT Overall
C_EWOK     = "#F39B7F"   # peach-orange — EWoK
CHANCE_COL = "#999999"   # mid-gray     — chance line

CONDITIONS_ALL = ["flat", "chunked", "balanced"]
LABELS = {
    "flat":     "Sliding-Window (Flat)",
    "chunked":  "EOS-Chunked",
    "balanced": "DENSITY-Balanced",
}
COLORS = {
    "flat":     C_FLAT,
    "chunked":  C_CHUNKED,
    "balanced": C_BALANCED,
}
CONDITIONS = ["flat", "chunked"]   # conditions that have full training data

# ── Clean academic style (Nature / ACL) ────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "DejaVu Sans",
    "font.size":           11,
    "axes.facecolor":      "white",
    "figure.facecolor":    "white",
    "axes.grid":           True,
    "grid.color":          "#e8e8e8",
    "grid.linewidth":      0.8,
    "grid.linestyle":      "-",
    "axes.axisbelow":      True,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.spines.left":    True,
    "axes.spines.bottom":  True,
    "axes.edgecolor":      "#cccccc",
    "figure.dpi":          150,
    "xtick.bottom":        True,
    "ytick.left":          True,
    "xtick.color":         "#555555",
    "ytick.color":         "#555555",
    "axes.labelcolor":     "#222222",
    "text.color":          "#222222",
})
LW      = 2.0
LW_RAW  = 0.7
ALPHA_RAW = 0.20
SMOOTH_W  = 15
CHANCE    = 0.5


# ── Helpers ────────────────────────────────────────────────────────────────────

def _smooth(xs: np.ndarray, ys: np.ndarray, window: int = SMOOTH_W):
    """Rolling-mean smoothing; trims edge artefacts."""
    if len(ys) < window:
        return xs, ys
    kernel = np.ones(window) / window
    ys_s   = np.convolve(ys, kernel, mode="same")
    half   = window // 2
    return xs[half:-half], ys_s[half:-half]


def _plot_line(ax, xs, ys, color, label, lw=LW, smooth=True, marker=None):
    """Raw faded thin line + smoothed bold line on top (reference style)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ax.plot(xs, ys, color=color, lw=LW_RAW, alpha=ALPHA_RAW, zorder=2)
    if smooth and len(xs) >= SMOOTH_W:
        xs_s, ys_s = _smooth(xs, ys)
        ax.plot(xs_s, ys_s, color=color, lw=lw, zorder=4, label=label)
    else:
        kw = dict(marker=marker, markersize=6) if marker else {}
        ax.plot(xs, ys, color=color, lw=lw, zorder=4, label=label, **kw)


def _chance_line(ax):
    ax.axhline(CHANCE, color=CHANCE_COL, lw=1.0, ls="--", zorder=1, label="Chance (0.50)")


def _save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    log.info("Saved: %s", path)


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_training_curves(results_dir: Path, condition: str) -> dict | None:
    p = results_dir / condition / "training_curves.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_checkpoint_evals(results_dir: Path, condition: str) -> list[dict] | None:
    p = results_dir / condition / "checkpoint_evals.json"
    if not p.exists():
        log.warning("Not found: %s", p)
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    ckpts = raw.get("checkpoints", [])
    # Sort by words seen
    return sorted(
        [c for c in ckpts if c.get("words_seen") is not None],
        key=lambda c: c["words_seen"],
    )


# ── Figure 1: Training loss vs words seen (cross-condition) ────────────────────

def plot_loss_vs_words(results_dir: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plotted = False

    for cond in CONDITIONS_ALL:
        data = load_training_curves(results_dir, cond)
        if data is None:
            continue
        words = data.get("words_seen")
        loss  = (data.get("loss") or data.get("train_loss") or
                 data.get("step_loss") or data.get("step_train_loss"))
        if not words or not loss or len(words) != len(loss):
            log.warning("No step-level loss for %s — skipping", cond)
            continue

        xs = np.array(words, dtype=float) / 1e6
        ys = np.array(loss,  dtype=float)

        # Remove extreme outliers (final batch spike / first warmup spike)
        q5, q95 = np.percentile(ys, [5, 95])
        mask = (ys >= q5 * 0.5) & (ys <= q95 * 2.0)
        xs, ys = xs[mask], ys[mask]

        _plot_line(ax, xs, ys, COLORS[cond], LABELS[cond])
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Words Seen During Training (M)", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Training Loss vs. Words Seen — All Conditions",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, frameon=True, facecolor="white",
              edgecolor="#cccccc", loc="upper right")
    fig.tight_layout()
    _save(fig, out_dir / "loss_vs_words.png")


# ── Figure 2: Loss by epoch ────────────────────────────────────────────────────

def plot_training_curves(results_dir: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plotted = False

    for cond in CONDITIONS_ALL:
        data = load_training_curves(results_dir, cond)
        if data is None:
            continue
        epochs     = data.get("epoch_num")
        train_loss = data.get("epoch_train_loss")
        eval_loss  = data.get("epoch_eval_loss")
        if not epochs:
            continue
        plotted = True
        col = COLORS[cond]
        lbl = LABELS[cond]
        ax.plot(epochs, train_loss, color=col, lw=LW, ls="-",
                marker="o", markersize=5, label=f"{lbl} — Train")
        ax.plot(epochs, eval_loss, color=col, lw=LW, ls="--",
                marker="s", markersize=4, label=f"{lbl} — Val")

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Training & Validation Loss by Epoch",
                 fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    solid = plt.Line2D([0], [0], color="#666", lw=LW, ls="-",  label="Train")
    dash  = plt.Line2D([0], [0], color="#666", lw=LW, ls="--", label="Validation")
    h, l  = ax.get_legend_handles_labels()
    ax.legend(h + [solid, dash], l + ["Train", "Validation"],
              fontsize=9, framealpha=0.8, loc="upper right")
    fig.tight_layout()
    _save(fig, out_dir / "training_curves.png")


# ── Figure 3: BLiMP + FBT training panel — one per condition ──────────────────

def plot_training_panel_combined(results_dir: Path, out_dir: Path) -> None:
    """
    Reference-style panel per condition:
    Raw faded thin lines + smoothed bold lines.
    Four metrics: BLiMP, FBT Overall, True Belief, False Belief.
    """
    for cond in CONDITIONS_ALL:
        ckpts = load_checkpoint_evals(results_dir, cond)
        if not ckpts:
            continue

        xs     = np.array([c["words_seen"] / 1e6 for c in ckpts])
        y_fbt  = np.array([c["fbt_overall"]      for c in ckpts], dtype=float)
        y_tb   = np.array([c["fbt_true_belief"]  for c in ckpts], dtype=float)
        y_fb   = np.array([c["fbt_false_belief"] for c in ckpts], dtype=float)

        # BLiMP: only the checkpoints that were evaluated
        blimp_ckpts = [c for c in ckpts if "blimp" in c]
        xs_b  = np.array([c["words_seen"] / 1e6 for c in blimp_ckpts])
        y_b   = np.array([c["blimp"]            for c in blimp_ckpts], dtype=float)

        fig, ax = plt.subplots(figsize=(10, 5))

        # Shaded band between FB and TB — visual gap indicator
        if len(xs) >= SMOOTH_W:
            _, tb_s = _smooth(xs, y_tb)
            _, fb_s = _smooth(xs, y_fb)
            xs_mid  = xs[SMOOTH_W//2:-(SMOOTH_W//2)]
            ax.fill_between(xs_mid, tb_s, fb_s, alpha=0.07, color="#888888", zorder=1)

        if len(xs_b) > 0:
            _plot_line(ax, xs_b, y_b,   C_BLIMP,  "BLiMP",        smooth=(len(xs_b) >= SMOOTH_W))
        _plot_line(ax, xs,   y_fbt, C_FBT_OV, "FBT Overall")
        _plot_line(ax, xs,   y_tb,  C_TB,     "True Belief")
        _plot_line(ax, xs,   y_fb,  C_FB,     "False Belief")

        _chance_line(ax)

        ax.set_xlabel("Words Seen During Training (M)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(0.15, 1.0)
        ax.set_title(
            f"BLiMP & FBT Accuracy During Training — {LABELS[cond]}",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax.legend(fontsize=10, frameon=True, facecolor="white",
                  edgecolor="#cccccc", loc="lower right")
        fig.tight_layout()
        _save(fig, out_dir / f"training_panel_{cond}.png")


# ── Figure 4: FBT by knowledge cue over training ──────────────────────────────

def plot_fbt_knowledge_cue_over_training(results_dir: Path, out_dir: Path) -> None:
    for cond in CONDITIONS_ALL:
        ckpts = load_checkpoint_evals(results_dir, cond)
        if not ckpts:
            continue
        ckpts = [c for c in ckpts if c.get("fbt_explicit_false_belief") is not None]
        if not ckpts:
            log.warning("No knowledge_cue data for %s — skipping", cond)
            continue

        xs     = np.array([c["words_seen"] / 1e6              for c in ckpts])
        exp_fb = np.array([c["fbt_explicit_false_belief"]      for c in ckpts], dtype=float)
        imp_fb = np.array([c["fbt_implicit_false_belief"]      for c in ckpts], dtype=float)
        exp_tb = np.array([c.get("fbt_explicit_true_belief", np.nan) for c in ckpts], dtype=float)
        imp_tb = np.array([c.get("fbt_implicit_true_belief", np.nan) for c in ckpts], dtype=float)

        fig, ax = plt.subplots(figsize=(10, 5))

        _plot_line(ax, xs, exp_fb, C_FB,  "Explicit — False Belief", lw=2.2)
        _plot_line(ax, xs, imp_fb, C_FB,  "Implicit — False Belief", lw=1.5)
        if not np.all(np.isnan(exp_tb)):
            _plot_line(ax, xs, exp_tb, C_TB, "Explicit — True Belief",  lw=2.2)
        if not np.all(np.isnan(imp_tb)):
            _plot_line(ax, xs, imp_tb, C_TB, "Implicit — True Belief",  lw=1.5)

        # Distinguish Explicit vs Implicit via line thickness in legend
        leg_elements = [
            mpatches.Patch(color=C_FB, label="False Belief"),
            mpatches.Patch(color=C_TB, label="True Belief"),
            plt.Line2D([0], [0], color="#555", lw=2.2, label="Explicit (solid)"),
            plt.Line2D([0], [0], color="#555", lw=1.5, label="Implicit (thin)"),
        ]
        _chance_line(ax)

        ax.set_xlabel("Words Seen During Training (M)", fontsize=12)
        ax.set_ylabel("P(Correct)", fontsize=12)
        ax.set_ylim(0.15, 1.0)
        ax.set_title(
            f"FBT by Knowledge Cue During Training — {LABELS[cond]}",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax.legend(handles=leg_elements + ax.get_legend_handles_labels()[0][-1:],
                  fontsize=9, frameon=True, facecolor="white", edgecolor="#cccccc",
                  loc="lower right")
        fig.tight_layout()
        _save(fig, out_dir / f"fbt_knowledge_cue_{cond}.png")


# ── Figure 5: BLiMP over training — all conditions ────────────────────────────

def plot_blimp_over_training(results_dir: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plotted = False

    for cond in CONDITIONS_ALL:
        ckpts = load_checkpoint_evals(results_dir, cond)
        if not ckpts:
            continue
        blimp_ckpts = [c for c in ckpts if "blimp" in c]
        if not blimp_ckpts:
            log.warning("No BLiMP data for %s", cond)
            continue

        xs = np.array([c["words_seen"] / 1e6 for c in blimp_ckpts])
        ys = np.array([c["blimp"]            for c in blimp_ckpts], dtype=float)
        _plot_line(ax, xs, ys, COLORS[cond], LABELS[cond],
                   smooth=(len(xs) >= SMOOTH_W))
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    _chance_line(ax)
    ax.set_xlabel("Words Seen During Training (M)", fontsize=12)
    ax.set_ylabel("BLiMP Accuracy", fontsize=12)
    ax.set_ylim(0.45, 0.80)
    ax.set_title("BLiMP Syntactic Accuracy Over Training — All Conditions",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="#cccccc")
    fig.tight_layout()
    _save(fig, out_dir / "blimp_over_training.png")


# ── Figure 6: EWoK over training — all conditions ─────────────────────────────

def plot_ewok_over_training(results_dir: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plotted = False

    for cond in CONDITIONS_ALL:
        ckpts = load_checkpoint_evals(results_dir, cond)
        if not ckpts:
            continue
        ewok_ckpts = [c for c in ckpts if "ewok" in c]
        if not ewok_ckpts:
            log.warning("No EWoK data for %s", cond)
            continue

        xs = np.array([c["words_seen"] / 1e6 for c in ewok_ckpts])
        ys = np.array([c["ewok"]             for c in ewok_ckpts], dtype=float)
        _plot_line(ax, xs, ys, COLORS[cond], LABELS[cond])
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    _chance_line(ax)
    ax.set_xlabel("Words Seen During Training (M)", fontsize=12)
    ax.set_ylabel("EWoK Accuracy", fontsize=12)
    ax.set_ylim(0.40, 0.65)
    ax.set_title("EWoK World-Knowledge Accuracy Over Training — All Conditions",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="#cccccc")
    fig.tight_layout()
    _save(fig, out_dir / "ewok_over_training.png")


# ── Figure 7: Fair cross-condition FBT comparison (equalized x-axis) ──────────

def plot_equalized_fbt_comparison(results_dir: Path, out_dir: Path) -> None:
    """
    All three conditions on the same x-axis, capped at the lowest training ceiling.
    Top panel: FBT Overall + True Belief + False Belief.
    Bottom panel: FB − TB gap (how much the model is biased toward False Belief).
    """
    all_data: dict[str, list[dict]] = {}
    for cond in CONDITIONS_ALL:
        ckpts = load_checkpoint_evals(results_dir, cond)
        if ckpts:
            all_data[cond] = ckpts

    if not all_data:
        log.warning("No checkpoint_evals found — skipping equalized comparison")
        return

    cap = min(max(c["words_seen"] for c in ckpts) for ckpts in all_data.values())
    log.info("Equalized cap: %.1fM words", cap / 1e6)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.06},
    )

    for cond, ckpts in all_data.items():
        filtered = [c for c in ckpts if c["words_seen"] <= cap * 1.02]
        if not filtered:
            continue
        xs  = np.array([c["words_seen"] / 1e6 for c in filtered])
        fbt = np.array([c["fbt_overall"]       for c in filtered], dtype=float)
        fb  = np.array([c["fbt_false_belief"]  for c in filtered], dtype=float)
        tb  = np.array([c["fbt_true_belief"]   for c in filtered], dtype=float)
        gap = fb - tb

        col = COLORS[cond]
        _plot_line(ax_top, xs, fbt, col, LABELS[cond], lw=2.2)
        _plot_line(ax_top, xs, fb,  col, None,          lw=1.5)
        _plot_line(ax_top, xs, tb,  col, None,          lw=1.5)
        _plot_line(ax_bot, xs, gap, col, LABELS[cond], lw=2.0)

    ax_top.axhline(CHANCE, color=CHANCE_COL, lw=1.0, ls="--", zorder=1)
    ax_top.set_ylabel("P(Correct)", fontsize=12)
    ax_top.set_ylim(0.15, 0.95)
    ax_top.set_title(
        f"FBT Accuracy — Matched Exposure ({cap/1e6:.1f}M words cap)",
        fontsize=13, fontweight="bold", pad=12,
    )

    cond_patches = [mpatches.Patch(color=COLORS[c], label=LABELS[c]) for c in all_data]
    style_lines  = [
        plt.Line2D([0], [0], color="#555", lw=2.2, label="Overall (bold)"),
        plt.Line2D([0], [0], color="#555", lw=1.5, label="False / True Belief (thin)"),
        plt.Line2D([0], [0], color=CHANCE_COL, lw=1.0, ls="--", label="Chance (0.50)"),
    ]
    ax_top.legend(handles=cond_patches + style_lines,
                  fontsize=9, frameon=True, facecolor="white",
                  edgecolor="#cccccc", loc="upper right", ncol=2)

    ax_bot.axhline(0, color=CHANCE_COL, lw=1.0, ls="--", zorder=1)
    ax_bot.set_xlabel("Words Seen During Training (M)", fontsize=12)
    ax_bot.set_ylabel("FB − TB Gap", fontsize=12)
    ax_bot.set_title("Belief Bias: False Belief Advantage over True Belief",
                     fontsize=11, pad=8)
    ax_bot.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.tight_layout()
    _save(fig, out_dir / "equalized_fbt_comparison.png")


# ── Figure 8: FBT final summary bar chart (all conditions) ────────────────────

def plot_fbt_summary_bars(results_dir: Path, out_dir: Path) -> None:
    """Bar chart of final-checkpoint FBT: Overall, False Belief, True Belief."""
    metrics   = ["Overall", "False Belief", "True Belief"]
    keys      = ["fbt_overall", "fbt_false_belief", "fbt_true_belief"]
    available = {}

    for cond in CONDITIONS_ALL:
        ckpts = load_checkpoint_evals(results_dir, cond)
        if not ckpts:
            continue
        last = ckpts[-1]
        available[cond] = {m: last.get(k, np.nan) for m, k in zip(metrics, keys)}

    if not available:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    n_cond  = len(available)
    bar_w   = 0.22
    x       = np.arange(len(metrics))

    for i, cond in enumerate(CONDITIONS_ALL):
        if cond not in available:
            continue
        vals   = [available[cond][m] for m in metrics]
        offset = (i - (n_cond - 1) / 2) * bar_w
        bars   = ax.bar(x + offset, vals, bar_w, color=COLORS[cond],
                        label=LABELS[cond], alpha=0.88,
                        edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="#333333")

    ax.axhline(CHANCE, color=CHANCE_COL, lw=1.2, ls="--", label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("P(Correct)", fontsize=12)
    ax.set_ylim(0.30, 1.0)
    ax.set_title("False Belief Test — Final Checkpoint Accuracy",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="#cccccc")
    fig.tight_layout()
    _save(fig, out_dir / "fbt_summary_bars.png")


# ── Figure 9: BLiMP + EWoK final summary ──────────────────────────────────────

def plot_blimp_ewok_summary(results_dir: Path, out_dir: Path) -> None:
    """Bar chart comparing final BLiMP and EWoK across all conditions."""
    available: dict[str, dict] = {}

    for cond in CONDITIONS_ALL:
        ckpts = load_checkpoint_evals(results_dir, cond)
        if not ckpts:
            continue
        blimp_ckpts = [c for c in ckpts if "blimp" in c]
        ewok_ckpts  = [c for c in ckpts if "ewok"  in c]
        blimp_final = blimp_ckpts[-1]["blimp"] if blimp_ckpts else np.nan
        ewok_final  = ewok_ckpts[-1]["ewok"]   if ewok_ckpts  else np.nan
        available[cond] = {"BLiMP": blimp_final, "EWoK": ewok_final}

    if not available:
        return

    benchmarks = ["BLiMP", "EWoK"]
    fig, ax    = plt.subplots(figsize=(7, 4.5))
    n_cond     = len(available)
    bar_w      = 0.22
    x          = np.arange(len(benchmarks))

    for i, cond in enumerate(CONDITIONS_ALL):
        if cond not in available:
            continue
        vals   = [available[cond].get(b, np.nan) for b in benchmarks]
        offset = (i - (n_cond - 1) / 2) * bar_w
        bars   = ax.bar(x + offset, vals, bar_w, color=COLORS[cond],
                        label=LABELS[cond], alpha=0.88,
                        edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="#333333")

    ax.axhline(CHANCE, color=CHANCE_COL, lw=1.2, ls="--", label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0.40, 0.80)
    ax.set_title("BLiMP & EWoK — Final Checkpoint Accuracy by Condition",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="#cccccc")
    fig.tight_layout()
    _save(fig, out_dir / "blimp_ewok_summary.png")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results_dir", type=Path,
                   default=Path(__file__).parent.parent / "results")
    p.add_argument("--out_dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = args.out_dir or args.results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    # ── Core thesis figures (7 total) ──────────────────────────────────────────
    plot_training_curves(args.results_dir, out_dir)          # Fig 1: epoch loss
    plot_loss_vs_words(args.results_dir, out_dir)            # Fig 2: step loss vs words
    plot_training_panel_combined(args.results_dir, out_dir)  # Fig 3/4: BLiMP+FBT per condition (flat & chunked)
    plot_blimp_over_training(args.results_dir, out_dir)      # Fig 5: BLiMP cross-condition
    plot_ewok_over_training(args.results_dir, out_dir)        # Fig 6: EWoK cross-condition
    plot_equalized_fbt_comparison(args.results_dir, out_dir) # Fig 7: fair FBT comparison + gap

    # ── Available for appendix / exploration (not called by default) ───────────
    # plot_fbt_knowledge_cue_over_training(args.results_dir, out_dir)
    # plot_fbt_summary_bars(args.results_dir, out_dir)
    # plot_blimp_ewok_summary(args.results_dir, out_dir)

    log.info("All done. Figures saved to %s", out_dir)


if __name__ == "__main__":
    main()
