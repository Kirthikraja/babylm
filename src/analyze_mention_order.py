"""
src/analyze_mention_order.py

Tests whether EOS-chunking amplifies primacy bias in the False Belief Test
by comparing how chunked vs flat models respond to mention-order manipulations.

Key metric: P(model chooses start location) — independent of condition.
  - Primacy effect: P(start choice | first_mention=Start) > P(start choice | first_mention=End)
  - Recency effect: P(start choice | recent_mention=Start) > P(start choice | recent_mention=End)

If EOS-chunking anchors models to sentence-level first mentions, chunked should show
a larger primacy effect than flat. This would confirm the EOS-boundary-as-anchor claim.

Requires: results/{chunked,flat}/fbt_all.json  (from eval_fbt.py --all_items)

Usage:
    python src/analyze_mention_order.py
    python src/analyze_mention_order.py --results_dir results --out_dir results/figures
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
import numpy as np
from scipy.stats import binomtest

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO, stream=sys.stdout,
)
log = logging.getLogger(__name__)

BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
LW     = 2.0
CONDITIONS = ["chunked", "flat"]
LABELS     = {"chunked": "EOS-Chunked", "flat": "Sliding-Window (Flat)"}
COLORS     = {"chunked": BLUE, "flat": ORANGE}


def load_items(results_dir: Path, condition: str) -> list[dict]:
    p = results_dir / condition / "fbt_all.json"
    if not p.exists():
        log.warning("Not found: %s — run: eval_fbt.py --all_items", p)
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("items", [])


def p_start(items: list[dict]) -> float:
    """Fraction of items where model chose the start location."""
    if not items:
        return float("nan")
    return sum(1 for i in items if i["model_choice"] == "start") / len(items)


def binom_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson confidence interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    result = binomtest(k, n)
    ci = result.proportion_ci(confidence_level=confidence, method="wilson")
    return (ci.low, ci.high)


def analyze(items: list[dict], label: str) -> dict:
    """Compute primacy/recency effects and significance tests."""
    log.info("\n── %s (%d items) ──", label, len(items))

    cells: dict[str, list[dict]] = {}
    for fm in ("Start", "End"):
        for rm in ("Start", "End"):
            key = f"first={fm}, recent={rm}"
            cells[key] = [i for i in items
                          if i["first_mention"] == fm and i["recent_mention"] == rm]

    results = {}
    for key, sub in cells.items():
        ps = p_start(sub)
        n  = len(sub)
        k  = sum(1 for i in sub if i["model_choice"] == "start")
        lo, hi = binom_ci(k, n)
        log.info("  %-32s  P(start)=%.3f  n=%d  95%%CI=[%.3f, %.3f]", key, ps, n, lo, hi)
        results[key] = {"p_start": ps, "n": n, "k": k, "ci_lo": lo, "ci_hi": hi}

    # Primacy effect: P(start | first=Start) - P(start | first=End)
    p_first_start = p_start([i for i in items if i["first_mention"] == "Start"])
    p_first_end   = p_start([i for i in items if i["first_mention"] == "End"])
    primacy_effect = p_first_start - p_first_end

    # Recency effect: P(start | recent=Start) - P(start | recent=End)
    p_recent_start = p_start([i for i in items if i["recent_mention"] == "Start"])
    p_recent_end   = p_start([i for i in items if i["recent_mention"] == "End"])
    recency_effect = p_recent_start - p_recent_end

    log.info("  Primacy effect  (first=S vs first=E):  %.3f  (%.3f - %.3f)",
             primacy_effect, p_first_start, p_first_end)
    log.info("  Recency effect  (recent=S vs recent=E): %.3f  (%.3f - %.3f)",
             recency_effect, p_recent_start, p_recent_end)

    # Significance: is first_mention=Start significantly biased toward start?
    n_fs = len([i for i in items if i["first_mention"] == "Start"])
    k_fs = sum(1 for i in items if i["first_mention"] == "Start" and i["model_choice"] == "start")
    bt = binomtest(k_fs, n_fs, p=0.5, alternative="two-sided")
    log.info("  Binomial test (first=Start, H0: P(start)=0.5): p=%.4f", bt.pvalue)

    return {
        "cells": results,
        "primacy_effect": primacy_effect,
        "recency_effect": recency_effect,
        "p_first_start": p_first_start,
        "p_first_end": p_first_end,
        "p_recent_start": p_recent_start,
        "p_recent_end": p_recent_end,
        "binom_p_first_start": bt.pvalue,
    }


def plot_mention_order(all_results: dict[str, dict], out_dir: Path) -> None:
    """2×2 grouped bar chart: P(start choice) by first_mention × recent_mention."""
    cell_labels = [
        "first=Start\nrecent=Start",
        "first=Start\nrecent=End",
        "first=End\nrecent=Start",
        "first=End\nrecent=End",
    ]
    cell_keys = [
        "first=Start, recent=Start",
        "first=Start, recent=End",
        "first=End, recent=Start",
        "first=End, recent=End",
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cell_keys))
    bar_w = 0.35

    for i, cond in enumerate(CONDITIONS):
        if cond not in all_results:
            continue
        vals  = [all_results[cond]["cells"].get(k, {}).get("p_start", float("nan"))
                 for k in cell_keys]
        ci_lo = [all_results[cond]["cells"].get(k, {}).get("ci_lo", float("nan"))
                 for k in cell_keys]
        ci_hi = [all_results[cond]["cells"].get(k, {}).get("ci_hi", float("nan"))
                 for k in cell_keys]
        yerr_lo = [v - lo for v, lo in zip(vals, ci_lo)]
        yerr_hi = [hi - v  for v, hi in zip(vals, ci_hi)]

        offset = (i - (len(CONDITIONS) - 1) / 2) * bar_w
        ax.bar(x + offset, vals, bar_w, color=COLORS[cond], label=LABELS[cond],
               alpha=0.85, edgecolor="white", linewidth=0.5,
               yerr=[yerr_lo, yerr_hi], capsize=4, error_kw={"elinewidth": 1.2})

    ax.axhline(0.5, color="gray", lw=1, ls=":", label="No bias (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(cell_labels, fontsize=10)
    ax.set_ylabel("P(model chooses start location)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Mention-Order Bias: Does EOS-Chunking Amplify Primacy?", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.8)

    # Shade the two "first=Start" cells to highlight primacy
    for xi in [0, 1]:
        ax.axvspan(xi - 0.45, xi + 0.45, alpha=0.05, color="blue", zorder=0)

    fig.tight_layout()
    out = out_dir / "fbt_mention_order.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)


def plot_primacy_recency_effects(all_results: dict[str, dict], out_dir: Path) -> None:
    """Side-by-side bar comparing primacy and recency effect sizes per condition."""
    effects = ["Primacy Effect\n(first=Start − first=End)", "Recency Effect\n(recent=Start − recent=End)"]
    keys    = ["primacy_effect", "recency_effect"]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(effects))
    bar_w = 0.35

    for i, cond in enumerate(CONDITIONS):
        if cond not in all_results:
            continue
        vals = [all_results[cond].get(k, float("nan")) for k in keys]
        offset = (i - (len(CONDITIONS) - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, color=COLORS[cond],
                      label=LABELS[cond], alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + 0.005 if v >= 0 else v - 0.018,
                        f"{v:+.3f}", ha="center", va="bottom", fontsize=9,
                        color=COLORS[cond], fontweight="bold")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(effects, fontsize=11)
    ax.set_ylabel("Effect size on P(start choice)", fontsize=12)
    ax.set_title("Primacy vs Recency Effect: Chunked vs Flat", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.8)

    fig.tight_layout()
    out = out_dir / "fbt_primacy_recency.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)


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

    all_results: dict[str, dict] = {}
    for cond in CONDITIONS:
        items = load_items(args.results_dir, cond)
        if not items:
            continue
        all_results[cond] = analyze(items, LABELS[cond])

    if not all_results:
        log.error("No fbt_all.json found. Run eval_fbt.py --all_items first.")
        sys.exit(1)

    # Summary comparison
    log.info("\n── Effect Size Comparison ──")
    for cond, res in all_results.items():
        log.info("  %-25s  primacy=%.3f  recency=%.3f",
                 LABELS[cond], res["primacy_effect"], res["recency_effect"])

    plot_mention_order(all_results, out_dir)
    plot_primacy_recency_effects(all_results, out_dir)
    log.info("Done. Figures saved to %s", out_dir)


if __name__ == "__main__":
    main()
