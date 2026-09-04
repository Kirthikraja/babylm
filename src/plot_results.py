"""
src/plot_results.py

Generates publication-quality figures for the chunked vs flat GPT-2 comparison.

Figures produced (saved to results/figures/):
  1. training_curves.png  — train/eval loss vs words seen  (style: Warstadt et al. 2023)
  2. eval_summary.png     — BLiMP / ZORRO / EWoK aggregate bar chart
  3. blimp_categories.png — BLiMP accuracy per linguistic category
  4. zorro_paradigms.png  — ZORRO accuracy per syntactic paradigm
  5. ewok_domains.png     — EWoK accuracy per knowledge domain

Style: matches the uploaded reference figure — blue/orange palette,
       solid/dashed for conditions, shaded ±std bands where available.

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

# ── Style (matches reference image) ───────────────────────────────────────────
BLUE   = "#1f77b4"   # matplotlib C0
ORANGE = "#ff7f0e"   # matplotlib C1
ALPHA  = 0.15        # shaded band transparency
LW     = 2.0         # line width
CHANCE = 0.5         # BLiMP/ZORRO chance level

CONDITIONS = ["chunked", "flat"]
LABELS     = {"chunked": "EOS-Chunked", "flat": "Sliding-Window (Flat)"}
COLORS     = {"chunked": BLUE, "flat": ORANGE}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

# ── BLiMP category groupings (Warstadt et al. 2020) ──────────────────────────
BLIMP_CATEGORIES: dict[str, list[str]] = {
    "Anaphor\nAgreement": [
        "anaphor_gender_agreement", "anaphor_number_agreement",
    ],
    "Argument\nStructure": [
        "animate_subject_passive", "animate_subject_trans", "causative",
        "drop_argument", "inchoative", "intransitive",
        "passive_1", "passive_2", "transitive",
    ],
    "Binding": [
        "principle_A_c_command", "principle_A_case_1", "principle_A_case_2",
        "principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3",
        "principle_A_reconstruction",
    ],
    "Control /\nRaising": [
        "existential_there_object_raising", "existential_there_subject_raising",
        "expletive_it_object_raising", "tough_vs_raising_1", "tough_vs_raising_2",
    ],
    "Det-Noun\nAgreement": [
        "determiner_noun_agreement_1", "determiner_noun_agreement_2",
        "determiner_noun_agreement_irregular_1", "determiner_noun_agreement_irregular_2",
        "determiner_noun_agreement_with_adj_1", "determiner_noun_agreement_with_adj_2",
        "determiner_noun_agreement_with_adj_irregular_1",
        "determiner_noun_agreement_with_adj_irregular_2",
    ],
    "Ellipsis": [
        "ellipsis_n_bar_1", "ellipsis_n_bar_2",
    ],
    "Filler-Gap": [
        "wh_questions_object_gap", "wh_questions_object_gap_long",
        "wh_questions_subject_gap", "wh_questions_subject_gap_long_distance",
    ],
    "Irregular\nForms": [
        "irregular_past_participle_adjectives", "irregular_past_participle_verbs",
    ],
    "Island\nEffects": [
        "adjunct_island", "complex_NP_island",
        "coordinate_structure_constraint_complex_left_branch",
        "coordinate_structure_constraint_object_extraction",
        "left_branch_island_echo_question", "left_branch_island_simple_question",
        "sentential_subject_island", "wh_island",
    ],
    "NPI\nLicensing": [
        "matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2",
        "only_npi_licensor_present", "only_npi_scope",
        "sentential_negation_npi_licensor_present", "sentential_negation_npi_scope",
    ],
    "Quantifiers": [
        "existential_there_quantifiers_1", "existential_there_quantifiers_2",
        "superlative_quantifiers_1", "superlative_quantifiers_2",
    ],
    "Subject-Verb\nAgreement": [
        "distractor_agreement_relational_noun", "distractor_agreement_relative_clause",
        "irregular_plural_subject_verb_agreement_1",
        "irregular_plural_subject_verb_agreement_2",
        "regular_plural_subject_verb_agreement_1",
        "regular_plural_subject_verb_agreement_2",
    ],
}

# ── ZORRO paradigm groupings (Ravfogel et al. 2021) ──────────────────────────
# lm-eval uses prefix "zorro_" followed by paradigm name
ZORRO_PARADIGMS: dict[str, list[str]] = {
    "Simple\nAgreement": ["simple_agrmt_subject_verb"],
    "Across\nPrep Phrase": ["across_1_prepositional_phrase"],
    "Across\nRelative Clause": ["across_1_relative_clause"],
    "In\nQuestion": ["in_question_with_aux"],
    "Long VP": ["long_vp_coordination"],
    "Conjunction": ["conjunction_coordination"],
    "Across Object\nRel. Clause": ["across_obj_relative_clause"],
}


# ── Data loading helpers ───────────────────────────────────────────────────────

def load_training_curves(results_dir: Path, condition: str) -> dict | None:
    p = results_dir / condition / "training_curves.json"
    if not p.exists():
        log.warning("Not found: %s", p)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_lmeval(results_dir: Path, condition: str, benchmark: str) -> dict | None:
    """Load lm-evaluation-harness JSON output.

    lm-eval >= 0.4 may write either a .json file or a directory containing
    results_*.json.  We try both.
    """
    direct = results_dir / condition / f"{benchmark}.json"
    if direct.exists():
        data = json.loads(direct.read_text(encoding="utf-8"))
        return data.get("results", data)

    # directory variant
    dir_path = results_dir / condition / benchmark
    if dir_path.is_dir():
        for f in sorted(dir_path.glob("results_*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            return data.get("results", data)

    log.warning("Not found: %s (tried file and directory)", direct)
    return None


def load_ewok(results_dir: Path, condition: str) -> dict | None:
    p = results_dir / condition / "ewok.json"
    if not p.exists():
        log.warning("Not found: %s", p)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def task_acc(task_results: dict, task_name: str) -> float | None:
    """Extract accuracy from an lm-eval task result dict."""
    row = task_results.get(task_name) or task_results.get(f"blimp_{task_name}") or \
          task_results.get(f"zorro_{task_name}")
    if row is None:
        return None
    return row.get("acc,none") or row.get("acc") or row.get("accuracy")


def category_acc(task_results: dict, subtasks: list[str], prefix: str = "") -> float | None:
    """Mean accuracy over a list of subtask names."""
    vals = []
    for sub in subtasks:
        v = task_acc(task_results, sub) or task_acc(task_results, f"{prefix}{sub}")
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


# ── Figure 1: Training curves ─────────────────────────────────────────────────

def plot_training_curves(results_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    plotted = False

    for cond in CONDITIONS:
        data = load_training_curves(results_dir, cond)
        if data is None:
            continue
        plotted = True
        words = [w / 1_000_000 for w in data["words_seen"]]   # → millions
        col   = COLORS[cond]
        lbl   = LABELS[cond]

        # ── left: loss ───────────────────────────────────────────────────────
        axes[0].plot(words, data["train_loss"], color=col, lw=LW,
                     linestyle="-",  label=f"{lbl} – Train")
        axes[0].plot(words, data["eval_loss"],  color=col, lw=LW,
                     linestyle="--", label=f"{lbl} – Validation")

        # ── right: perplexity ─────────────────────────────────────────────
        train_ppl = [np.exp(l) for l in data["train_loss"]]
        eval_ppl  = [np.exp(l) for l in data["eval_loss"]]
        axes[1].plot(words, train_ppl, color=col, lw=LW,
                     linestyle="-",  label=f"{lbl} – Train")
        axes[1].plot(words, eval_ppl,  color=col, lw=LW,
                     linestyle="--", label=f"{lbl} – Validation")

    if not plotted:
        log.warning("No training curves found — skipping figure 1")
        plt.close(fig)
        return

    for ax, ylabel, title in zip(
        axes,
        ["Cross-Entropy Loss", "Perplexity"],
        ["Training & Validation Loss", "Training & Validation Perplexity"],
    ):
        ax.set_xlabel("Words Seen (M)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.8)

    # Legend hint matching reference image style
    solid_patch = plt.Line2D([0], [0], color="gray", lw=LW, ls="-",  label="Train")
    dash_patch  = plt.Line2D([0], [0], color="gray", lw=LW, ls="--", label="Validation")
    fig.legend(handles=[solid_patch, dash_patch], loc="lower center",
               ncol=2, fontsize=10, framealpha=0.8, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    out = out_dir / "training_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Figure 2: Aggregate summary bar chart ─────────────────────────────────────

def plot_eval_summary(results_dir: Path, out_dir: Path) -> None:
    benchmarks = ["BLiMP", "ZORRO", "EWoK"]
    data: dict[str, dict[str, float]] = {b: {} for b in benchmarks}

    for cond in CONDITIONS:
        blimp = load_lmeval(results_dir, cond, "blimp")
        if blimp:
            vals = [v for k in blimp for v in [blimp[k].get("acc,none") or blimp[k].get("acc")]
                    if v is not None and "blimp" in k]
            if vals:
                data["BLiMP"][cond] = float(np.mean(vals))

        zorro = load_lmeval(results_dir, cond, "zorro")
        if zorro:
            vals = [v for k in zorro for v in [zorro[k].get("acc,none") or zorro[k].get("acc")]
                    if v is not None and "zorro" in k]
            if vals:
                data["ZORRO"][cond] = float(np.mean(vals))

        ewok = load_ewok(results_dir, cond)
        if ewok:
            data["EWoK"][cond] = ewok["overall_accuracy"]

    # Check we have at least something
    if all(not d for d in data.values()):
        log.warning("No evaluation results found — skipping figure 2")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    n_benchmarks = len(benchmarks)
    n_conditions = len(CONDITIONS)
    bar_w = 0.32
    x = np.arange(n_benchmarks)

    for i, cond in enumerate(CONDITIONS):
        vals  = [data[b].get(cond, float("nan")) for b in benchmarks]
        offset = (i - (n_conditions - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, color=COLORS[cond],
                      label=LABELS[cond], alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(CHANCE, color="gray", lw=1, ls=":", label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Evaluation Summary: EOS-Chunked vs Sliding-Window", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.8)

    fig.tight_layout()
    out = out_dir / "eval_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Figure 3: BLiMP per-category ──────────────────────────────────────────────

def plot_blimp_categories(results_dir: Path, out_dir: Path) -> None:
    results: dict[str, dict[str, float | None]] = {}
    for cond in CONDITIONS:
        raw = load_lmeval(results_dir, cond, "blimp")
        if raw is None:
            continue
        results[cond] = {}
        for cat, subtasks in BLIMP_CATEGORIES.items():
            results[cond][cat] = category_acc(raw, subtasks, prefix="blimp_")

    if not results:
        log.warning("No BLiMP results — skipping figure 3")
        return

    cats = list(BLIMP_CATEGORIES.keys())
    fig, ax = plt.subplots(figsize=(14, 5))
    bar_w = 0.35
    x = np.arange(len(cats))

    for i, cond in enumerate(CONDITIONS):
        if cond not in results:
            continue
        vals = [results[cond].get(c) or float("nan") for c in cats]
        offset = (i - (len(CONDITIONS) - 1) / 2) * bar_w
        ax.bar(x + offset, vals, bar_w, color=COLORS[cond],
               label=LABELS[cond], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.axhline(CHANCE, color="gray", lw=1, ls=":", label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8.5)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("BLiMP: Accuracy by Linguistic Category", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.8)

    fig.tight_layout()
    out = out_dir / "blimp_categories.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Figure 4: ZORRO per-paradigm ──────────────────────────────────────────────

def plot_zorro_paradigms(results_dir: Path, out_dir: Path) -> None:
    results: dict[str, dict[str, float | None]] = {}
    for cond in CONDITIONS:
        raw = load_lmeval(results_dir, cond, "zorro")
        if raw is None:
            continue
        results[cond] = {}
        # Also accept any zorro_* key directly (paradigm = key without prefix)
        all_keys = list(raw.keys())
        zorro_keys = [k for k in all_keys if "zorro" in k.lower()]

        if ZORRO_PARADIGMS and any(
            category_acc(raw, sub, prefix="zorro_") is not None
            for sub in next(iter(ZORRO_PARADIGMS.values()))
        ):
            # Use defined paradigm groupings
            for par, subtasks in ZORRO_PARADIGMS.items():
                results[cond][par] = category_acc(raw, subtasks, prefix="zorro_")
        else:
            # Fall back: one bar per zorro task
            for k in zorro_keys:
                v = raw[k].get("acc,none") or raw[k].get("acc")
                short = k.replace("zorro_", "").replace("_", "\n")
                results[cond][short] = v

    if not results:
        log.warning("No ZORRO results — skipping figure 4")
        return

    # Use keys from whichever condition has results
    paradigms = list(next(iter(results.values())).keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_w = 0.35
    x = np.arange(len(paradigms))

    for i, cond in enumerate(CONDITIONS):
        if cond not in results:
            continue
        vals = [results[cond].get(p) or float("nan") for p in paradigms]
        offset = (i - (len(CONDITIONS) - 1) / 2) * bar_w
        ax.bar(x + offset, vals, bar_w, color=COLORS[cond],
               label=LABELS[cond], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.axhline(CHANCE, color="gray", lw=1, ls=":", label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(paradigms, fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("ZORRO: Accuracy by Syntactic Paradigm", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.8)

    fig.tight_layout()
    out = out_dir / "zorro_paradigms.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Figure 5: EWoK per-domain ─────────────────────────────────────────────────

def plot_ewok_domains(results_dir: Path, out_dir: Path) -> None:
    results: dict[str, dict[str, float]] = {}
    for cond in CONDITIONS:
        data = load_ewok(results_dir, cond)
        if data is None:
            continue
        results[cond] = data.get("by_domain", {})

    if not results:
        log.warning("No EWoK results — skipping figure 5")
        return

    # Union of domains across both conditions
    all_domains = sorted({d for r in results.values() for d in r})
    if not all_domains:
        log.warning("EWoK results contain no domain breakdown — skipping figure 5")
        return

    # Clean domain label formatting
    clean = {d: d.replace("_", " ").title() for d in all_domains}
    labels = [clean[d] for d in all_domains]

    fig, ax = plt.subplots(figsize=(12, 5))
    bar_w = 0.35
    x = np.arange(len(all_domains))

    for i, cond in enumerate(CONDITIONS):
        if cond not in results:
            continue
        vals = [results[cond].get(d, float("nan")) for d in all_domains]
        offset = (i - (len(CONDITIONS) - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, color=COLORS[cond],
                      label=LABELS[cond], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.axhline(CHANCE, color="gray", lw=1, ls=":", label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("EWoK: World Knowledge Accuracy by Domain", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.8)

    fig.tight_layout()
    out = out_dir / "ewok_domains.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results_dir", type=Path,
                   default=Path(__file__).parent.parent / "results")
    p.add_argument("--out_dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    plot_training_curves(args.results_dir, out_dir)
    plot_eval_summary(args.results_dir, out_dir)
    plot_blimp_categories(args.results_dir, out_dir)
    plot_zorro_paradigms(args.results_dir, out_dir)
    plot_ewok_domains(args.results_dir, out_dir)

    log.info("Done. Figures saved to %s", out_dir)


if __name__ == "__main__":
    main()
