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

GREEN = "#2ca02c"   # matplotlib C2

CONDITIONS       = ["chunked", "flat"]           # training curves only
CONDITIONS_ALL   = ["chunked", "flat", "balanced"]
LABELS     = {"chunked": "EOS-Chunked", "flat": "Sliding-Window (Flat)", "balanced": "DENSITY-Balanced"}
COLORS     = {"chunked": BLUE, "flat": ORANGE, "balanced": GREEN}

# Semantic colors for False Belief / True Belief trajectories (matches reference figs)
C_FB      = "#E8735A"    # coral  — False Belief line
C_TB      = "#47A9A9"    # teal   — True Belief line
C_OVERALL = "#555555"    # dark gray — FBT Overall in per-condition plots
MARKERS   = {"chunked": "o", "flat": "s", "balanced": "^"}
MARKER_SZ = 5

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

    Tries in order:
      1. <condition>/<benchmark>.json  (explicit output path)
      2. <condition>/<benchmark>_*.json  (timestamped, e.g. blimp_2026-09-04T....json)
      3. <condition>/<benchmark>/results_*.json  (directory variant)
    """
    direct = results_dir / condition / f"{benchmark}.json"
    if direct.exists():
        data = json.loads(direct.read_text(encoding="utf-8"))
        return data.get("results", data)

    # timestamped file written by lm-eval when --output_path is a directory
    candidates = sorted((results_dir / condition).glob(f"{benchmark}_*.json"))
    if candidates:
        data = json.loads(candidates[-1].read_text(encoding="utf-8"))
        log.info("Loaded %s from %s", benchmark, candidates[-1].name)
        return data.get("results", data)

    # directory variant
    dir_path = results_dir / condition / benchmark
    if dir_path.is_dir():
        for f in sorted(dir_path.glob("results_*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            return data.get("results", data)

    log.warning("Not found: %s (tried file, glob, and directory)", direct)
    return None


def load_ewok(results_dir: Path, condition: str) -> dict | None:
    p = results_dir / condition / "ewok.json"
    if not p.exists():
        log.warning("Not found: %s", p)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_fbt(results_dir: Path, condition: str) -> dict | None:
    for name in ("fbt_all.json", "fbt.json"):
        p = results_dir / condition / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    log.warning("No FBT result found for %s", condition)
    return None


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


# ── Figure 1: Training curves (loss vs epoch, all 3 conditions) ──────────────

def plot_training_curves(results_dir: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False

    for cond in CONDITIONS_ALL:
        data = load_training_curves(results_dir, cond)
        if data is None:
            continue
        epochs = data.get("epoch_num")
        train_loss = data.get("epoch_train_loss")
        eval_loss  = data.get("epoch_eval_loss")
        if not epochs:
            log.warning("No epoch-level data for %s — skipping", cond)
            continue
        plotted = True
        col = COLORS[cond]
        lbl = LABELS[cond]

        ax.plot(epochs, train_loss, color=col, lw=LW, linestyle="-",
                marker="o", markersize=5, label=f"{lbl} – Train")
        ax.plot(epochs, eval_loss,  color=col, lw=LW, linestyle="--",
                marker="s", markersize=4, label=f"{lbl} – Validation")

        # Mark early stopping point with a vertical dotted line
        stopped_at = data.get("early_stopped_at")
        if stopped_at is not None:
            ax.axvline(x=stopped_at, color=col, lw=1.2, linestyle=":",
                       alpha=0.7)
            ax.annotate(f"stopped\n(ep {stopped_at})",
                        xy=(stopped_at, min(eval_loss)),
                        xytext=(stopped_at + 0.3, min(eval_loss) + 0.05),
                        fontsize=8, color=col, va="bottom")

    if not plotted:
        log.warning("No epoch-level training curves found — skipping figure 1")
        plt.close(fig)
        return

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Training & Validation Loss by Epoch", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Chance line not applicable for loss; add a clean legend instead
    solid_patch = plt.Line2D([0], [0], color="gray", lw=LW, ls="-",  label="Train")
    dash_patch  = plt.Line2D([0], [0], color="gray", lw=LW, ls="--", label="Validation")
    dot_patch   = plt.Line2D([0], [0], color="gray", lw=1.2, ls=":",  label="Early stop")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [solid_patch, dash_patch, dot_patch],
              labels  + ["Train", "Validation", "Early stop"],
              fontsize=9, framealpha=0.8, loc="upper right")

    fig.tight_layout()
    out = out_dir / "training_curves.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Figure 1b: Cross-condition loss vs words seen ────────────────────────────

def plot_loss_vs_words(results_dir: Path, out_dir: Path) -> None:
    """Overlay training loss for all three conditions on a shared words-seen axis."""
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False

    for cond in CONDITIONS_ALL:
        data = load_training_curves(results_dir, cond)
        if data is None:
            continue

        words = data.get("words_seen")
        # Try multiple common key names for step-level loss
        loss = (data.get("loss") or data.get("train_loss") or
                data.get("step_loss") or data.get("step_train_loss"))

        if not words or not loss or len(words) != len(loss):
            log.warning("No step-level loss data for %s (tried loss/train_loss/step_loss)", cond)
            continue

        xs = np.array(words) / 1e6
        ys = np.array(loss, dtype=float)

        # Smooth with a rolling mean (window=5) for readability
        if len(ys) > 10:
            kernel = np.ones(5) / 5
            ys_smooth = np.convolve(ys, kernel, mode="same")
            ax.plot(xs, ys,        color=COLORS[cond], lw=0.5, alpha=0.25, zorder=2)
            ax.plot(xs, ys_smooth, color=COLORS[cond], lw=LW,  alpha=0.90,
                    label=LABELS[cond], zorder=3)
        else:
            ax.plot(xs, ys, color=COLORS[cond], lw=LW, marker="o",
                    markersize=4, label=LABELS[cond], zorder=3)
        plotted = True

    if not plotted:
        log.warning("No step-level loss vs words data — skipping loss_vs_words figure")
        plt.close(fig)
        return

    ax.set_xlabel("Words Seen During Training (M)", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Training Loss vs Words Seen — All Conditions", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.tight_layout()
    out = out_dir / "loss_vs_words.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Figure 2: BLiMP + EWoK summary (all 3 conditions) ────────────────────────

def plot_eval_summary(results_dir: Path, out_dir: Path) -> None:
    benchmarks = ["BLiMP", "EWoK"]
    data: dict[str, dict[str, float]] = {b: {} for b in benchmarks}

    for cond in CONDITIONS_ALL:
        blimp = load_lmeval(results_dir, cond, "blimp")
        if blimp:
            vals = [v for k in blimp for v in [blimp[k].get("acc,none") or blimp[k].get("acc")]
                    if v is not None and "blimp" in k]
            if vals:
                data["BLiMP"][cond] = float(np.mean(vals))

        ewok = load_ewok(results_dir, cond)
        if ewok:
            data["EWoK"][cond] = ewok.get("overall_accuracy") or ewok.get("accuracy", float("nan"))

    if all(not d for d in data.values()):
        log.warning("No evaluation results found — skipping eval summary")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    n_benchmarks = len(benchmarks)
    n_conditions = len(CONDITIONS_ALL)
    bar_w = 0.25
    x = np.arange(n_benchmarks)

    for i, cond in enumerate(CONDITIONS_ALL):
        vals = [data[b].get(cond, float("nan")) for b in benchmarks]
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
    ax.set_ylim(0.4, 0.75)
    ax.set_title("BLiMP & EWoK Accuracy by Training Condition", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.8)

    fig.tight_layout()
    out = out_dir / "eval_summary.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Figure NEW: FBT False Belief vs True Belief ───────────────────────────────

def plot_fbt_belief_condition(results_dir: Path, out_dir: Path) -> None:
    fb_accs: dict[str, float] = {}
    tb_accs: dict[str, float] = {}

    for cond in CONDITIONS_ALL:
        data = load_fbt(results_dir, cond)
        if data is None:
            continue
        by_cond = data.get("by_condition", {})
        fb_accs[cond] = by_cond.get("False Belief", float("nan"))
        tb_accs[cond] = by_cond.get("True Belief",  float("nan"))

    if not fb_accs:
        log.warning("No FBT results — skipping FBT belief condition figure")
        return

    belief_types = ["False Belief", "True Belief"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_w = 0.25
    x = np.arange(len(belief_types))
    n_conditions = len(CONDITIONS_ALL)

    for i, cond in enumerate(CONDITIONS_ALL):
        vals = [fb_accs.get(cond, float("nan")), tb_accs.get(cond, float("nan"))]
        offset = (i - (n_conditions - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, color=COLORS[cond],
                      label=LABELS[cond], alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(CHANCE, color="#aaaaaa", lw=1.2, ls="--", label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(belief_types, fontsize=13)
    ax.set_ylabel("P(Correct)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("False Belief Test: Accuracy by Belief Condition", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.tight_layout()
    out = out_dir / "fbt_belief_condition.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
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
    fig.savefig(out, bbox_inches="tight", facecolor="white")
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
    fig.savefig(out, bbox_inches="tight", facecolor="white")
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
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Figure 6: FBT condition × cue breakdown ──────────────────────────────────

def plot_fbt_breakdown(results_dir: Path, out_dir: Path) -> None:
    results: dict[str, dict[str, float]] = {}
    for cond in CONDITIONS_ALL:
        data = load_fbt(results_dir, cond)
        if data is None:
            continue
        results[cond] = data.get("by_condition_x_cue", {})

    if not results:
        log.warning("No FBT results — skipping FBT breakdown figure")
        return

    # Canonical order matching Tom's paper
    cell_order = [
        "False Belief × Explicit",
        "False Belief × Implicit",
        "True Belief × Explicit",
        "True Belief × Implicit",
    ]
    # Use whatever keys are actually present
    all_keys = sorted({k for r in results.values() for k in r})
    cells = [k for k in cell_order if k in all_keys] or all_keys

    short_labels = [c.replace(" × ", "\n×\n") for c in cells]

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_w = 0.35
    x = np.arange(len(cells))

    for i, cond in enumerate(CONDITIONS_ALL):
        if cond not in results:
            continue
        vals = [results[cond].get(c, float("nan")) for c in cells]
        offset = (i - (len(CONDITIONS_ALL) - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, color=COLORS[cond],
                      label=LABELS[cond], alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(CHANCE, color="#aaaaaa", lw=1.2, ls="--", label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel("P(Correct)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("False Belief Test: Accuracy by Condition & Knowledge Cue", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.tight_layout()
    out = out_dir / "fbt_breakdown.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Figure 7: BLiMP vs EWoK slope chart ──────────────────────────────────────

def plot_blimp_vs_ewok(results_dir: Path, out_dir: Path) -> None:
    scores: dict[str, dict[str, float]] = {}

    for cond in CONDITIONS:
        blimp = load_lmeval(results_dir, cond, "blimp")
        ewok  = load_ewok(results_dir, cond)
        if blimp is None or ewok is None:
            continue
        vals = [v for k in blimp for v in [blimp[k].get("acc,none") or blimp[k].get("acc")]
                if v is not None and "blimp" in k]
        if not vals:
            continue
        scores[cond] = {
            "BLiMP": float(np.mean(vals)),
            "EWoK":  ewok["overall_accuracy"],
        }

    if len(scores) < 2:
        log.warning("Not enough data for slope chart — skipping figure 6")
        return

    benchmarks = ["BLiMP", "EWoK"]
    x = [0, 1]

    fig, ax = plt.subplots(figsize=(6, 5))

    for cond in CONDITIONS:
        if cond not in scores:
            continue
        y = [scores[cond][b] for b in benchmarks]
        ax.plot(x, y, color=COLORS[cond], lw=LW + 0.5, linestyle="-",
                marker="o", markersize=8, label=LABELS[cond], zorder=3)
        # annotate end points
        for xi, yi, bm in zip(x, y, benchmarks):
            ha = "right" if xi == 0 else "left"
            offset = -0.04 if xi == 0 else 0.04
            ax.text(xi + offset, yi, f"{yi:.3f}", ha=ha, va="center",
                    fontsize=10, color=COLORS[cond], fontweight="bold")

    ax.axhline(CHANCE, color="gray", lw=1, ls=":", label="Chance (0.50)", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0.45, 0.75)
    ax.set_xlim(-0.3, 1.3)
    ax.set_title("Syntactic vs World-Knowledge Trade-off", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.8, loc="center")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)

    fig.tight_layout()
    out = out_dir / "blimp_vs_ewok_slope.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Figure 8: Accuracy over training (Figure-10 style) ───────────────────────

def load_checkpoint_evals(results_dir: Path, condition: str) -> dict | None:
    p = results_dir / condition / "checkpoint_evals.json"
    if not p.exists():
        log.warning("Not found: %s", p)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _plot_checkpoint_accuracy_single(
    results_dir: Path, out_dir: Path, cond: str
) -> None:
    """One figure per condition — ggplot2-style academic layout."""
    C_BLIMP = "#9467bd"

    data = load_checkpoint_evals(results_dir, cond)
    if data is None:
        return
    ckpts = data.get("checkpoints", [])
    if not ckpts:
        log.warning("No checkpoints in data for %s — skipping", cond)
        return

    has_words = any(c.get("words_seen") for c in ckpts)
    xs         = [c["words_seen"] / 1_000_000 for c in ckpts] if has_words else [c["step"] for c in ckpts]
    xlabel_str = "Words Seen (M)" if has_words else "Training Step"

    fbt       = [c["fbt_overall"]      for c in ckpts]
    fb        = [c["fbt_false_belief"] for c in ckpts]
    tb        = [c["fbt_true_belief"]  for c in ckpts]
    has_blimp = all("blimp" in c for c in ckpts)
    blimp     = [c["blimp"] for c in ckpts] if has_blimp else []

    lbl = LABELS[cond]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Shaded band between FB and TB shows gap at a glance
    ax.fill_between(xs, tb, fb, alpha=0.10, color="#888888", zorder=1)

    if has_blimp and blimp:
        ax.plot(xs, blimp, color=C_BLIMP, lw=LW, ls="-", marker="D",
                markersize=MARKER_SZ - 1, zorder=3, label="BLiMP")

    ax.plot(xs, fb,  color=C_FB,      lw=2.0, ls="-",  marker="o",
            markersize=MARKER_SZ,     zorder=4, label="False Belief")
    ax.plot(xs, tb,  color=C_TB,      lw=2.0, ls="-",  marker="o",
            markersize=MARKER_SZ,     zorder=4, label="True Belief")
    ax.plot(xs, fbt, color=C_OVERALL, lw=1.6, ls="--", marker="s",
            markersize=MARKER_SZ - 1, zorder=3, alpha=0.85, label="FBT Overall")

    ax.axhline(CHANCE, color="#aaaaaa", lw=1.2, ls="--", zorder=1, label="Chance (0.50)")

    ax.set_xlabel(xlabel_str, fontsize=12)
    ax.set_ylabel("P(Correct)", fontsize=12)
    ax.set_ylim(0.15, 0.90)
    ax.set_title(f"FBT Performance During Training — {lbl}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, frameon=True, facecolor="white",
              edgecolor="#cccccc", loc="lower right")

    fig.tight_layout()
    out = out_dir / f"checkpoint_accuracy_{cond}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", out)


def plot_checkpoint_accuracy(results_dir: Path, out_dir: Path) -> None:
    for cond in CONDITIONS_ALL:
        _plot_checkpoint_accuracy_single(results_dir, out_dir, cond)


# ── Figure 9: Equalized three-way FBT comparison ─────────────────────────────

def plot_equalized_fbt_comparison(
    results_dir: Path,
    out_dir: Path,
    cap_words: float | None = None,
) -> None:
    """Two-panel fair comparison: all three conditions on a matched x-axis.

    Top panel   : FBT Overall / True Belief / False Belief for all conditions,
                  x-axis capped at the lowest training ceiling.
    Bottom panel: FB − TB gap (bias severity) per condition.
    """
    all_data: dict[str, list[dict]] = {}
    for cond in CONDITIONS_ALL:
        d = load_checkpoint_evals(results_dir, cond)
        if d is not None:
            ckpts = [c for c in d.get("checkpoints", []) if c.get("words_seen") is not None]
            if ckpts:
                all_data[cond] = sorted(ckpts, key=lambda c: c["words_seen"])

    if not all_data:
        log.warning("No checkpoint_evals.json found — skipping equalized comparison")
        return

    if cap_words is None:
        cap_words = min(max(c["words_seen"] for c in ckpts) for ckpts in all_data.values())
        log.info("Equalized cap: %.2fM words (lowest ceiling across conditions)", cap_words / 1e6)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
    )

    for cond, ckpts in all_data.items():
        filtered = [c for c in ckpts if c["words_seen"] <= cap_words * 1.01]
        if not filtered:
            continue
        xs  = [c["words_seen"] / 1e6 for c in filtered]
        fbt = [c["fbt_overall"]      for c in filtered]
        fb  = [c["fbt_false_belief"] for c in filtered]
        tb  = [c["fbt_true_belief"]  for c in filtered]
        gap = [f - t for f, t in zip(fb, tb)]

        col = COLORS[cond]
        mk  = MARKERS.get(cond, "o")
        ax_top.plot(xs, fbt, color=col, lw=2.0, ls="-",  marker=mk,
                    markersize=MARKER_SZ, zorder=3)
        ax_top.plot(xs, tb,  color=col, lw=1.5, ls="--", marker=mk,
                    markersize=MARKER_SZ - 1, alpha=0.80, zorder=3)
        ax_top.plot(xs, fb,  color=col, lw=1.5, ls=":",  marker=mk,
                    markersize=MARKER_SZ - 1, alpha=0.80, zorder=3)
        ax_bot.plot(xs, gap, color=col, lw=2.0, ls="-",  marker=mk,
                    markersize=MARKER_SZ, zorder=3, label=LABELS[cond])

    ax_top.axhline(CHANCE, color="#aaaaaa", lw=1.2, ls="--", zorder=1)
    ax_top.set_ylabel("P(Correct)", fontsize=12)
    ax_top.set_ylim(0.15, 0.90)
    ax_top.set_title(
        f"FBT Accuracy — Matched Exposure (cap: {cap_words/1e6:.1f}M words)",
        fontsize=13, fontweight="bold",
    )

    # Two-part legend: condition colour + line style
    cond_patches = [mpatches.Patch(color=COLORS[c], label=LABELS[c]) for c in all_data]
    style_lines  = [
        plt.Line2D([0], [0], color="#555", lw=2.0, ls="-",  label="Overall"),
        plt.Line2D([0], [0], color="#555", lw=1.5, ls="--", label="True Belief"),
        plt.Line2D([0], [0], color="#555", lw=1.5, ls=":",  label="False Belief"),
        plt.Line2D([0], [0], color="#aaa", lw=1.2, ls="--", label="Chance (0.50)"),
    ]
    ax_top.legend(handles=cond_patches + style_lines,
                  fontsize=8, frameon=True, facecolor="white",
                  edgecolor="#cccccc", loc="upper right", ncol=2)

    ax_bot.axhline(0, color="#aaaaaa", lw=1.2, ls="--", zorder=1)
    ax_bot.set_xlabel("Words Seen (M)", fontsize=12)
    ax_bot.set_ylabel("FB − TB Gap", fontsize=12)
    ax_bot.set_title("Belief Bias Severity (False Belief − True Belief Accuracy)",
                     fontsize=11)
    ax_bot.legend(fontsize=9, frameon=True, facecolor="white",
                  edgecolor="#cccccc", loc="upper right")

    fig.tight_layout()
    out = out_dir / "equalized_three_way_fbt_comparison.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Figure 9b: Reference-style combined panel (BLiMP + FBT metrics) ──────────

def _smooth(xs: np.ndarray, ys: np.ndarray, window: int = 15):
    """Rolling mean. Trims edge artefacts introduced by zero-padding."""
    if len(ys) < window:
        return xs, ys
    kernel  = np.ones(window) / window
    ys_s    = np.convolve(ys, kernel, mode="same")
    half    = window // 2
    return xs[half:-half], ys_s[half:-half]


def plot_training_panel_combined(results_dir: Path, out_dir: Path) -> None:
    """
    One figure per condition — exactly matches the supervisor reference style:
      raw noisy trajectory (faded thin line) + smoothed bold line on top,
      four metrics: BLiMP (purple), FBT Overall (green), True Belief (blue),
      False Belief (pink).
    """
    C_BLIMP   = "#9467bd"   # purple
    C_FBT_OV  = "#5b8c2a"   # olive green — FBT Overall
    C_TB_LINE = "#5b9ec9"   # steel blue  — True Belief
    C_FB_LINE = "#d9826a"   # dusty coral — False Belief
    ALPHA_RAW = 0.22
    SMOOTH_W  = 15

    for cond in CONDITIONS_ALL:
        data = load_checkpoint_evals(results_dir, cond)
        if data is None:
            continue
        ckpts = sorted(
            [c for c in data.get("checkpoints", []) if c.get("words_seen") is not None],
            key=lambda c: c["words_seen"],
        )
        if not ckpts:
            continue

        xs        = np.array([c["words_seen"] / 1e6 for c in ckpts])
        y_fbt     = np.array([c["fbt_overall"]      for c in ckpts], dtype=float)
        y_tb      = np.array([c["fbt_true_belief"]  for c in ckpts], dtype=float)
        y_fb      = np.array([c["fbt_false_belief"] for c in ckpts], dtype=float)
        blimp_raw = np.array([c["blimp"] for c in ckpts if "blimp" in c], dtype=float)
        xs_blimp  = np.array([c["words_seen"] / 1e6 for c in ckpts if "blimp" in c])

        fig, ax = plt.subplots(figsize=(10, 5))

        def _plot(x, y, color, label):
            ax.plot(x, y, color=color, lw=0.8, alpha=ALPHA_RAW, zorder=2)
            if len(x) >= SMOOTH_W:
                xs_s, ys_s = _smooth(x, y, SMOOTH_W)
                ax.plot(xs_s, ys_s, color=color, lw=2.2, zorder=4, label=label)
            else:
                ax.plot(x, y, color=color, lw=2.2, zorder=4, label=label)

        if len(xs_blimp) > 0:
            _plot(xs_blimp, blimp_raw, C_BLIMP,   "BLiMP")
        _plot(xs, y_fbt, C_FBT_OV,  "FBT Overall")
        _plot(xs, y_tb,  C_TB_LINE, "True Belief")
        _plot(xs, y_fb,  C_FB_LINE, "False Belief")

        ax.axhline(CHANCE, color="#aaaaaa", lw=1.2, ls="--", zorder=1)
        ax.set_xlabel("Words Seen During Training (M)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(0.20, 1.0)
        ax.set_title(
            f"BLiMP & FBT Accuracy During Training — {LABELS[cond]}",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=10, frameon=True, facecolor="white",
                  edgecolor="#cccccc", loc="lower right")

        fig.tight_layout()
        out = out_dir / f"training_panel_{cond}.png"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log.info("Saved: %s", out)


# ── Figure 10: FBT Explicit vs Implicit over training ────────────────────────

def plot_fbt_knowledge_cue_over_training(results_dir: Path, out_dir: Path) -> None:
    """Per-condition: explicit-FB, implicit-FB, explicit-TB, implicit-TB over words seen."""
    for cond in CONDITIONS_ALL:
        data = load_checkpoint_evals(results_dir, cond)
        if data is None:
            continue
        ckpts = [c for c in data.get("checkpoints", [])
                 if c.get("words_seen") is not None
                 and c.get("fbt_explicit_false_belief") is not None]
        if not ckpts:
            log.warning("No knowledge_cue breakdown in checkpoint_evals for %s — skipping", cond)
            continue

        ckpts = sorted(ckpts, key=lambda c: c["words_seen"])
        xs = [c["words_seen"] / 1e6 for c in ckpts]

        exp_fb = [c["fbt_explicit_false_belief"] for c in ckpts]
        imp_fb = [c["fbt_implicit_false_belief"] for c in ckpts]
        exp_tb = [c.get("fbt_explicit_true_belief") for c in ckpts]
        imp_tb = [c.get("fbt_implicit_true_belief") for c in ckpts]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(xs, exp_fb, color=C_FB,      lw=2.0, ls="-",  marker="o",
                markersize=MARKER_SZ, zorder=4, label="Explicit — False Belief")
        ax.plot(xs, imp_fb, color=C_FB,      lw=2.0, ls="--", marker="s",
                markersize=MARKER_SZ, zorder=4, label="Implicit — False Belief",  alpha=0.85)

        if all(v is not None for v in exp_tb):
            ax.plot(xs, exp_tb, color=C_TB, lw=2.0, ls="-",  marker="o",
                    markersize=MARKER_SZ, zorder=4, label="Explicit — True Belief")
        if all(v is not None for v in imp_tb):
            ax.plot(xs, imp_tb, color=C_TB, lw=2.0, ls="--", marker="s",
                    markersize=MARKER_SZ, zorder=4, label="Implicit — True Belief", alpha=0.85)

        ax.axhline(CHANCE, color="#aaaaaa", lw=1.2, ls="--", zorder=1)
        ax.set_xlabel("Words Seen During Training (M)", fontsize=12)
        ax.set_ylabel("P(Correct)", fontsize=12)
        ax.set_ylim(0.15, 1.0)
        ax.set_title(
            f"FBT by Knowledge Cue During Training — {LABELS[cond]}",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="#cccccc",
                  loc="lower right")

        fig.tight_layout()
        out = out_dir / f"fbt_knowledge_cue_{cond}.png"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log.info("Saved: %s", out)


# ── Figure 11: BLiMP over training ───────────────────────────────────────────

def plot_blimp_over_training(results_dir: Path, out_dir: Path) -> None:
    """All three conditions: BLiMP accuracy vs words seen on the same axis."""
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False

    for cond in CONDITIONS_ALL:
        data = load_checkpoint_evals(results_dir, cond)
        if data is None:
            continue
        ckpts = [c for c in data.get("checkpoints", [])
                 if c.get("words_seen") is not None and c.get("blimp") is not None]
        if not ckpts:
            log.warning("No BLiMP data in checkpoint_evals for %s", cond)
            continue
        ckpts = sorted(ckpts, key=lambda c: c["words_seen"])
        xs = [c["words_seen"] / 1e6 for c in ckpts]
        ys = [c["blimp"] for c in ckpts]

        ax.plot(xs, ys, color=COLORS[cond], lw=LW, marker=MARKERS.get(cond, "o"),
                markersize=MARKER_SZ, label=LABELS[cond], zorder=3)
        plotted = True

    if not plotted:
        log.warning("No BLiMP-over-training data — skipping figure")
        plt.close(fig)
        return

    ax.axhline(CHANCE, color="#aaaaaa", lw=1.2, ls="--", zorder=1, label="Chance (0.50)")
    ax.set_xlabel("Words Seen During Training (M)", fontsize=12)
    ax.set_ylabel("BLiMP Accuracy", fontsize=12)
    ax.set_ylim(0.45, 1.0)
    ax.set_title("BLiMP Accuracy Over Training — All Conditions",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.tight_layout()
    out = out_dir / "blimp_over_training.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", out)


# ── Figure 12: EWoK over training ────────────────────────────────────────────

def plot_ewok_over_training(results_dir: Path, out_dir: Path) -> None:
    """All three conditions: EWoK accuracy vs words seen on the same axis."""
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False

    for cond in CONDITIONS_ALL:
        data = load_checkpoint_evals(results_dir, cond)
        if data is None:
            continue
        ckpts = [c for c in data.get("checkpoints", [])
                 if c.get("words_seen") is not None and c.get("ewok") is not None]
        if not ckpts:
            log.warning("No EWoK data in checkpoint_evals for %s", cond)
            continue
        ckpts = sorted(ckpts, key=lambda c: c["words_seen"])
        xs = [c["words_seen"] / 1e6 for c in ckpts]
        ys = [c["ewok"] for c in ckpts]

        ax.plot(xs, ys, color=COLORS[cond], lw=LW, marker=MARKERS.get(cond, "o"),
                markersize=MARKER_SZ, label=LABELS[cond], zorder=3)
        plotted = True

    if not plotted:
        log.warning("No EWoK-over-training data — skipping figure")
        plt.close(fig)
        return

    ax.axhline(CHANCE, color="#aaaaaa", lw=1.2, ls="--", zorder=1, label="Chance (0.50)")
    ax.set_xlabel("Words Seen During Training (M)", fontsize=12)
    ax.set_ylabel("EWoK Accuracy", fontsize=12)
    ax.set_ylim(0.40, 0.80)
    ax.set_title("EWoK World-Knowledge Accuracy Over Training — All Conditions",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.tight_layout()
    out = out_dir / "ewok_over_training.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
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

    plot_training_curves(args.results_dir, out_dir)              # Fig 1:  epoch loss curves
    plot_loss_vs_words(args.results_dir, out_dir)                # Fig 1b: loss vs words (cross-condition)
    plot_fbt_belief_condition(args.results_dir, out_dir)         # Fig 2:  FB vs TB accuracy (final)
    plot_fbt_breakdown(args.results_dir, out_dir)                # Fig 3:  2×2 condition × cue (final)
    plot_eval_summary(args.results_dir, out_dir)                 # Fig 4:  BLiMP + EWoK summary bar
    plot_checkpoint_accuracy(args.results_dir, out_dir)          # Fig 8:  FBT accuracy over training
    plot_equalized_fbt_comparison(args.results_dir, out_dir)     # Fig 9:  matched x-axis comparison
    plot_training_panel_combined(args.results_dir, out_dir)          # Fig 9b: reference-style BLiMP+FBT panel
    plot_fbt_knowledge_cue_over_training(args.results_dir, out_dir)  # Fig 10: explicit/implicit over training
    plot_blimp_over_training(args.results_dir, out_dir)          # Fig 11: BLiMP over training
    plot_ewok_over_training(args.results_dir, out_dir)           # Fig 12: EWoK over training

    log.info("Done. Figures saved to %s", out_dir)


if __name__ == "__main__":
    main()
