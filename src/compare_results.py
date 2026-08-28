"""
src/compare_results.py

Loads BLiMP, ZORRO, and EWoK result files for both conditions and prints
a side-by-side comparison table. Saves results/comparison.csv.

Usage:
    python src/compare_results.py
"""

from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def load_lm_eval_score(path: Path, task_prefix: str) -> dict[str, float]:
    """Parse lm-evaluation-harness output JSON and return {sub-task: accuracy}."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", {})
    scores = {}
    for task, metrics in results.items():
        if task.startswith(task_prefix):
            acc = metrics.get("acc,none") or metrics.get("acc") or metrics.get("accuracy")
            if acc is not None:
                scores[task] = float(acc)
    return scores


def load_ewok_score(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("overall_accuracy")


def main() -> None:
    base = Path(__file__).parent.parent
    results_dir = base / "results"
    conditions = ["chunked", "flat"]

    rows = []

    for cond in conditions:
        cond_dir = results_dir / cond

        # BLiMP — average across all sub-tasks
        blimp_scores = load_lm_eval_score(cond_dir / "blimp.json", "blimp_")
        blimp_avg = sum(blimp_scores.values()) / len(blimp_scores) if blimp_scores else None

        # ZORRO — average across all sub-tasks
        zorro_scores = load_lm_eval_score(cond_dir / "zorro.json", "zorro_")
        zorro_avg = sum(zorro_scores.values()) / len(zorro_scores) if zorro_scores else None

        # EWoK — overall accuracy
        ewok_acc = load_ewok_score(cond_dir / "ewok.json")

        rows.append({
            "condition": cond,
            "BLiMP_avg": round(blimp_avg * 100, 2) if blimp_avg is not None else "N/A",
            "ZORRO_avg": round(zorro_avg * 100, 2) if zorro_avg is not None else "N/A",
            "EWoK_acc":  round(ewok_acc  * 100, 2) if ewok_acc  is not None else "N/A",
            "n_blimp_tasks": len(blimp_scores),
            "n_zorro_tasks": len(zorro_scores),
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("CHUNKED vs FLAT — Evaluation Summary")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60 + "\n")

    out_path = results_dir / "comparison.csv"
    df.to_csv(out_path, index=False)
    log.info("Saved comparison table to %s", out_path)


if __name__ == "__main__":
    main()
