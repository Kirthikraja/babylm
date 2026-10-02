"""
src/compare_bias_severity.py

Reports the False-Belief minus True-Belief gap (bias severity) for each condition
at matched training exposure checkpoints. Produces the numbers needed for the
thesis comparison table and flags which condition is least biased.

Usage:
    python src/compare_bias_severity.py
    python src/compare_bias_severity.py --n_snapshots 4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO, stream=sys.stdout,
)
log = logging.getLogger(__name__)

CONDITIONS_ALL = ["chunked", "flat", "balanced"]
LABELS = {
    "chunked":  "EOS-Chunked",
    "flat":     "Sliding-Window (Flat)",
    "balanced": "DENSITY-Balanced",
}


def load_checkpoint_evals(results_dir: Path, condition: str) -> list[dict] | None:
    p = results_dir / condition / "checkpoint_evals.json"
    if not p.exists():
        log.warning("Not found: %s", p)
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    ckpts = [c for c in data.get("checkpoints", []) if c.get("words_seen") is not None]
    return sorted(ckpts, key=lambda c: c["words_seen"])


def nearest_checkpoint(ckpts: list[dict], target_words: float) -> dict | None:
    """Return the checkpoint with words_seen closest to target_words."""
    if not ckpts:
        return None
    return min(ckpts, key=lambda c: abs(c["words_seen"] - target_words))


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results_dir", type=Path,
                   default=Path(__file__).parent.parent / "results")
    p.add_argument("--n_snapshots", type=int, default=3,
                   help="Number of matched-exposure snapshots to report")
    args = p.parse_args()

    # ── Load all checkpoint evals ─────────────────────────────────────────────
    all_data: dict[str, list[dict]] = {}
    for cond in CONDITIONS_ALL:
        ckpts = load_checkpoint_evals(args.results_dir, cond)
        if ckpts:
            all_data[cond] = ckpts
        else:
            log.warning("Skipping %s — no checkpoint_evals.json", cond)

    if not all_data:
        log.error("No checkpoint eval data found in %s", args.results_dir)
        sys.exit(1)

    # ── Determine matched exposure range ─────────────────────────────────────
    max_per_cond = {cond: max(c["words_seen"] for c in ckpts)
                    for cond, ckpts in all_data.items()}
    cap_words = min(max_per_cond.values())
    cap_cond  = min(max_per_cond, key=max_per_cond.get)

    print("=" * 70)
    print("  BIAS SEVERITY COMPARISON — MATCHED EXPOSURE")
    print("=" * 70)
    print(f"  Match cap    : {cap_words/1e6:.2f}M words  (ceiling of '{cap_cond}')")
    print(f"  Conditions   : {', '.join(all_data.keys())}")
    print(f"  Snapshots    : {args.n_snapshots} evenly spaced between 0 and cap")
    print()

    # ── Build evenly-spaced snapshot targets ─────────────────────────────────
    # Include the final cap point plus n_snapshots-1 earlier ones
    targets = [cap_words * (i + 1) / args.n_snapshots
               for i in range(args.n_snapshots)]

    # ── Per-snapshot table ────────────────────────────────────────────────────
    for target in targets:
        print(f"  ── At ~{target/1e6:.2f}M words ──────────────────────────────────────")
        row: dict[str, dict] = {}
        for cond, ckpts in all_data.items():
            ck = nearest_checkpoint([c for c in ckpts if c["words_seen"] <= cap_words * 1.05],
                                    target)
            if ck is None:
                continue
            fb   = ck["fbt_false_belief"]
            tb   = ck["fbt_true_belief"]
            gap  = fb - tb
            row[cond] = {"fb": fb, "tb": tb, "gap": gap,
                         "overall": ck["fbt_overall"],
                         "actual_words": ck["words_seen"]}

        if not row:
            print("    (no data)")
            continue

        for cond, vals in row.items():
            lbl = LABELS.get(cond, cond)
            print(f"    {lbl:<30}  "
                  f"FB={vals['fb']:.3f}  TB={vals['tb']:.3f}  "
                  f"Gap={vals['gap']:+.3f}  "
                  f"(actual: {vals['actual_words']/1e6:.2f}M words)")

        # Plain-language sentence
        gaps = {c: v["gap"] for c, v in row.items()}
        if len(gaps) > 1:
            max_cond = max(gaps, key=gaps.get)
            min_cond = min(gaps, key=gaps.get)
            print(f"\n    At matched exposure of {target/1e6:.1f}M words:")
            for cond, g in sorted(gaps.items(), key=lambda x: -x[1]):
                print(f"      {LABELS.get(cond, cond)}: gap = {g:+.3f}")
            if gaps[max_cond] > 0 and gaps[min_cond] >= 0:
                ratio = gaps[max_cond] / max(gaps[min_cond], 1e-6)
                print(f"    → {LABELS.get(max_cond, max_cond)} shows {ratio:.1f}× more bias than "
                      f"{LABELS.get(min_cond, min_cond)}")
        print()

    # ── Final summary at cap ──────────────────────────────────────────────────
    print("=" * 70)
    print("  FINAL MATCHED CHECKPOINT SUMMARY")
    print("=" * 70)
    final_gaps: dict[str, float] = {}
    for cond, ckpts in all_data.items():
        ck = nearest_checkpoint([c for c in ckpts if c["words_seen"] <= cap_words * 1.05],
                                cap_words)
        if ck is None:
            continue
        fb  = ck["fbt_false_belief"]
        tb  = ck["fbt_true_belief"]
        gap = fb - tb
        final_gaps[cond] = gap
        print(f"  {LABELS.get(cond, cond):<30}  "
              f"FB={fb:.3f}  TB={tb:.3f}  Gap={gap:+.3f}  "
              f"Overall={ck['fbt_overall']:.3f}")

    if final_gaps:
        print()
        least_biased = min(final_gaps, key=lambda c: abs(final_gaps[c]))
        most_biased  = max(final_gaps, key=lambda c: abs(final_gaps[c]))
        print(f"  Least biased  : {LABELS.get(least_biased, least_biased)} "
              f"(gap = {final_gaps[least_biased]:+.3f})")
        print(f"  Most biased   : {LABELS.get(most_biased, most_biased)} "
              f"(gap = {final_gaps[most_biased]:+.3f})")
        print()
        print(f"  → '{LABELS.get(least_biased, least_biased)}' has the smallest FB-TB gap.")
        print(f"    A smaller gap means the model is less systematically biased toward")
        print(f"    start-location responses, which means think-vector extraction will")
        print(f"    face a cleaner signal — fewer baseline-bias confounds.")
        print()
        print(f"  NOTE: 'least biased' is not the same as 'best substrate for think")
        print(f"  vectors'. The flat model may still be the best choice if it shows")
        print(f"  a STABLE and CLEAR gap (even if larger), because:")
        print(f"    1. It has the most checkpoints to trace the vector's emergence.")
        print(f"    2. A reliable gap is more useful than a noisy near-zero gap.")
        print(f"  Combine this table with the equalized_three_way_fbt_comparison.png")
        print(f"  to decide based on both gap size AND trajectory stability.")


if __name__ == "__main__":
    main()
