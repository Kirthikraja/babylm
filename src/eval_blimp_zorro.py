"""
src/eval_blimp_zorro.py

Evaluates a GPT-2 checkpoint on BLiMP and ZORRO using lm-evaluation-harness.

Writes results to results/<condition>/blimp_zorro.json

Usage:
    python src/eval_blimp_zorro.py --model_path models/chunked/final --condition chunked
    python src/eval_blimp_zorro.py --model_path models/flat/final    --condition flat
"""

from __future__ import annotations
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# All BLiMP sub-tasks available in lm-eval
BLIMP_TASKS = "blimp"      # runs the full BLiMP suite (67 sub-tasks)
ZORRO_TASKS = "zorro"      # syntactic generalization across 23 phenomena


def run_lm_eval(
    model_path: str,
    tasks: str,
    out_path: Path,
    device: str = "cuda",
    batch_size: int = 64,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=float16",
        "--tasks", tasks,
        "--device", device,
        "--batch_size", str(batch_size),
        "--output_path", str(out_path),
        "--log_samples",
    ]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        log.error("lm_eval failed for tasks=%s", tasks)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to HuggingFace model directory (e.g. models/chunked/final)")
    p.add_argument("--condition", choices=["chunked", "flat"], required=True)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--tasks", nargs="+", default=["blimp", "zorro"],
                   choices=["blimp", "zorro"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(__file__).parent.parent
    out_dir = args.output_dir or base / "results" / args.condition

    for task in args.tasks:
        out_path = out_dir / f"{task}.json"
        run_lm_eval(args.model_path, task, out_path, args.device, args.batch_size)
        log.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
