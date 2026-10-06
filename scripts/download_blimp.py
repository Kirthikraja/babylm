"""
scripts/download_blimp.py

Download BLiMP from HuggingFace and save as per-paradigm JSONL files
in data/blimp_cache/ for use by eval_checkpoints.py.

Usage (run once, from repo root):
    python scripts/download_blimp.py
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

BLIMP_PARADIGMS = [
    "anaphor_gender_agreement", "anaphor_number_agreement",
    "animate_subject_passive", "animate_subject_trans", "causative",
    "drop_argument", "inchoative", "intransitive", "passive_1", "passive_2",
    "transitive", "principle_A_c_command", "principle_A_case_1",
    "principle_A_case_2", "principle_A_domain_1", "principle_A_domain_2",
    "principle_A_domain_3", "principle_A_reconstruction",
    "existential_there_object_raising", "existential_there_subject_raising",
    "expletive_it_object_raising", "tough_vs_raising_1", "tough_vs_raising_2",
    "determiner_noun_agreement_1", "determiner_noun_agreement_2",
    "determiner_noun_agreement_irregular_1", "determiner_noun_agreement_irregular_2",
    "determiner_noun_agreement_with_adj_1", "determiner_noun_agreement_with_adj_2",
    "determiner_noun_agreement_with_adj_irregular_1",
    "determiner_noun_agreement_with_adj_irregular_2",
    "ellipsis_n_bar_1", "ellipsis_n_bar_2",
    "wh_questions_object_gap", "wh_questions_object_gap_long",
    "wh_questions_subject_gap", "wh_questions_subject_gap_long_distance",
    "irregular_past_participle_adjectives", "irregular_past_participle_verbs",
    "adjunct_island", "complex_NP_island",
    "coordinate_structure_constraint_complex_left_branch",
    "coordinate_structure_constraint_object_extraction",
    "left_branch_island_echo_question", "left_branch_island_simple_question",
    "sentential_subject_island", "wh_island",
    "matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2",
    "only_npi_licensor_present", "only_npi_scope",
    "sentential_negation_npi_licensor_present", "sentential_negation_npi_scope",
    "existential_there_quantifiers_1", "existential_there_quantifiers_2",
    "superlative_quantifiers_1", "superlative_quantifiers_2",
    "distractor_agreement_relational_noun", "distractor_agreement_relative_clause",
    "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2",
    "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2",
]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out_dir", type=Path,
                   default=Path(__file__).parent.parent / "data" / "blimp_cache")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets library not installed — run: pip install datasets")
        sys.exit(1)

    total = 0
    for paradigm in BLIMP_PARADIGMS:
        out_path = args.out_dir / f"{paradigm}.jsonl"
        if out_path.exists():
            log.info("  %s: already exists — skipping", paradigm)
            continue
        try:
            ds = load_dataset("nyu-mll/blimp", paradigm, split="train")
        except Exception as exc:
            log.warning("  %s: failed (%s) — skipping", paradigm, exc)
            continue
        with out_path.open("w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps({
                    "sentence_good": row["sentence_good"],
                    "sentence_bad":  row["sentence_bad"],
                }) + "\n")
        total += len(ds)
        log.info("  %s: %d items", paradigm, len(ds))

    log.info("Done. %d total items saved to %s", total, args.out_dir)


if __name__ == "__main__":
    main()
