"""
src/download_blimp_cache.py

Downloads the BLiMP dataset from HuggingFace and saves each paradigm as a
.jsonl file under data/blimp_cache/, which eval_checkpoints.py reads for
BLiMP scoring at each checkpoint.

Usage:
    python src/download_blimp_cache.py
    python src/download_blimp_cache.py --out_dir data/blimp_cache
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path,
                   default=Path(__file__).parent.parent / "data" / "blimp_cache")
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Run: pip install datasets")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    PARADIGMS = [
        "adjunct_island", "anaphor_gender_agreement", "anaphor_number_agreement",
        "animate_subject_passive", "animate_subject_trans", "causative",
        "complex_NP_island", "coordinate_structure_constraint_complex_left_branch",
        "coordinate_structure_constraint_object_extraction",
        "determiner_noun_agreement_1", "determiner_noun_agreement_2",
        "determiner_noun_agreement_irregular_1", "determiner_noun_agreement_irregular_2",
        "determiner_noun_agreement_with_adj_2", "determiner_noun_agreement_with_adj_irregular_1",
        "determiner_noun_agreement_with_adj_irregular_2", "determiner_noun_agreement_with_adjective_1",
        "distractor_agreement_relational_noun", "distractor_agreement_relative_clause",
        "drop_argument", "ellipsis_n_bar_1", "ellipsis_n_bar_2",
        "existential_there_object_raising", "existential_there_quantifiers_1",
        "existential_there_quantifiers_2", "existential_there_subject_raising",
        "expletive_it_object_raising", "inchoative", "intransitive",
        "irregular_past_participle_adjectives", "irregular_past_participle_verbs",
        "irregular_plural_subject_verb_agreement_1", "irregular_plural_subject_verb_agreement_2",
        "left_branch_island_echo_question", "left_branch_island_simple_question",
        "matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2",
        "only_npi_licensor_present", "only_npi_scope", "passive_1", "passive_2",
        "principle_A_c_command", "principle_A_case_1", "principle_A_case_2",
        "principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3",
        "principle_A_reconstruction", "regular_plural_subject_verb_agreement_1",
        "regular_plural_subject_verb_agreement_2", "sentential_negation_npi_licensor_present",
        "sentential_negation_npi_scope", "sentential_subject_island",
        "superlative_quantifiers_1", "superlative_quantifiers_2",
        "tough_vs_raising_1", "tough_vs_raising_2", "transitive", "wh_island",
        "wh_questions_object_gap", "wh_questions_subject_gap",
        "wh_questions_subject_gap_long_distance", "wh_vs_that_no_gap",
        "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap",
        "wh_vs_that_with_gap_long_distance",
    ]

    total = 0
    for paradigm in PARADIGMS:
        log.info("Downloading paradigm: %s", paradigm)
        ds = load_dataset("nyu-mll/blimp", paradigm, split="train")
        items = [{"sentence_good": row["sentence_good"],
                  "sentence_bad":  row["sentence_bad"]} for row in ds]
        out_file = args.out_dir / f"{paradigm}.jsonl"
        out_file.write_text(
            "\n".join(json.dumps(item) for item in items),
            encoding="utf-8",
        )
        total += len(items)

    log.info("Saved %d items across %d paradigms → %s",
             total, len(PARADIGMS), args.out_dir)


if __name__ == "__main__":
    main()
