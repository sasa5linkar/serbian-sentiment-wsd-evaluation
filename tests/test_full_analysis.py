from __future__ import annotations

from openpyxl import Workbook

from serbian_sentiment_wsd_eval.full_analysis import (
    build_lemma_wsd_overlap_rows,
    build_missing_sentiment_reports,
    build_prediction_wide_rows,
    build_report_metric_rows,
    build_significance_rows,
    build_wsd_avg_comparison_rows,
    build_wsd_stage_coverage_rows,
    classify_sense_id,
    exact_mcnemar_p,
    holm_bonferroni,
    load_gold_rows,
)


def test_classify_sense_id_splits_wordnet_custom_and_new_sense_ids():
    assert classify_sense_id("ENG30-01234567-n") == "eng30_wordnet_like"
    assert classify_sense_id("ENG30-123-n") == "eng30_nonstandard"
    assert classify_sense_id("CVMWE-0224") == "elexis_custom_or_mwe"
    assert classify_sense_id("LLM-0023") == "elexis_custom_or_mwe"
    assert classify_sense_id("DRJ0-209628/0") == "elexis_custom_or_mwe"
    assert classify_sense_id("RSSJ-27928/0") == "elexis_custom_or_mwe"
    assert classify_sense_id("NEW_SENSE") == "new_sense"
    assert classify_sense_id("BILI-00000025") == "other"


def test_build_missing_sentiment_reports_groups_by_id_and_class():
    wsd_rows = [
        {
            "sentence_id": "S1",
            "target_type": "token",
            "target_text": "vreme",
            "lemma": "vreme",
            "upos": "NOUN",
            "KBid": "ENG30-01234567-n",
        },
        {
            "sentence_id": "S1",
            "target_type": "mwe",
            "target_text": "u toku",
            "lemma": "u toku",
            "upos": "ADP",
            "KBid": "CVMWE-1",
        },
        {
            "sentence_id": "S2",
            "target_type": "token",
            "target_text": "loš",
            "lemma": "loš",
            "upos": "ADJ",
            "KBid": "ENG30-123-a",
        },
        {
            "sentence_id": "S3",
            "target_type": "token",
            "target_text": "novo",
            "lemma": "novo",
            "upos": "ADJ",
            "KBid": "NEW_SENSE",
        },
    ]
    sentiment_ids_by_table = {"S0": {"ENG30-01234567-n"}, "A7": set()}

    by_id, by_class = build_missing_sentiment_reports(wsd_rows, sentiment_ids_by_table)

    assert by_id == [
        {
            "table": "A7",
            "KBid": "ENG30-01234567-n",
            "id_class": "eng30_wordnet_like",
            "prefix": "ENG30",
            "count": 1,
            "sentences": "S1",
            "lemmas": "vreme",
            "target_types": "token",
            "upos": "NOUN",
        },
        {
            "table": "A7",
            "KBid": "CVMWE-1",
            "id_class": "elexis_custom_or_mwe",
            "prefix": "CVMWE",
            "count": 1,
            "sentences": "S1",
            "lemmas": "u toku",
            "target_types": "mwe",
            "upos": "ADP",
        },
        {
            "table": "A7",
            "KBid": "ENG30-123-a",
            "id_class": "eng30_nonstandard",
            "prefix": "ENG30",
            "count": 1,
            "sentences": "S2",
            "lemmas": "loš",
            "target_types": "token",
            "upos": "ADJ",
        },
        {
            "table": "S0",
            "KBid": "CVMWE-1",
            "id_class": "elexis_custom_or_mwe",
            "prefix": "CVMWE",
            "count": 1,
            "sentences": "S1",
            "lemmas": "u toku",
            "target_types": "mwe",
            "upos": "ADP",
        },
        {
            "table": "S0",
            "KBid": "ENG30-123-a",
            "id_class": "eng30_nonstandard",
            "prefix": "ENG30",
            "count": 1,
            "sentences": "S2",
            "lemmas": "loš",
            "target_types": "token",
            "upos": "ADJ",
        },
    ]
    assert by_class == [
        {"table": "A7", "id_class": "elexis_custom_or_mwe", "ids": 1, "mentions": 1},
        {"table": "A7", "id_class": "eng30_nonstandard", "ids": 1, "mentions": 1},
        {"table": "A7", "id_class": "eng30_wordnet_like", "ids": 1, "mentions": 1},
        {"table": "S0", "id_class": "elexis_custom_or_mwe", "ids": 1, "mentions": 1},
        {"table": "S0", "id_class": "eng30_nonstandard", "ids": 1, "mentions": 1},
    ]


def test_build_lemma_wsd_overlap_rows_marks_token_level_quadrants():
    tokens = [
        {"sample_id": "S1", "token_index": "0", "lemma": "dobar", "upos": "ADJ"},
        {"sample_id": "S1", "token_index": "1", "lemma": "program", "upos": "NOUN"},
        {"sample_id": "S1", "token_index": "2", "lemma": "los", "upos": "ADJ"},
        {"sample_id": "S1", "token_index": "3", "lemma": "nepoznat", "upos": "NOUN"},
    ]
    wsd_rows = [
        {"sentence_id": "S1", "token_indices": "0", "KBid": "ENG30-1-a", "Possible": "ENG30-1-a"},
        {"sentence_id": "S1", "token_indices": "1", "KBid": "ENG30-2-n", "Possible": "ENG30-2-n"},
        {"sentence_id": "S1", "token_indices": "2", "KBid": "NEW_SENSE", "Possible": ""},
    ]
    polarity = {"dobar": 1, "los": -1}
    sentiment_ids = {"ENG30-1-a", "ENG30-2-n"}

    rows = build_lemma_wsd_overlap_rows(tokens, wsd_rows, polarity, sentiment_ids)

    assert [(row["lemma"], row["quadrant"]) for row in rows] == [
        ("dobar", "both"),
        ("program", "wsd_only"),
        ("los", "lemma_only"),
        ("nepoznat", "neither"),
    ]


def test_build_wsd_stage_coverage_rows_summarizes_by_all_target_type_and_upos():
    wsd_rows = [
        {
            "target_type": "token",
            "upos": "NOUN",
            "Origine": "DISTILLED",
            "KBid": "ENG30-1-n",
            "NumberOfSenses": "2",
            "Possible": "ENG30-1-n;CVMWE-1",
        },
        {
            "target_type": "mwe",
            "upos": "ADP",
            "Origine": "FIRST",
            "KBid": "CVMWE-1",
            "NumberOfSenses": "1",
            "Possible": "CVMWE-1",
        },
        {
            "target_type": "token",
            "upos": "ADJ",
            "Origine": "None",
            "KBid": "NEW_SENSE",
            "NumberOfSenses": "0",
            "Possible": "",
        },
    ]

    rows = build_wsd_stage_coverage_rows(wsd_rows, {"ENG30-1-n"})

    all_row = next(row for row in rows if row["group"] == "all")
    assert all_row == {
        "group": "all",
        "eligible_targets": 3,
        "candidates_found": 2,
        "distilled": 1,
        "first": 1,
        "new_sense": 1,
        "selected_has_sentiment": 1,
        "candidate_any_has_sentiment": 1,
    }
    assert {row["group"] for row in rows} >= {"target_type=token", "target_type=mwe", "upos=NOUN"}


def test_build_prediction_wide_rows_pivots_score_configs():
    score_rows = [
        {"sample_id": "S1", "sentence_text": "Text", "gold_label": "1", "config": "L0", "label": "positive", "score": "1.0", "coverage": "0.5"},
        {"sample_id": "S1", "sentence_text": "Text", "gold_label": "1", "config": "A7-wsd", "label": "neutral", "score": "0.0", "coverage": "0.25"},
    ]

    rows = build_prediction_wide_rows(score_rows)

    assert rows == [
        {
            "sample_id": "S1",
            "sentence_text": "Text",
            "gold_label": "1",
            "L0_label": "positive",
            "L0_score": "1.0",
            "L0_coverage": "0.5",
            "A7-wsd_label": "neutral",
            "A7-wsd_score": "0.0",
            "A7-wsd_coverage": "0.25",
        }
    ]


def test_load_gold_rows_keeps_numeric_zero_as_neutral(tmp_path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Combined"
    sheet.append(["sample_id", "sentiment_label", "annotator"])
    sheet.append(["S1", 0, "A"])
    sheet.append(["S2", -1, "B"])
    sheet.append(["S3", 1, "C"])
    sheet.append(["S4", None, "D"])
    path = tmp_path / "gold.xlsx"
    workbook.save(path)

    rows = load_gold_rows(path)

    assert [(row["sample_id"], row["normalized_gold"]) for row in rows] == [
        ("S1", "neutral"),
        ("S2", "negative"),
        ("S3", "positive"),
    ]


def test_build_report_metric_rows_renames_configs_and_keeps_avg2_control():
    theta_rows = [
        {"theta": "0.0", "config": "L0", "n": "3", "accuracy": "0.4", "macro_f1": "0.3", "coverage_mean": "0.2"},
        {"theta": "0.0", "config": "S0-wsd", "n": "3", "accuracy": "0.5", "macro_f1": "0.4", "coverage_mean": "0.3"},
        {"theta": "0.0", "config": "A1-avg", "n": "3", "accuracy": "0.6", "macro_f1": "0.9", "coverage_mean": "0.4"},
        {"theta": "0.01", "config": "A1-avg2", "n": "3", "accuracy": "0.7", "macro_f1": "0.5", "coverage_mean": "0.6"},
        {"theta": "0.02", "config": "A1-avg2", "n": "3", "accuracy": "0.8", "macro_f1": "0.4", "coverage_mean": "0.6"},
        {"theta": "0.03", "config": "A1-wsd", "n": "3", "accuracy": "0.9", "macro_f1": "0.55", "coverage_mean": "0.3"},
    ]

    rows = build_report_metric_rows(theta_rows)

    assert rows == [
        {"resource": "L0", "method": "lemma", "internal_config": "L0", "theta": "0.0", "n": "3", "accuracy": "0.4", "macro_f1": "0.3", "coverage_mean": "0.2"},
        {"resource": "S0", "method": "wsd", "internal_config": "S0-wsd", "theta": "0.0", "n": "3", "accuracy": "0.5", "macro_f1": "0.4", "coverage_mean": "0.3"},
        {"resource": "S1", "method": "wsd", "internal_config": "A1-wsd", "theta": "0.03", "n": "3", "accuracy": "0.9", "macro_f1": "0.55", "coverage_mean": "0.3"},
        {"resource": "S1", "method": "avg", "internal_config": "A1-avg2", "theta": "0.01", "n": "3", "accuracy": "0.7", "macro_f1": "0.5", "coverage_mean": "0.6"},
    ]


def test_build_wsd_avg_comparison_rows_pairs_report_methods():
    rows = [
        {"resource": "S1", "method": "wsd", "theta": "0.01", "accuracy": "0.4", "macro_f1": "0.3", "coverage_mean": "0.2"},
        {"resource": "S1", "method": "avg", "theta": "0.02", "accuracy": "0.5", "macro_f1": "0.45", "coverage_mean": "0.6"},
        {"resource": "S2", "method": "wsd", "theta": "0.01", "accuracy": "0.6", "macro_f1": "0.5", "coverage_mean": "0.2"},
    ]

    assert build_wsd_avg_comparison_rows(rows) == [
        {
            "resource": "S1",
            "wsd_theta": "0.01",
            "wsd_accuracy": "0.4",
            "wsd_macro_f1": "0.3",
            "wsd_coverage": "0.2",
            "avg_theta": "0.02",
            "avg_accuracy": "0.5",
            "avg_macro_f1": "0.45",
            "avg_coverage": "0.6",
            "delta_avg_minus_wsd": 0.15,
        }
    ]


def test_exact_mcnemar_and_holm_bonferroni():
    assert exact_mcnemar_p(1, 3) == 0.625

    adjusted = holm_bonferroni([0.01, 0.04, 0.03])

    assert adjusted == [0.03, 0.06, 0.06]


def test_build_significance_rows_for_predefined_pairs():
    score_rows = [
        {"sample_id": "S1", "gold_label": "positive", "config": "L0", "score": "1.0", "coverage": "1"},
        {"sample_id": "S2", "gold_label": "negative", "config": "L0", "score": "-1.0", "coverage": "1"},
        {"sample_id": "S3", "gold_label": "positive", "config": "L0", "score": "-1.0", "coverage": "1"},
        {"sample_id": "S1", "gold_label": "positive", "config": "A7-avg2", "score": "0.2", "coverage": "1"},
        {"sample_id": "S2", "gold_label": "negative", "config": "A7-avg2", "score": "-0.2", "coverage": "1"},
        {"sample_id": "S3", "gold_label": "positive", "config": "A7-avg2", "score": "0.2", "coverage": "1"},
    ]
    report_rows = [
        {"resource": "L0", "method": "lemma", "internal_config": "L0", "theta": "0.0"},
        {"resource": "S7", "method": "avg", "internal_config": "A7-avg2", "theta": "0.1"},
    ]

    rows = build_significance_rows(
        score_rows,
        report_rows,
        comparisons=[("S7", "avg", "L0", "lemma")],
        randomization_iterations=20,
        random_seed=1,
    )

    assert len(rows) == 1
    assert rows[0]["system_a"] == "S7-avg"
    assert rows[0]["system_b"] == "L0-lemma"
    assert rows[0]["delta_accuracy"] == 0.333333333333
    assert rows[0]["mcnemar_b"] == 1
    assert rows[0]["mcnemar_c"] == 0
    assert 0.0 <= float(rows[0]["randomization_p"]) <= 1.0
    assert rows[0]["holm_p"] == rows[0]["randomization_p"]
