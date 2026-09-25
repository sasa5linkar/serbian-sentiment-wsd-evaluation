from __future__ import annotations

from serbian_sentiment_wsd_eval.evaluation import normalize_gold_label


def test_normalize_gold_label_preserves_numeric_zero_as_neutral():
    assert normalize_gold_label(0) == "neutral"
    assert normalize_gold_label(0.0) == "neutral"
    assert normalize_gold_label("0") == "neutral"
