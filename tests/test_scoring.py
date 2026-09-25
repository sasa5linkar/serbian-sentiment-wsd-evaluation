from __future__ import annotations

from serbian_sentiment_wsd_eval.scoring import (
    ScoreResult,
    build_lemma_pos_index,
    label_lemma_score,
    label_synset_score,
    score_dataset,
    score_lemma_units,
    score_synset_units,
)


def test_score_lemma_units_averages_covered_content_lemmas():
    units = [
        {"lemma": "dobar", "upos": "ADJ"},
        {"lemma": "los", "upos": "ADJ"},
        {"lemma": "i", "upos": "CCONJ"},
        {"lemma": "nepoznat", "upos": "NOUN"},
    ]
    polarity = {"dobar": 1, "los": -1}

    result = score_lemma_units(units, polarity)

    assert result == ScoreResult(score=0.0, label="neutral", covered_units=2, total_units=3)


def test_score_lemma_units_returns_neutral_without_coverage():
    result = score_lemma_units([{"lemma": "i", "upos": "CCONJ"}], {"dobar": 1})

    assert result == ScoreResult(score=0.0, label="neutral", covered_units=0, total_units=0)


def test_score_synset_units_uses_selected_senses_and_threshold():
    units = [
        {"KBid": "ENG30-1-n", "Possible": "ENG30-1-n;ENG30-2-n"},
        {"KBid": "NEW_SENSE", "Possible": ""},
    ]
    sentiments = {
        "ENG30-1-n": {"POS": 0.8, "NEG": 0.1},
        "ENG30-2-n": {"POS": 0.1, "NEG": 0.8},
    }

    result = score_synset_units(units, sentiments, theta=0.33, mode="wsd")

    assert result == ScoreResult(score=0.7, label="positive", covered_units=1, total_units=2)


def test_score_synset_units_can_average_candidate_senses_without_wsd():
    units = [{"KBid": "ENG30-1-n", "Possible": "ENG30-1-n;ENG30-2-n"}]
    sentiments = {
        "ENG30-1-n": {"POS": 0.8, "NEG": 0.1},
        "ENG30-2-n": {"POS": 0.1, "NEG": 0.2},
    }

    result = score_synset_units(units, sentiments, theta=0.33, mode="avg")

    assert result == ScoreResult(score=0.3, label="neutral", covered_units=1, total_units=1)


def test_score_synset_units_can_average_all_table_senses_for_same_lemma_and_pos():
    units = [{"lemma": "banka", "upos": "NOUN", "KBid": "NEW_SENSE", "Possible": ""}]
    sentiments = {
        "ENG30-1-n": {"POS": 0.8, "NEG": 0.1, "Lemme": "banka,bankarska ustanova", "Vrsta": "n"},
        "ENG30-2-n": {"POS": 0.1, "NEG": 0.4, "Lemme": "banka", "Vrsta": "n"},
        "ENG30-3-v": {"POS": 1.0, "NEG": 0.0, "Lemme": "banka", "Vrsta": "v"},
    }

    result = score_synset_units(
        units,
        sentiments,
        theta=0.1,
        mode="avg2",
        lemma_pos_index=build_lemma_pos_index(sentiments),
    )

    assert result == ScoreResult(score=0.2, label="positive", covered_units=1, total_units=1)


def test_label_helpers_are_explicit_about_thresholds():
    assert label_lemma_score(0.1) == "positive"
    assert label_lemma_score(-0.1) == "negative"
    assert label_lemma_score(0.0) == "neutral"
    assert label_synset_score(0.34, theta=0.33) == "positive"
    assert label_synset_score(-0.34, theta=0.33) == "negative"
    assert label_synset_score(0.33, theta=0.33) == "neutral"


def test_score_dataset_loads_each_synset_table_once(tmp_path, monkeypatch):
    resources = tmp_path / "resources"
    synsets = resources / "synset_sentiment"
    synsets.mkdir(parents=True)
    (resources / "lemma_polarity.csv").write_text("lemma,polarity\nlemma,1\n", encoding="utf-8-sig")
    (synsets / "S0.csv").write_text("ID,POS,NEG\nENG30-1-n,0.8,0.1\n", encoding="utf-8-sig")
    sample = tmp_path / "sample.tsv"
    sample.write_text(
        "sample_id\tsentence_text\tsentiment_label\nS1\tText.\t\nS2\tText.\t\n",
        encoding="utf-8-sig",
    )
    tokens = tmp_path / "tokens.tsv"
    tokens.write_text(
        "sample_id\ttoken_index\ttoken_text\tupos\tlemma\n"
        "S1\t0\tLemma\tNOUN\tlemma\n"
        "S2\t0\tLemma\tNOUN\tlemma\n",
        encoding="utf-8-sig",
    )
    wsd = tmp_path / "wsd.tsv"
    wsd.write_text(
        "sentence_id\ttarget_type\ttoken_indices\ttarget_text\tlemma\tupos\tKBid\tNumberOfSenses\tPossible\tExplanation\tOrigine\n"
        "S1\ttoken\t0\tLemma\tlemma\tNOUN\tENG30-1-n\t1\tENG30-1-n\tfirst\tFIRST\n"
        "S2\ttoken\t0\tLemma\tlemma\tNOUN\tENG30-1-n\t1\tENG30-1-n\tfirst\tFIRST\n",
        encoding="utf-8",
    )

    import serbian_sentiment_wsd_eval.scoring as scoring_module

    calls = []
    real_loader = scoring_module.load_sentiment_table

    def counting_loader(path):
        calls.append(path)
        return real_loader(path)

    monkeypatch.setattr(scoring_module, "load_sentiment_table", counting_loader)

    score_dataset(
        sample_path=sample,
        token_path=tokens,
        wsd_path=wsd,
        resources_dir=resources,
        out_path=tmp_path / "scores.tsv",
    )

    assert calls == [synsets / "S0.csv"]
