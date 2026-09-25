from __future__ import annotations

import csv

from serbian_sentiment_wsd_eval.resources import (
    normalize_polarity_lexicon,
    normalize_sentiment_table,
)


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_normalize_polarity_lexicon_marks_duplicate_conflicts_neutral(tmp_path):
    source = tmp_path / "recnikPolariteta.csv"
    source.write_text(
        '"dobar";1;\n'
        '"los";;1\n'
        '"neverovatno";1;\n'
        '"neverovatno";;1\n',
        encoding="utf-8-sig",
    )
    out = tmp_path / "lemma_polarity.csv"

    report = normalize_polarity_lexicon(source, out)

    rows = {row["lemma"]: row for row in read_csv(out)}
    assert rows["dobar"]["polarity"] == "1"
    assert rows["los"]["polarity"] == "-1"
    assert rows["neverovatno"]["polarity"] == "0"
    assert report["duplicate_conflicts"] == ["neverovatno"]
    assert report["rows_written"] == 3
    assert out.read_bytes().startswith(b"\xef\xbb\xbf")


def test_normalize_sentiment_table_keeps_sentiment_and_lemma_pos_metadata(tmp_path):
    source = tmp_path / "srbsentiwordnet_a7.csv"
    source.write_text(
        ",ID,POS,NEG,Lemme,Definicija,Vrsta\n"
        "0,ENG30-1-n,0.25,0.75,lemma,definition,n\n",
        encoding="utf-8-sig",
    )
    out = tmp_path / "A7.csv"

    report = normalize_sentiment_table(source, out, label="A7")

    rows = read_csv(out)
    assert rows == [{"ID": "ENG30-1-n", "POS": "0.25", "NEG": "0.75", "Lemme": "lemma", "Vrsta": "n"}]
    assert report["label"] == "A7"
    assert report["rows_written"] == 1
