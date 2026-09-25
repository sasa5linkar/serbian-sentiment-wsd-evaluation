from __future__ import annotations

import csv

from serbian_sentiment_wsd_eval.coverage import write_sentiment_coverage_report


def read_tsv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_write_sentiment_coverage_report_lists_missing_selected_wsd_ids(tmp_path):
    resources = tmp_path / "resources"
    synsets = resources / "synset_sentiment"
    synsets.mkdir(parents=True)
    (synsets / "S0.csv").write_text("ID,POS,NEG\nENG30-1-n,0.1,0.0\n", encoding="utf-8-sig")
    (synsets / "A1.csv").write_text("ID,POS,NEG\nENG30-2-n,0.0,0.1\n", encoding="utf-8-sig")
    wsd = tmp_path / "wsd.tsv"
    wsd.write_text(
        "sentence_id\ttarget_type\ttoken_indices\ttarget_text\tlemma\tupos\tKBid\tNumberOfSenses\tPossible\tExplanation\tOrigine\n"
        "S1\ttoken\t0\tprogram\tprogram\tNOUN\tENG30-1-n\t2\tENG30-1-n;ENG30-2-n\t\tDISTILLED\n"
        "S2\ttoken\t0\traditi\traditi\tVERB\tCVMWE-1\t1\tCVMWE-1\t\tFIRST\n"
        "S3\ttoken\t0\tnovo\tnovo\tADJ\tNEW_SENSE\t0\t\t\tNone\n",
        encoding="utf-8",
    )
    out = tmp_path / "coverage.tsv"

    report = write_sentiment_coverage_report(wsd, resources, out)

    rows = read_tsv(out)
    assert rows == [
        {"table": "A1", "KBid": "ENG30-1-n", "count": "1", "sentences": "S1"},
        {"table": "A1", "KBid": "CVMWE-1", "count": "1", "sentences": "S2"},
        {"table": "S0", "KBid": "CVMWE-1", "count": "1", "sentences": "S2"},
    ]
    assert report["selected_ids"] == 2
    assert report["missing_rows"] == 3
