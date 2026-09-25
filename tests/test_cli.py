from __future__ import annotations

import csv
import json
from pathlib import Path

import serbian_sentiment_wsd_eval.cli as cli_module
from serbian_sentiment_wsd_eval.cli import main


def read_tsv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_score_and_evaluate_cli_with_fixture_data(tmp_path):
    resources = tmp_path / "resources"
    synsets = resources / "synset_sentiment"
    synsets.mkdir(parents=True)
    (resources / "lemma_polarity.csv").write_text(
        "lemma,polarity\n"
        "dobar,1\n"
        "los,-1\n",
        encoding="utf-8-sig",
    )
    (synsets / "S0.csv").write_text(
        "ID,POS,NEG\nENG30-1-n,0.8,0.1\nENG30-2-n,0.1,0.8\n",
        encoding="utf-8-sig",
    )
    (synsets / "A1.csv").write_text(
        "ID,POS,NEG\nENG30-1-n,0.1,0.0\nENG30-2-n,0.0,0.1\n",
        encoding="utf-8-sig",
    )
    sample = tmp_path / "sample.tsv"
    sample.write_text(
        "sample_id\tsentence_text\tsentiment_label\n"
        "S1\tDobar test.\tpositive\n",
        encoding="utf-8-sig",
    )
    tokens = tmp_path / "tokens.tsv"
    tokens.write_text(
        "sample_id\ttoken_index\ttoken_text\tupos\tlemma\n"
        "S1\t0\tDobar\tADJ\tdobar\n"
        "S1\t1\ttest\tNOUN\ttest\n",
        encoding="utf-8-sig",
    )
    wsd = tmp_path / "wsd.tsv"
    wsd.write_text(
        "sentence_id\ttarget_type\ttoken_indices\ttarget_text\tlemma\tupos\tKBid\tNumberOfSenses\tPossible\tExplanation\tOrigine\n"
        "S1\ttoken\t0\tDobar\tdobar\tADJ\tENG30-1-n\t2\tENG30-1-n;ENG30-2-n\tfirst\tFIRST\n",
        encoding="utf-8",
    )
    scores = tmp_path / "scores.tsv"
    eval_dir = tmp_path / "eval"

    assert main([
        "score",
        "--sample",
        str(sample),
        "--tokens",
        str(tokens),
        "--wsd",
        str(wsd),
        "--resources",
        str(resources),
        "--out",
        str(scores),
    ]) == 0
    rows = read_tsv(scores)
    configs = {row["config"] for row in rows}
    assert {"L0", "S0-wsd", "S0-avg", "S0-avg2", "A1-wsd", "A1-avg", "A1-avg2"} <= configs

    assert main(["evaluate", "--scores", str(scores), "--out-dir", str(eval_dir)]) == 0
    metrics = (eval_dir / "metrics.csv").read_text(encoding="utf-8-sig")
    assert "macro_f1" in metrics
    assert "L0" in metrics


def test_run_distilled_wsd_cli_delegates_to_distilled_runner(tmp_path, monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        out = Path(kwargs["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("sentence_id\tKBid\nS1\tENG30-1-n\n", encoding="utf-8-sig")
        return out

    monkeypatch.setattr(cli_module, "run_distilled_wsd_from_paths", fake_run, raising=False)
    out = tmp_path / "wsd.tsv"

    assert main([
        "run-distilled-wsd",
        "--annotations-jsonl",
        str(tmp_path / "annotations.jsonl"),
        "--sense-repo",
        str(tmp_path / "repo.xlsx"),
        "--model",
        str(tmp_path / "model"),
        "--out",
        str(out),
        "--sense-sheet",
        "SrWSD-v2",
        "--sense-sheet",
        "SrWSD-eliminisanPOS",
    ]) == 0

    assert calls == [
        {
            "annotations_jsonl": tmp_path / "annotations.jsonl",
            "sense_repo": tmp_path / "repo.xlsx",
            "model": tmp_path / "model",
            "out_path": out,
            "sense_sheets": ["SrWSD-v2", "SrWSD-eliminisanPOS"],
            "text_prefix": None,
        }
    ]


def test_score_cli_can_write_sentiment_coverage_report(tmp_path):
    resources = tmp_path / "resources"
    synsets = resources / "synset_sentiment"
    synsets.mkdir(parents=True)
    (resources / "lemma_polarity.csv").write_text("lemma,polarity\nprogram,1\n", encoding="utf-8-sig")
    (synsets / "S0.csv").write_text("ID,POS,NEG\nENG30-1-n,0.8,0.1\n", encoding="utf-8-sig")
    sample = tmp_path / "sample.tsv"
    sample.write_text(
        "sample_id\tsentence_text\tsentiment_label\nS1\tProgram radi.\t\n",
        encoding="utf-8-sig",
    )
    tokens = tmp_path / "tokens.tsv"
    tokens.write_text(
        "sample_id\ttoken_index\ttoken_text\tupos\tlemma\nS1\t0\tProgram\tNOUN\tprogram\n",
        encoding="utf-8-sig",
    )
    wsd = tmp_path / "wsd.tsv"
    wsd.write_text(
        "sentence_id\ttarget_type\ttoken_indices\ttarget_text\tlemma\tupos\tKBid\tNumberOfSenses\tPossible\tExplanation\tOrigine\n"
        "S1\ttoken\t0\tProgram\tprogram\tNOUN\tCVMWE-1\t1\tCVMWE-1\t\tFIRST\n",
        encoding="utf-8",
    )
    scores = tmp_path / "scores.tsv"
    coverage = tmp_path / "coverage.tsv"

    assert main([
        "score",
        "--sample",
        str(sample),
        "--tokens",
        str(tokens),
        "--wsd",
        str(wsd),
        "--resources",
        str(resources),
        "--out",
        str(scores),
        "--coverage-out",
        str(coverage),
    ]) == 0

    rows = read_tsv(coverage)
    assert rows == [{"table": "S0", "KBid": "CVMWE-1", "count": "1", "sentences": "S1"}]


def test_demo_distilled_cli_prints_json_for_one_sample(tmp_path, monkeypatch, capsys):
    def fake_demo(**kwargs):
        assert kwargs["sample_id"] == "S1"
        return {
            "sample_id": "S1",
            "sentence": "Program radi.",
            "decisions": [{"target_text": "Program", "KBid": "ENG30-1-n"}],
        }

    monkeypatch.setattr(cli_module, "build_distilled_demo", fake_demo, raising=False)

    assert main([
        "demo-distilled",
        "--sample-id",
        "S1",
        "--annotations-jsonl",
        str(tmp_path / "annotations.jsonl"),
        "--sense-repo",
        str(tmp_path / "repo.xlsx"),
        "--model",
        str(tmp_path / "model"),
        "--resources",
        str(tmp_path / "resources"),
    ]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "sample_id": "S1",
        "sentence": "Program radi.",
        "decisions": [{"target_text": "Program", "KBid": "ENG30-1-n"}],
    }


def test_configure_utf8_stdio_reconfigures_windows_streams(monkeypatch):
    class FakeStream:
        def __init__(self):
            self.calls = []

        def reconfigure(self, **kwargs):
            self.calls.append(kwargs)

    stdout = FakeStream()
    stderr = FakeStream()
    monkeypatch.setattr(cli_module.sys, "stdout", stdout)
    monkeypatch.setattr(cli_module.sys, "stderr", stderr)

    cli_module._configure_utf8_stdio()

    assert stdout.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert stderr.calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_run_full_analysis_cli_delegates_to_workflow(tmp_path, monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return {"out_dir": str(kwargs["out_dir"]), "metrics_by_config": "metrics.tsv"}

    monkeypatch.setattr(cli_module, "run_full_analysis", fake_run, raising=False)
    out_dir = tmp_path / "analysis"

    assert main([
        "run-full-analysis",
        "--gold-workbook",
        str(tmp_path / "gold.xlsx"),
        "--sample",
        str(tmp_path / "sample.tsv"),
        "--tokens",
        str(tmp_path / "tokens.tsv"),
        "--annotations-jsonl",
        str(tmp_path / "annotations.jsonl"),
        "--sense-repo",
        str(tmp_path / "repo.xlsx"),
        "--model",
        str(tmp_path / "model"),
        "--resources",
        str(tmp_path / "resources"),
        "--out-dir",
        str(out_dir),
        "--theta",
        "0.2",
        "--theta-sweep",
        "0.0,0.2,0.5",
        "--reuse-wsd",
        str(tmp_path / "existing_wsd.tsv"),
    ]) == 0

    assert calls == [
        {
            "gold_workbook": tmp_path / "gold.xlsx",
            "sample": tmp_path / "sample.tsv",
            "tokens": tmp_path / "tokens.tsv",
            "annotations_jsonl": tmp_path / "annotations.jsonl",
            "sense_repo": tmp_path / "repo.xlsx",
            "model": tmp_path / "model",
            "resources": tmp_path / "resources",
            "out_dir": out_dir,
            "theta": 0.2,
            "theta_sweep": [0.0, 0.2, 0.5],
            "reuse_wsd": tmp_path / "existing_wsd.tsv",
            "sense_sheets": None,
            "text_prefix": None,
        }
    ]
