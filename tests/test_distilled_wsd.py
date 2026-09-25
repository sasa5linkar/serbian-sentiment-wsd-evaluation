from __future__ import annotations

import csv
import json

from serbian_sentiment_wsd_eval.distilled_wsd import (
    ElexisSense,
    run_distilled_wsd,
    write_wsd_tsv,
)


class ReverseRanker:
    def rank(self, text: str, definitions: list[str]) -> list[int]:
        assert "**program**" in text
        return list(reversed(range(len(definitions))))


class SenseAwareRanker:
    def rank_senses(self, text: str, candidates: list[ElexisSense]) -> list[int]:
        assert "**program**" in text
        assert [candidate.sense_id for candidate in candidates] == ["ENG30-1-n", "ENG30-2-n"]
        return [1, 0]


class BatchSenseRanker:
    def __init__(self):
        self.calls = 0

    def rank(self, text: str, definitions: list[str]) -> list[int]:
        raise AssertionError("batch ranker should avoid per-target rank calls")

    def rank_senses_batch(
        self,
        texts: list[str],
        candidates_batch: list[list[ElexisSense]],
    ) -> list[list[int]]:
        self.calls += 1
        assert texts == ["Ovaj **program** radi."]
        assert [[candidate.sense_id for candidate in candidates] for candidates in candidates_batch] == [
            ["ENG30-1-n", "ENG30-2-n"]
        ]
        return [[1, 0]]


def write_annotations(path):
    payload = {
        "id": "S1",
        "text": "Ovaj program radi.",
        "tokens": [
            {
                "index": 0,
                "text": "Ovaj",
                "start": 0,
                "end": 4,
                "pos": "PRO",
                "upos": "DET",
                "lemma": "ovaj",
                "named_entity": "*",
                "mwe_id": "*",
            },
            {
                "index": 1,
                "text": "program",
                "start": 5,
                "end": 12,
                "pos": "N",
                "upos": "NOUN",
                "lemma": "program",
                "named_entity": "*",
                "mwe_id": "*",
            },
            {
                "index": 2,
                "text": "radi",
                "start": 13,
                "end": 17,
                "pos": "V",
                "upos": "VERB",
                "lemma": "raditi",
                "named_entity": "*",
                "mwe_id": "*",
            },
        ],
        "mwes": [],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def test_run_distilled_wsd_uses_ranker_for_multiple_candidates_and_first_for_singletons(tmp_path):
    annotations = tmp_path / "annotations.jsonl"
    write_annotations(annotations)
    senses = [
        ElexisSense("ENG30-1-n", "program", "NOUN", "javna priredba"),
        ElexisSense("ENG30-2-n", "program", "NOUN", "televizijski sadrzaj"),
        ElexisSense("ENG30-3-v", "raditi", "VERB", "biti u funkciji"),
    ]

    decisions = run_distilled_wsd(
        annotations_jsonl=annotations,
        senses=senses,
        ranker=ReverseRanker(),
    )

    assert [(row.target_text, row.selected_sense_id, row.origin) for row in decisions] == [
        ("program", "ENG30-2-n", "DISTILLED"),
        ("radi", "ENG30-3-v", "FIRST"),
    ]
    assert decisions[0].candidate_sense_ids == ["ENG30-1-n", "ENG30-2-n"]


def test_run_distilled_wsd_uses_sense_aware_ranker_when_available(tmp_path):
    annotations = tmp_path / "annotations.jsonl"
    write_annotations(annotations)
    senses = [
        ElexisSense("ENG30-1-n", "program", "NOUN", "javna priredba"),
        ElexisSense("ENG30-2-n", "program", "NOUN", "televizijski sadrzaj"),
    ]

    decisions = run_distilled_wsd(
        annotations_jsonl=annotations,
        senses=senses,
        ranker=SenseAwareRanker(),
    )

    assert decisions[0].selected_sense_id == "ENG30-2-n"
    assert decisions[0].origin == "DISTILLED"


def test_run_distilled_wsd_uses_batch_ranker_when_available(tmp_path):
    annotations = tmp_path / "annotations.jsonl"
    write_annotations(annotations)
    senses = [
        ElexisSense("ENG30-1-n", "program", "NOUN", "javna priredba"),
        ElexisSense("ENG30-2-n", "program", "NOUN", "televizijski sadrzaj"),
    ]
    ranker = BatchSenseRanker()

    decisions = run_distilled_wsd(
        annotations_jsonl=annotations,
        senses=senses,
        ranker=ranker,
    )

    assert ranker.calls == 1
    assert decisions[0].selected_sense_id == "ENG30-2-n"
    assert decisions[0].origin == "DISTILLED"


def test_run_distilled_wsd_records_new_sense_when_no_candidates(tmp_path):
    annotations = tmp_path / "annotations.jsonl"
    write_annotations(annotations)

    decisions = run_distilled_wsd(
        annotations_jsonl=annotations,
        senses=[],
        ranker=ReverseRanker(),
    )

    assert [(row.target_text, row.selected_sense_id, row.origin) for row in decisions] == [
        ("program", "NEW_SENSE", "None"),
        ("radi", "NEW_SENSE", "None"),
    ]


def test_write_wsd_tsv_uses_existing_scoring_columns(tmp_path):
    annotations = tmp_path / "annotations.jsonl"
    write_annotations(annotations)
    decisions = run_distilled_wsd(
        annotations_jsonl=annotations,
        senses=[ElexisSense("ENG30-1-n", "program", "NOUN", "definition")],
        ranker=ReverseRanker(),
    )
    out = tmp_path / "wsd.tsv"

    write_wsd_tsv(decisions, out)

    with out.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert list(rows[0]) == [
        "sentence_id",
        "target_type",
        "token_indices",
        "target_text",
        "lemma",
        "upos",
        "KBid",
        "NumberOfSenses",
        "Possible",
        "Explanation",
        "Origine",
    ]
    assert rows[0]["KBid"] == "ENG30-1-n"
