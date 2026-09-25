from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .constants import CONTENT_UPOS
from .csvio import read_delimited, write_dicts


UPOS_TO_WORDNET_POS = {
    "NOUN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "b",
}


@dataclass(frozen=True)
class ScoreResult:
    score: float
    label: str
    covered_units: int
    total_units: int


def label_lemma_score(score: float) -> str:
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def label_synset_score(score: float, *, theta: float) -> str:
    if score > theta:
        return "positive"
    if score < -theta:
        return "negative"
    return "neutral"


def score_lemma_units(
    units: list[dict[str, str]],
    polarity: dict[str, int],
    *,
    content_upos: set[str] = CONTENT_UPOS,
) -> ScoreResult:
    content_units = [unit for unit in units if unit.get("upos") in content_upos]
    values = [
        polarity[unit.get("lemma", "")]
        for unit in content_units
        if unit.get("lemma", "") in polarity
    ]
    if not values:
        return ScoreResult(score=0.0, label="neutral", covered_units=0, total_units=len(content_units))
    score = round(sum(values) / len(values), 12)
    return ScoreResult(
        score=score,
        label=label_lemma_score(score),
        covered_units=len(values),
        total_units=len(content_units),
    )


def score_synset_units(
    units: list[dict[str, str]],
    sentiments: dict[str, dict[str, float]],
    *,
    theta: float,
    mode: str,
    lemma_pos_index: dict[tuple[str, str], list[float]] | None = None,
) -> ScoreResult:
    if mode not in {"wsd", "avg", "avg2"}:
        raise ValueError("mode must be 'wsd', 'avg', or 'avg2'")

    contributions: list[float] = []
    for unit in units:
        if mode == "wsd":
            synset_id = _clean_synset_id(unit.get("KBid", ""))
            if synset_id in sentiments:
                contributions.append(_synset_delta(sentiments[synset_id]))
        elif mode == "avg":
            candidate_scores = [
                _synset_delta(sentiments[candidate])
                for candidate in _parse_candidates(unit.get("Possible", ""))
                if candidate in sentiments
            ]
            if candidate_scores:
                contributions.append(sum(candidate_scores) / len(candidate_scores))
        else:
            index = lemma_pos_index if lemma_pos_index is not None else build_lemma_pos_index(sentiments)
            wn_pos = UPOS_TO_WORDNET_POS.get((unit.get("upos") or "").strip())
            if not wn_pos:
                continue
            lemma_scores = index.get((_normalize_lemma(unit.get("lemma", "")), wn_pos), [])
            if lemma_scores:
                contributions.append(sum(lemma_scores) / len(lemma_scores))

    if not contributions:
        return ScoreResult(score=0.0, label="neutral", covered_units=0, total_units=len(units))
    score = round(sum(contributions) / len(contributions), 12)
    return ScoreResult(
        score=score,
        label=label_synset_score(score, theta=theta),
        covered_units=len(contributions),
        total_units=len(units),
    )


def load_polarity_table(path: str | Path) -> dict[str, int]:
    rows = read_delimited(path, delimiter=",")
    return {row["lemma"]: int(row["polarity"]) for row in rows if row.get("lemma")}


def load_sentiment_table(path: str | Path) -> dict[str, dict[str, float]]:
    rows = read_delimited(path, delimiter=",")
    return {
        row["ID"]: {
            "POS": float(row["POS"]),
            "NEG": float(row["NEG"]),
            "Lemme": row.get("Lemme", ""),
            "Vrsta": row.get("Vrsta", ""),
        }
        for row in rows
        if row.get("ID")
    }


def build_lemma_pos_index(sentiments: dict[str, dict[str, float]]) -> dict[tuple[str, str], list[float]]:
    index: dict[tuple[str, str], list[float]] = {}
    for sentiment in sentiments.values():
        wn_pos = str(sentiment.get("Vrsta", "")).strip().lower()
        if not wn_pos:
            continue
        for lemma in _split_lemmas(str(sentiment.get("Lemme", ""))):
            index.setdefault((lemma, wn_pos), []).append(_synset_delta(sentiment))
    return index


def score_dataset(
    *,
    sample_path: str | Path,
    token_path: str | Path,
    wsd_path: str | Path,
    resources_dir: str | Path,
    out_path: str | Path,
    theta: float = 0.33,
) -> Path:
    samples = read_delimited(sample_path, delimiter="\t")
    tokens_by_sentence = _group_by(read_delimited(token_path, delimiter="\t"), "sample_id")
    wsd_by_sentence = _group_by(read_delimited(wsd_path, delimiter="\t"), "sentence_id")

    resource_path = Path(resources_dir)
    polarity = load_polarity_table(resource_path / "lemma_polarity.csv")
    synset_tables = _discover_synset_tables(resource_path / "synset_sentiment")
    loaded_synset_tables = []
    for label, table_path in synset_tables:
        sentiments = load_sentiment_table(table_path)
        loaded_synset_tables.append((label, sentiments, build_lemma_pos_index(sentiments)))

    rows: list[dict[str, object]] = []
    for sample in samples:
        sample_id = sample.get("sample_id", "")
        base = {
            "sample_id": sample_id,
            "sentence_text": sample.get("sentence_text", ""),
            "gold_label": sample.get("sentiment_label", ""),
            "theta": theta,
        }
        lemma_score = score_lemma_units(tokens_by_sentence.get(sample_id, []), polarity)
        rows.append(_score_row(base, "L0", lemma_score))

        decisions = wsd_by_sentence.get(sample_id, [])
        for label, sentiments, lemma_pos_index in loaded_synset_tables:
            rows.append(_score_row(base, f"{label}-wsd", score_synset_units(
                decisions, sentiments, theta=theta, mode="wsd"
            )))
            rows.append(_score_row(base, f"{label}-avg", score_synset_units(
                decisions, sentiments, theta=theta, mode="avg"
            )))
            rows.append(_score_row(base, f"{label}-avg2", score_synset_units(
                decisions, sentiments, theta=theta, mode="avg2", lemma_pos_index=lemma_pos_index
            )))

    return write_dicts(
        out_path,
        rows,
        [
            "sample_id",
            "sentence_text",
            "gold_label",
            "config",
            "score",
            "label",
            "covered_units",
            "total_units",
            "coverage",
            "theta",
        ],
        delimiter="\t",
    )


def _score_row(base: dict[str, object], config: str, result: ScoreResult) -> dict[str, object]:
    total = result.total_units
    coverage = round(result.covered_units / total, 12) if total else 0.0
    return {
        **base,
        "config": config,
        "score": result.score,
        "label": result.label,
        "covered_units": result.covered_units,
        "total_units": total,
        "coverage": coverage,
    }


def _group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get(key, ""), []).append(row)
    return grouped


def _discover_synset_tables(path: Path) -> list[tuple[str, Path]]:
    tables = [(table.stem, table) for table in path.glob("*.csv")]
    order = {"S0": 0, **{f"A{i}": i for i in range(1, 8)}}
    return sorted(tables, key=lambda item: (order.get(item[0], 100), item[0]))


def _parse_candidates(value: str) -> list[str]:
    return [_clean_synset_id(part) for part in value.split(";") if _clean_synset_id(part)]


def _split_lemmas(value: str) -> list[str]:
    return [
        normalized
        for part in value.split(",")
        if (normalized := _normalize_lemma(part))
    ]


def _normalize_lemma(value: str) -> str:
    return value.strip().casefold()


def _clean_synset_id(value: str) -> str:
    return value.split("[", 1)[0].strip()


def _synset_delta(sentiment: dict[str, float]) -> float:
    return sentiment["POS"] - sentiment["NEG"]
