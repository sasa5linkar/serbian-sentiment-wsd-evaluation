from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from .constants import CONTENT_UPOS
from .elexis import ElexisSense, ElexisSenseIndex, load_elexis_senses


ENTITY_ALLOWED_LABELS = {"*", "_", "", "ROLE", "EVENT", "DEMO", "PRODUCT", "WORK"}
WSD_TSV_COLUMNS = [
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


class Ranker(Protocol):
    def rank(self, text: str, definitions: list[str]) -> list[int]:
        """Return candidate indexes ordered best to worst."""


@dataclass(frozen=True)
class WSDTarget:
    sentence_id: str
    target_type: str
    token_indices: list[int]
    target_text: str
    lemma: str
    upos: str
    candidates: list[ElexisSense]
    marked_sentence: str


@dataclass(frozen=True)
class WSDDecision:
    sentence_id: str
    target_type: str
    token_indices: list[int]
    target_text: str
    lemma: str
    upos: str
    selected_sense_id: str
    origin: str
    sense_count: int
    candidate_sense_ids: list[str]
    explanation: str


class DistilledTransformerRanker:
    def __init__(self, model_path: str | Path, *, text_prefix: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only when optional deps are missing
            raise RuntimeError("Install with `pip install -e .[wsd]` to use distilled WSD") from exc

        self.torch = torch
        self.model_path = Path(model_path)
        self.text_prefix = _resolve_text_prefix(self.model_path, text_prefix)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
        self.model.eval()

    def encode(self, texts: list[str]):
        inputs = [self._prefix(text) for text in texts]
        with self.torch.no_grad():
            batch = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            output = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return self.torch.nn.functional.normalize(pooled, p=2, dim=1)

    def rank(self, text: str, definitions: list[str]) -> list[int]:
        embeddings = self.encode([text, *definitions])
        scores = (embeddings[0:1] @ embeddings[1:].T).squeeze(0).tolist()
        return [index for index, _score in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]

    def _prefix(self, text: str) -> str:
        if not self.text_prefix or text.startswith(self.text_prefix):
            return text
        return f"{self.text_prefix}{text}"


class CachedDistilledTransformerRanker(DistilledTransformerRanker):
    def __init__(
        self,
        model_path: str | Path,
        *,
        senses: Iterable[ElexisSense],
        text_prefix: str | None = None,
        batch_size: int = 64,
    ) -> None:
        super().__init__(model_path, text_prefix=text_prefix)
        self.batch_size = batch_size
        self.definition_embeddings = {}
        unique_senses: list[ElexisSense] = []
        seen: set[str] = set()
        for sense in senses:
            if sense.sense_id in seen:
                continue
            unique_senses.append(sense)
            seen.add(sense.sense_id)
        for start in range(0, len(unique_senses), batch_size):
            batch = unique_senses[start:start + batch_size]
            embeddings = self.encode([sense.definition for sense in batch]).cpu()
            for sense, embedding in zip(batch, embeddings):
                self.definition_embeddings[sense.sense_id] = embedding

    def rank_senses(self, text: str, candidates: list[ElexisSense]) -> list[int]:
        text_embedding = self.encode([text]).cpu()
        candidate_embeddings = self.torch.stack([
            self.definition_embeddings[candidate.sense_id]
            for candidate in candidates
        ])
        scores = (text_embedding @ candidate_embeddings.T).squeeze(0).tolist()
        return [index for index, _score in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]

    def rank_senses_batch(
        self,
        texts: list[str],
        candidates_batch: list[list[ElexisSense]],
    ) -> list[list[int]]:
        text_embeddings = []
        for start in range(0, len(texts), self.batch_size):
            text_embeddings.extend(self.encode(texts[start:start + self.batch_size]).cpu())

        orders: list[list[int]] = []
        for text_embedding, candidates in zip(text_embeddings, candidates_batch):
            candidate_embeddings = self.torch.stack([
                self.definition_embeddings[candidate.sense_id]
                for candidate in candidates
            ])
            scores = (text_embedding.unsqueeze(0) @ candidate_embeddings.T).squeeze(0).tolist()
            orders.append([
                index
                for index, _score in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
            ])
        return orders


def run_distilled_wsd(
    *,
    annotations_jsonl: str | Path,
    senses: Iterable[ElexisSense],
    ranker: Ranker,
) -> list[WSDDecision]:
    return _run_distilled_wsd_sentences(_load_annotations(annotations_jsonl), senses, ranker)


def _run_distilled_wsd_sentences(
    sentences: Iterable[dict],
    senses: Iterable[ElexisSense],
    ranker: Ranker,
) -> list[WSDDecision]:
    targets = _collect_targets(sentences, senses)
    return _decide_targets(targets, ranker)


def _collect_targets(
    sentences: Iterable[dict],
    senses: Iterable[ElexisSense],
) -> list[WSDTarget]:
    index = ElexisSenseIndex(senses)
    targets: list[WSDTarget] = []

    for sentence in sentences:
        mwe_token_indices: set[int] = set()
        for mwe in sentence.get("mwes", []) or []:
            token_indices = [int(index) for index in mwe.get("token_indices", [])]
            mwe_token_indices.update(token_indices)
            targets.append(
                WSDTarget(
                    sentence_id=str(sentence.get("id") or ""),
                    target_type="mwe",
                    token_indices=token_indices,
                    target_text=str(mwe.get("text") or ""),
                    lemma=str(mwe.get("lemma") or ""),
                    upos=_mwe_upos(sentence, token_indices),
                    candidates=index.candidates_for_mwe(str(mwe.get("lemma") or "")),
                    marked_sentence=_mark_token_indices(sentence, token_indices),
                )
            )

        for token in sentence.get("tokens", []) or []:
            token_index = int(token.get("index") or 0)
            if token_index in mwe_token_indices or not _token_is_wsd_target(token):
                continue
            targets.append(
                WSDTarget(
                    sentence_id=str(sentence.get("id") or ""),
                    target_type="token",
                    token_indices=[token_index],
                    target_text=str(token.get("text") or ""),
                    lemma=str(token.get("lemma") or ""),
                    upos=str(token.get("upos") or ""),
                    candidates=index.candidates_for_token(str(token.get("lemma") or ""), str(token.get("upos") or "")),
                    marked_sentence=_mark_spans(sentence, [(int(token.get("start") or 0), int(token.get("end") or 0))]),
                )
            )
    return targets


def _decide_targets(targets: list[WSDTarget], ranker: Ranker) -> list[WSDDecision]:
    batch_orders: dict[int, list[int]] = {}
    multi_targets = [
        (index, target)
        for index, target in enumerate(targets)
        if len(target.candidates) > 1
    ]
    if multi_targets and hasattr(ranker, "rank_senses_batch"):
        orders = ranker.rank_senses_batch(
            [target.marked_sentence for _index, target in multi_targets],
            [target.candidates for _index, target in multi_targets],
        )
        batch_orders = {
            index: order
            for (index, _target), order in zip(multi_targets, orders)
        }

    return [
        _decide_target(target=target, ranker=ranker, order=batch_orders.get(index))
        for index, target in enumerate(targets)
    ]


def _unique_candidate_senses(targets: Iterable[WSDTarget]) -> list[ElexisSense]:
    unique: list[ElexisSense] = []
    seen: set[str] = set()
    for target in targets:
        for candidate in target.candidates:
            if candidate.sense_id in seen:
                continue
            unique.append(candidate)
            seen.add(candidate.sense_id)
    return unique


def run_distilled_wsd_from_paths(
    *,
    annotations_jsonl: str | Path,
    sense_repo: str | Path,
    model: str | Path,
    out_path: str | Path,
    sense_sheets: Sequence[str] | None = None,
    text_prefix: str | None = None,
) -> Path:
    senses = load_elexis_senses(sense_repo, sheets=sense_sheets)
    sentences = _load_annotations(annotations_jsonl)
    targets = _collect_targets(sentences, senses)
    ranker = CachedDistilledTransformerRanker(
        model,
        senses=_unique_candidate_senses(targets),
        text_prefix=text_prefix,
    )
    decisions = _decide_targets(targets, ranker)
    return write_wsd_tsv(decisions, out_path)


def build_distilled_demo(
    *,
    sample_id: str,
    annotations_jsonl: str | Path,
    sense_repo: str | Path,
    model: str | Path,
    resources: str | Path,
    sense_sheets: Sequence[str] | None = None,
    text_prefix: str | None = None,
) -> dict[str, object]:
    matching = [
        sentence
        for sentence in _load_annotations(annotations_jsonl)
        if str(sentence.get("id") or "") == sample_id
    ]
    if not matching:
        raise ValueError(f"Sample id not found in annotations: {sample_id}")
    sentence = matching[0]
    senses = load_elexis_senses(sense_repo, sheets=sense_sheets)
    ranker = DistilledTransformerRanker(model, text_prefix=text_prefix)
    decisions = _run_distilled_wsd_sentences([sentence], senses, ranker)
    return {
        "sample_id": sample_id,
        "sentence": str(sentence.get("text") or ""),
        "decisions": decisions_to_demo_rows(decisions, resources),
    }


def write_wsd_tsv(decisions: Iterable[WSDDecision], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WSD_TSV_COLUMNS, delimiter="\t")
        writer.writeheader()
        for decision in decisions:
            writer.writerow(
                {
                    "sentence_id": decision.sentence_id,
                    "target_type": decision.target_type,
                    "token_indices": ",".join(str(index) for index in decision.token_indices),
                    "target_text": decision.target_text,
                    "lemma": decision.lemma,
                    "upos": decision.upos,
                    "KBid": decision.selected_sense_id,
                    "NumberOfSenses": str(decision.sense_count),
                    "Possible": ";".join(decision.candidate_sense_ids),
                    "Explanation": decision.explanation,
                    "Origine": decision.origin,
                }
            )
    return out_path


def decisions_to_demo_rows(decisions: Iterable[WSDDecision], resources_dir: str | Path) -> list[dict[str, object]]:
    from .scoring import load_sentiment_table

    synset_dir = Path(resources_dir) / "synset_sentiment"
    sentiment_tables = {
        path.stem: load_sentiment_table(path)
        for path in sorted(synset_dir.glob("*.csv"))
    }
    rows: list[dict[str, object]] = []
    for decision in decisions:
        available = {
            table: values[decision.selected_sense_id]
            for table, values in sentiment_tables.items()
            if decision.selected_sense_id in values
        }
        rows.append(
            {
                "target_text": decision.target_text,
                "lemma": decision.lemma,
                "upos": decision.upos,
                "KBid": decision.selected_sense_id,
                "Origine": decision.origin,
                "Possible": decision.candidate_sense_ids,
                "sentiment": available,
                "missing_sentiment_tables": [
                    table for table in sentiment_tables if table not in available
                ],
            }
        )
    return rows


def _decide_target(
    *,
    target: WSDTarget,
    ranker: Ranker,
    order: list[int] | None = None,
) -> WSDDecision:
    candidates = target.candidates
    candidate_ids = [candidate.sense_id for candidate in candidates]
    if not candidates:
        return WSDDecision(
            sentence_id=target.sentence_id,
            target_type=target.target_type,
            token_indices=target.token_indices,
            target_text=target.target_text,
            lemma=target.lemma,
            upos=target.upos,
            selected_sense_id="NEW_SENSE",
            origin="None",
            sense_count=0,
            candidate_sense_ids=[],
            explanation="No candidates found in Elexis sense repository.",
        )
    if len(candidates) == 1:
        return WSDDecision(
            sentence_id=target.sentence_id,
            target_type=target.target_type,
            token_indices=target.token_indices,
            target_text=target.target_text,
            lemma=target.lemma,
            upos=target.upos,
            selected_sense_id=candidates[0].sense_id,
            origin="FIRST",
            sense_count=1,
            candidate_sense_ids=candidate_ids,
            explanation="Only one candidate sense.",
        )

    if order is None and hasattr(ranker, "rank_senses"):
        order = ranker.rank_senses(target.marked_sentence, candidates)
    elif order is None:
        definitions = [candidate.definition for candidate in candidates]
        order = ranker.rank(target.marked_sentence, definitions)
    selected_index = order[0] if order else 0
    return WSDDecision(
        sentence_id=target.sentence_id,
        target_type=target.target_type,
        token_indices=target.token_indices,
        target_text=target.target_text,
        lemma=target.lemma,
        upos=target.upos,
        selected_sense_id=candidates[selected_index].sense_id,
        origin="DISTILLED",
        sense_count=len(candidates),
        candidate_sense_ids=candidate_ids,
        explanation=f"Selected by distilled WSD ranker from {len(candidates)} candidates.",
    )


def _load_annotations(path: str | Path) -> list[dict]:
    sentences: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                sentences.append(json.loads(line))
    return sentences


def _token_is_wsd_target(token: dict) -> bool:
    lemma = str(token.get("lemma") or "")
    if str(token.get("upos") or "") not in CONTENT_UPOS:
        return False
    if not lemma or lemma in {"_", "*"} or lemma[0].isdigit():
        return False
    return str(token.get("named_entity") or "*") in ENTITY_ALLOWED_LABELS


def _mwe_upos(sentence: dict, token_indices: list[int]) -> str:
    wanted = set(token_indices)
    for token in sentence.get("tokens", []) or []:
        if int(token.get("index") or 0) in wanted and token.get("upos"):
            return str(token["upos"])
    return ""


def _mark_token_indices(sentence: dict, token_indices: list[int]) -> str:
    wanted = set(token_indices)
    spans = [
        (int(token.get("start") or 0), int(token.get("end") or 0))
        for token in sentence.get("tokens", []) or []
        if int(token.get("index") or 0) in wanted
    ]
    return _mark_spans(sentence, spans)


def _mark_spans(sentence: dict, spans: list[tuple[int, int]]) -> str:
    text = str(sentence.get("text") or "")
    pieces: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        pieces.append(text[cursor:start])
        pieces.append(f"**{text[start:end]}**")
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _resolve_text_prefix(model_path: Path, text_prefix: str | None) -> str:
    if text_prefix is not None:
        return text_prefix
    config_path = model_path / "training_config.json"
    if not config_path.exists():
        return "query: "
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "query: "
    return str(config.get("text_prefix") or "query: ")
