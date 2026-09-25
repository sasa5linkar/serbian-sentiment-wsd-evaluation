from __future__ import annotations

from pathlib import Path

DEFAULT_SAMPLE = Path("data/input/sentences.tsv")
DEFAULT_TOKENS = Path("data/input/tokens.tsv")
DEFAULT_ANNOTATIONS = Path("data/input/annotations.jsonl")
DEFAULT_AGENTIC_WSD_ROOT = Path("external/serbian-agentic-wsd")
DEFAULT_SENSE_REPO = Path("data/external/srpwn_full_raw.csv")
DEFAULT_POLARITY_SOURCE = Path("data/external/recnikPolariteta.csv")
DEFAULT_SENTIMENT_SOURCE_DIR = Path("data/external/sentiment")
DEFAULT_ELEXIS_REPO = Path("data/external/Elexis-WSD-Repo.xlsx")
DEFAULT_DISTILLED_MODEL = Path("models/wsd-distilled-mling")

DEFAULT_RESOURCE_DIR = Path("data/resources")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_WSD_OUT = DEFAULT_OUTPUT_DIR / "wsd_first.tsv"
DEFAULT_WSD_DISTILLED_OUT = DEFAULT_OUTPUT_DIR / "wsd_distilled.tsv"
DEFAULT_SCORES_OUT = DEFAULT_OUTPUT_DIR / "sentence_scores.tsv"
DEFAULT_EVAL_DIR = DEFAULT_OUTPUT_DIR / "evaluation"

CONTENT_UPOS = {"NOUN", "VERB", "ADJ", "ADV"}

SYNSET_SOURCE_FILES = {
    "S0": "swn30_sentiment.csv",
    "A1": "srbsentiwordnet_a1.csv",
    "A2": "srbsentiwordnet_a2.csv",
    "A3": "srbsentiwordnet_a3.csv",
    "A4": "srbsentiwordnet_a4.csv",
    "A5": "srbsentiwordnet_a5.csv",
    "A6": "srbsentiwordnet_a6.csv",
    "A7": "srbsentiwordnet_a7.csv",
}

LABELS = ("positive", "negative", "neutral")
