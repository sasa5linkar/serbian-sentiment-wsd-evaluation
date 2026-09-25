from __future__ import annotations

import sys
from pathlib import Path

from .constants import DEFAULT_AGENTIC_WSD_ROOT, DEFAULT_ANNOTATIONS, DEFAULT_SENSE_REPO, DEFAULT_WSD_OUT


def run_deterministic_wsd(
    *,
    annotations_jsonl: str | Path = DEFAULT_ANNOTATIONS,
    sense_repo: str | Path = DEFAULT_SENSE_REPO,
    agentic_wsd_root: str | Path = DEFAULT_AGENTIC_WSD_ROOT,
    out_path: str | Path = DEFAULT_WSD_OUT,
) -> Path:
    root = Path(agentic_wsd_root)
    src_path = root / "src"
    if not src_path.exists():
        raise RuntimeError(f"Cannot find serbian-agentic-wsd src path: {src_path}")
    src_string = str(src_path.resolve())
    if src_string not in sys.path:
        sys.path.insert(0, src_string)

    from serbian_agentic_wsd.io import load_annotations, write_wsd_tsv
    from serbian_agentic_wsd.wsd import run_wsd

    sentences = load_annotations(annotations_jsonl)
    decisions = run_wsd(sentences, sense_repo, use_agent=False)
    return write_wsd_tsv(decisions, out_path)
