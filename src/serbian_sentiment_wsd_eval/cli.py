from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .constants import (
    DEFAULT_AGENTIC_WSD_ROOT,
    DEFAULT_ANNOTATIONS,
    DEFAULT_DISTILLED_MODEL,
    DEFAULT_ELEXIS_REPO,
    DEFAULT_EVAL_DIR,
    DEFAULT_POLARITY_SOURCE,
    DEFAULT_RESOURCE_DIR,
    DEFAULT_SAMPLE,
    DEFAULT_SCORES_OUT,
    DEFAULT_SENSE_REPO,
    DEFAULT_SENTIMENT_SOURCE_DIR,
    DEFAULT_TOKENS,
    DEFAULT_WSD_DISTILLED_OUT,
    DEFAULT_WSD_OUT,
)
from .coverage import write_sentiment_coverage_report
from .distilled_wsd import build_distilled_demo, run_distilled_wsd_from_paths
from .evaluation import evaluate_scores
from .full_analysis import run_full_analysis
from .resources import prepare_resources
from .scoring import score_dataset
from .wsd import run_deterministic_wsd


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sswe",
        description="Serbian sentence-level sentiment WSD evaluation tools.",
    )
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare-resources", help="Normalize small input resources.")
    prepare.add_argument("--polarity-source", type=Path, default=DEFAULT_POLARITY_SOURCE)
    prepare.add_argument("--sentiment-source-dir", type=Path, default=DEFAULT_SENTIMENT_SOURCE_DIR)
    prepare.add_argument("--out-dir", type=Path, default=DEFAULT_RESOURCE_DIR)
    prepare.set_defaults(func=_cmd_prepare_resources)

    wsd = subparsers.add_parser("run-wsd", help="Run deterministic first-candidate WSD.")
    wsd.add_argument("--annotations-jsonl", type=Path, default=DEFAULT_ANNOTATIONS)
    wsd.add_argument("--sense-repo", type=Path, default=DEFAULT_SENSE_REPO)
    wsd.add_argument("--agentic-wsd-root", type=Path, default=DEFAULT_AGENTIC_WSD_ROOT)
    wsd.add_argument("--out", type=Path, default=DEFAULT_WSD_OUT)
    wsd.set_defaults(func=_cmd_run_wsd)

    distilled_wsd = subparsers.add_parser("run-distilled-wsd", help="Run local distilled Elexis WSD.")
    distilled_wsd.add_argument("--annotations-jsonl", type=Path, default=DEFAULT_ANNOTATIONS)
    distilled_wsd.add_argument("--sense-repo", type=Path, default=DEFAULT_ELEXIS_REPO)
    distilled_wsd.add_argument("--model", type=Path, default=DEFAULT_DISTILLED_MODEL)
    distilled_wsd.add_argument("--out", type=Path, default=DEFAULT_WSD_DISTILLED_OUT)
    distilled_wsd.add_argument("--sense-sheet", action="append", default=None)
    distilled_wsd.add_argument("--text-prefix", default=None)
    distilled_wsd.set_defaults(func=_cmd_run_distilled_wsd)

    score = subparsers.add_parser("score", help="Score sentences for all configured resources.")
    score.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    score.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    score.add_argument("--wsd", type=Path, default=DEFAULT_WSD_OUT)
    score.add_argument("--resources", type=Path, default=DEFAULT_RESOURCE_DIR)
    score.add_argument("--out", type=Path, default=DEFAULT_SCORES_OUT)
    score.add_argument("--coverage-out", type=Path, default=None)
    score.add_argument("--theta", type=float, default=0.33)
    score.set_defaults(func=_cmd_score)

    demo = subparsers.add_parser("demo-distilled", help="Run distilled WSD and sentiment lookup for one sample.")
    demo.add_argument("--sample-id", required=True)
    demo.add_argument("--annotations-jsonl", type=Path, default=DEFAULT_ANNOTATIONS)
    demo.add_argument("--sense-repo", type=Path, default=DEFAULT_ELEXIS_REPO)
    demo.add_argument("--model", type=Path, default=DEFAULT_DISTILLED_MODEL)
    demo.add_argument("--resources", type=Path, default=DEFAULT_RESOURCE_DIR)
    demo.add_argument("--sense-sheet", action="append", default=None)
    demo.add_argument("--text-prefix", default=None)
    demo.set_defaults(func=_cmd_demo_distilled)

    full = subparsers.add_parser("run-full-analysis", help="Run full gold evaluation and coverage analysis.")
    full.add_argument("--gold-workbook", type=Path, default=Path("outputs/annotation_merge/combined_sentiment_annotations.xlsx"))
    full.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    full.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    full.add_argument("--annotations-jsonl", type=Path, default=DEFAULT_ANNOTATIONS)
    full.add_argument("--sense-repo", type=Path, default=DEFAULT_ELEXIS_REPO)
    full.add_argument("--model", type=Path, default=DEFAULT_DISTILLED_MODEL)
    full.add_argument("--resources", type=Path, default=DEFAULT_RESOURCE_DIR)
    full.add_argument("--out-dir", type=Path, default=Path("outputs/full_distilled_analysis"))
    full.add_argument("--theta", type=float, default=0.33)
    full.add_argument("--theta-sweep", default="0.0,0.1,0.2,0.33,0.5")
    full.add_argument("--reuse-wsd", type=Path, default=None)
    full.add_argument("--sense-sheet", action="append", default=None)
    full.add_argument("--text-prefix", default=None)
    full.set_defaults(func=_cmd_run_full_analysis)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate score labels when gold labels exist.")
    evaluate.add_argument("--scores", type=Path, default=DEFAULT_SCORES_OUT)
    evaluate.add_argument("--gold", type=Path, default=None)
    evaluate.add_argument("--out-dir", type=Path, default=DEFAULT_EVAL_DIR)
    evaluate.set_defaults(func=_cmd_evaluate)
    return parser


def _cmd_prepare_resources(args: argparse.Namespace) -> int:
    report = prepare_resources(
        polarity_source=args.polarity_source,
        sentiment_source_dir=args.sentiment_source_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _cmd_run_wsd(args: argparse.Namespace) -> int:
    out = run_deterministic_wsd(
        annotations_jsonl=args.annotations_jsonl,
        sense_repo=args.sense_repo,
        agentic_wsd_root=args.agentic_wsd_root,
        out_path=args.out,
    )
    print(out)
    return 0


def _cmd_run_distilled_wsd(args: argparse.Namespace) -> int:
    out = run_distilled_wsd_from_paths(
        annotations_jsonl=args.annotations_jsonl,
        sense_repo=args.sense_repo,
        model=args.model,
        out_path=args.out,
        sense_sheets=args.sense_sheet,
        text_prefix=args.text_prefix,
    )
    print(out)
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    out = score_dataset(
        sample_path=args.sample,
        token_path=args.tokens,
        wsd_path=args.wsd,
        resources_dir=args.resources,
        out_path=args.out,
        theta=args.theta,
    )
    print(out)
    if args.coverage_out:
        report = write_sentiment_coverage_report(args.wsd, args.resources, args.coverage_out)
        print(json.dumps({"coverage_report": str(args.coverage_out), **report}, ensure_ascii=False))
    return 0


def _cmd_demo_distilled(args: argparse.Namespace) -> int:
    payload = build_distilled_demo(
        sample_id=args.sample_id,
        annotations_jsonl=args.annotations_jsonl,
        sense_repo=args.sense_repo,
        model=args.model,
        resources=args.resources,
        sense_sheets=args.sense_sheet,
        text_prefix=args.text_prefix,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_run_full_analysis(args: argparse.Namespace) -> int:
    result = run_full_analysis(
        gold_workbook=args.gold_workbook,
        sample=args.sample,
        tokens=args.tokens,
        annotations_jsonl=args.annotations_jsonl,
        sense_repo=args.sense_repo,
        model=args.model,
        resources=args.resources,
        out_dir=args.out_dir,
        theta=args.theta,
        theta_sweep=_parse_theta_sweep(args.theta_sweep),
        reuse_wsd=args.reuse_wsd,
        sense_sheets=args.sense_sheet,
        text_prefix=args.text_prefix,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _parse_theta_sweep(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _cmd_evaluate(args: argparse.Namespace) -> int:
    report = evaluate_scores(scores_path=args.scores, gold_path=args.gold, out_dir=args.out_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
