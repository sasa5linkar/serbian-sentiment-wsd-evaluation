"""Score synthetic sentences using saved WSD selections; no model is run."""
import argparse
import json
from pathlib import Path
from serbian_sentiment_wsd_eval.scoring import score_dataset
from serbian_sentiment_wsd_eval.evaluation import evaluate_scores
from serbian_sentiment_wsd_eval.csvio import read_delimited, write_dicts

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
args = parser.parse_args()
source = Path(__file__).resolve().parent
scores = score_dataset(sample_path=source / "sample.tsv", token_path=source / "tokens.tsv", wsd_path=source / "wsd.tsv", resources_dir=source / "resources", out_path=args.out / "scores.tsv", theta=0.1)
report = evaluate_scores(scores_path=scores, out_dir=args.out / "evaluation")
metrics = read_delimited(Path(report["metrics_path"]), delimiter=",")
selected = []
for row in metrics:
    config = row["config"]
    if config == "L0" or config == "S0-wsd" or (config.startswith("A") and (config.endswith("-wsd") or config.endswith("-avg2"))):
        label = "S0" if config == "S0-wsd" else config.replace("A", "S", 1).replace("-avg2", "-avg")
        selected.append({"report_config": label, **row})
write_dicts(args.out / "report_metrics.csv", selected, ["report_config", "config", "n", "accuracy", "macro_f1", "coverage_mean"])
print(json.dumps({"demo_only": True, "sentences_per_config": 4, "report_configurations": len(selected), "first_row": selected[0], "outputs": str(args.out)}, ensure_ascii=False, indent=2))
