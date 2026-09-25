"""Apply fixed reported thresholds to saved scores, without threshold search."""
import argparse
import json
from pathlib import Path
from serbian_sentiment_wsd_eval.csvio import read_delimited, write_dicts
from serbian_sentiment_wsd_eval.scoring import label_lemma_score, label_synset_score
from serbian_sentiment_wsd_eval.evaluation import evaluate_scores

parser = argparse.ArgumentParser()
parser.add_argument("--scores", type=Path, required=True)
parser.add_argument("--thresholds", type=Path, default=Path(__file__).resolve().parents[1] / "configs/reported_thresholds.json")
parser.add_argument("--out", type=Path, default=Path("outputs/reported_threshold_evaluation"))
args = parser.parse_args()
thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))["thresholds"]
rows = []
for row in read_delimited(args.scores, delimiter="\t"):
    config = row["config"]
    if config not in thresholds:
        continue
    theta = float(thresholds[config])
    row["theta"] = theta
    row["label"] = label_lemma_score(float(row["score"])) if config == "L0" else label_synset_score(float(row["score"]), theta=theta)
    rows.append(row)
if not rows:
    parser.error("No score configurations match the threshold file.")
saved = args.out / "scores.tsv"
if saved.resolve() == args.scores.resolve():
    parser.error("Choose a different output directory to preserve the input scores.")
write_dicts(saved, rows, list(rows[0]), delimiter="\t")
report = evaluate_scores(scores_path=saved, out_dir=args.out / "evaluation")
print(json.dumps(report, indent=2))
