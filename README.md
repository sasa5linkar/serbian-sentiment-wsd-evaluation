# Serbian Sentiment WSD Evaluation

Compare sentence-level Serbian sentiment from a lemma lexicon (L0), mapped SentiWordNet scores (S0), and enriched Serbian WordNet lexicons (S1–S7), using either a WSD-selected sense or a lemma-and-POS average.

This public package accompanies section 5.11 of Saša Petalinkar's doctoral dissertation. **Version 0.1.0** includes code, tests, the reported aggregate results and thresholds, and synthetic inputs with saved WSD decisions. The Python distribution remains `serbian-sentiment-wsd-eval`; the command is `sswe`.

## Install and run

Python 3.10+ is required; tested on Python 3.12. In an activated virtual environment:

```bash
git clone --branch v0.1.0 https://github.com/sasa5linkar/serbian-sentiment-wsd-evaluation.git
cd serbian-sentiment-wsd-evaluation
python -m pip install .
python examples/run_demo.py
```

The example scores four synthetic sentences and writes `outputs/demo/scores.tsv`, the full evaluation, and a **16-configuration** report table. Every selected configuration has N=4, accuracy=0.75, macro-F1=0.777777777778 and mean coverage=0.75. The fourth example is uncovered and falls back to neutral; its gold label is negative. The toy lexicons deliberately share scores so the example illustrates processing and coverage, not differences between trained resources.

No model download, training or API call is needed. The same workflow is available through the CLI:

```bash
sswe score --sample examples/sample.tsv --tokens examples/tokens.tsv --wsd examples/wsd.tsv --resources examples/resources --theta 0.1 --out outputs/demo/scores.tsv --coverage-out outputs/demo/missing_sentiment.tsv
sswe evaluate --scores outputs/demo/scores.tsv --out-dir outputs/demo/evaluation
```

## Configuration names

| Code | Report/dissertation | Meaning |
|---|---|---|
| L0 | L0 | Lemma polarity from Senti-pol-sr |
| S0-wsd | S0 | Mapped SentiWordNet scores for WSD-selected synsets |
| A1-wsd … A7-wsd | S1-wsd … S7-wsd | Enriched lexicons, selected sense |
| A1-avg2 … A7-avg2 | S1-avg … S7-avg | Average over all matching synsets for the target lemma and POS |

The old `A*-avg` output averages only WSD candidate IDs from Possible. It remains available for compatibility, but it is **not the main no-WSD control**. The main control is `avg2`. S1–S7 correspond to SVM/Bernoulli NB, AdaBoost/Bernoulli NB, RNN, transformer, BERTić, GPT2-Orao and Jerteh-355 lexicons, respectively.

## Use research inputs

Supply sentence IDs/text/labels, token lemmas and UPOS, saved WSD decisions, and sentiment resources. [docs/data_formats.md](docs/data_formats.md) describes the columns and an inventory join. Public S1–S7 CSV files provide **ID/POS/NEG** scores; avg2 additionally requires **Lemme/Vrsta** from a compatible, permitted SrpWN inventory.

Use [tools/enrich_lexicon.py](tools/enrich_lexicon.py) to join by ID:

```bash
python tools/enrich_lexicon.py --scores examples/polarity_only.csv --inventory examples/inventory.csv --out outputs/demo/joined_A7.csv
```

Preserve distinct meanings and all lemmas within each synset. Obtain the full inventory and L0 resource separately under their own terms. Full research sentence texts, individual annotator worksheets and trained WSD weights are not bundled in this release. The included input data are synthetic; the [results](results/README.md) directory contains the reported research aggregates.

## Scores, thresholds and coverage

For synsets, each contribution is **POS − NEG**. WSD uses the selected ID. avg2 first averages matching synsets for the same lemma and POS. Sentence scores average the covered contributions. Scores above θ are positive, below −θ negative, and otherwise neutral. No contributions gives score 0, neutral and zero coverage. L0 uses the sign of its lemma score.

[configs/reported_thresholds.json](configs/reported_thresholds.json) preserves the **reported per-configuration thresholds**. They differ from the program's generic θ=0.33 default. To apply them to saved scores without retuning:

```bash
python tools/evaluate_reported_thresholds.py --scores outputs/demo/scores.tsv --out outputs/fixed_thresholds_demo
```

With the demo file this produces demonstration metrics only. Use saved research scores with final labels to evaluate the research set. The reported study has **2,536 labelled sentences** (908 negative, 989 neutral, 639 positive); its summary table is provided in [results/reported_metrics.csv](results/reported_metrics.csv).

The optional `run-full-analysis` command produces coverage tables and performs a threshold sweep. It selects the best macro-F1 on the supplied labelled set; using that same set for selection and reporting is not an independent test estimate. The fixed-threshold helper above performs no sweep. See [docs/report_notes.md](docs/report_notes.md) for coverage definitions and the predefined paired comparisons.

## Optional WSD inference

Scoring accepts an existing WSD TSV and is independent of model availability. To generate new WSD selections locally, install `.[wsd]` and supply a compatible distilled checkpoint and inventory:

```bash
python -m pip install ".[wsd]"
sswe run-distilled-wsd --annotations-jsonl data/input/annotations.jsonl --sense-repo data/external/Elexis-WSD-Repo.xlsx --model models/wsd-distilled-mling --out outputs/wsd.tsv
```

[Serbian WSD Distillation](https://github.com/sasa5linkar/serbian-wsd-distillation) contains the related training/evaluation package. The separate `run-wsd` first-candidate integration requires an external serbian-agentic-wsd checkout; it is not needed for scoring or the bundled example.

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite this software. Cite the [sentiment lexicon article](https://doi.org/10.1108/EL-08-2024-0253) when using its lexicons, and the [IDA WSD article](https://doi.org/10.1177/1088467X261469292) when using that framework or data. The sentence-level results in this repository accompany the dissertation; they are not attributed to those articles as new results.

Code, documentation and synthetic examples use [Apache-2.0](LICENSE). External lexicons, corpora, annotations and pretrained weights retain their own terms. The license does not relicense those resources.

## Related resources

- [Sentiment lexicons S1–S7 and 24 Hugging Face classifiers](https://github.com/sasa5linkar/Serbian-WordNet-Sentiment-Lexicon-Analysis)
- [Synthetic sentiment evaluation set](https://github.com/sasa5linkar/SWN-synth-eval-set)
- [IDA WSD data and research code](https://github.com/te-sla/A-Semi-Automated-LLM-Based-Framework-for-Word-Sense-Disambiguation-in-Serbian)
- [WordNet expansion experiments](https://github.com/sasa5linkar/wordnet_autotranslate-)

- [srpskiwn Python wrapper](https://github.com/sasa5linkar/srpskiwn)

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## WSD checkpoint publication

The distilled WSD checkpoints are separate from the sentiment classifiers. Their [model cards and publication/download procedure](https://github.com/sasa5linkar/serbian-wsd-distillation/blob/main/docs/huggingface_models.md) are maintained with the distillation software. Weights are not yet released; the evaluator accepts a complete local checkpoint directory. Use the exact Hub revision from a publication receipt once one is available.

For a reusable application API and an offline example, see the [Serbian WordNet Sentiment Toolkit](https://github.com/sasa5linkar/serbian-wordnet-sentiment-toolkit). The toolkit reports zero coverage as `unscored`; this research evaluator preserves its neutral fallback.
