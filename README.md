# Serbian Sentiment WSD Evaluation

Compare sentence-level Serbian sentiment from a lemma lexicon (L0), mapped SentiWordNet scores (S0), and enriched Serbian WordNet lexicons (S1–S7), using either a WSD-selected sense or a lemma-and-POS average.

This public package accompanies section 5.11 of Saša Petalinkar's doctoral dissertation. **Version 0.1.0** includes code, tests, the reported aggregate results and thresholds, and synthetic inputs with saved WSD decisions. The Python distribution remains `serbian-sentiment-wsd-eval`; the command is `sswe`.

## Install and run

Python 3.10+ is required; tested on Python 3.12. In an activated virtual environment:

```bash
git clone https://github.com/sasa5linkar/serbian-sentiment-wsd-evaluation.git
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

## Optional Hugging Face example

[examples/load_hf_wsd.py](examples/load_hf_wsd.py) offers all three published rankers:

| Option | Model | When to choose it |
|---|---|---|
| `mling` (default) | [E5 Large](https://huggingface.co/Tanor/serbian-wsd-distilled-e5-large) | Starting choice: highest reported strict WSD accuracy of these three (72.6%). |
| `simple` | [MiniLM](https://huggingface.co/Tanor/serbian-wsd-distilled-minilm) | Smaller download: about 91 MB of weights; reported strict accuracy 68.5%. |
| `tesla` | [TeslaXLM](https://huggingface.co/Tanor/serbian-wsd-distilled-teslaxlm) | Comparison model; reported strict accuracy 51.3%, without improvement after distillation. |

These are the dissertation's reported results, not scores from this example. E5 and TeslaXLM each have about 2.24 GB of weights. The default only selects a model; it does not authorize a download.

List choices without installing model libraries:

~~~bash
python examples/load_hf_wsd.py --list
~~~

Run the small ranking example in a separate optional environment. This first call explicitly downloads MiniLM; replace `simple` with `mling` or `tesla` to choose another model:

~~~bash
uv run --no-project --with "sentence-transformers==5.5.0" --with "transformers==5.8.1" python examples/load_hf_wsd.py --model simple --download
~~~

Subsequent calls can omit `--download` to use the pinned cached copy. Use `--local-dir PATH` to read a complete existing checkpoint, or combine it with `--download` to download into that directory. Package installation by `uv` is separate from model download; `--no-project` keeps these example dependencies separate from the project's environment.

The example loads only local files after the explicit download, reads the saved `text_prefix` (including E5's `query: `), and lets SentenceTransformers load the saved pooling and tokenizer settings. It ranks two illustrative Serbian definitions and prints sense IDs with cosine scores. The sample is a usage demonstration, not an accuracy test or a full sentiment pipeline. A supplied local directory is used as-is; the pinned revision applies to Hub downloads and cached snapshots.

To obtain only the checkpoint for the existing application, without running the example:

~~~bash
uv run --no-project --with huggingface_hub python examples/load_hf_wsd.py --model mling --download --download-only --local-dir models/wsd-distilled-mling
~~~

Use the downloaded directory with the existing `sswe run-distilled-wsd --model` option. When selecting MiniLM or TeslaXLM for that evaluator, also pass `--text-prefix ""`.

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

The distilled WSD checkpoints are now public: [E5 Large (`mling`)](https://huggingface.co/Tanor/serbian-wsd-distilled-e5-large), [MiniLM (`simple`)](https://huggingface.co/Tanor/serbian-wsd-distilled-minilm), and [TeslaXLM (`tesla`)](https://huggingface.co/Tanor/serbian-wsd-distilled-teslaxlm). These rank candidate senses and are separate from the sentiment classifiers. Their [model cards, licenses, and pinned revisions](https://github.com/sasa5linkar/serbian-wsd-distillation/blob/main/docs/huggingface_models.md) are maintained with the distillation software.

To use the E5 checkpoint with the optional WSD command above, explicitly download it first:

```bash
hf download Tanor/serbian-wsd-distilled-e5-large --revision 749f694999b256039011acfb8f0a4b1b4f388c8c --local-dir models/wsd-distilled-mling
```

The evaluator reads the saved E5 `query: ` prefix from `training_config.json`. When selecting MiniLM or TeslaXLM instead, explicitly pass `--text-prefix ""` to keep their empty prefix. Keep the tokenizer and all configuration files in the downloaded directory. The compatible sense inventory remains a separate input; the model license is MIT and does not change the terms of that inventory.

For a reusable application API and an offline example, see the [Serbian WordNet Sentiment Toolkit](https://github.com/sasa5linkar/serbian-wordnet-sentiment-toolkit). The toolkit reports zero coverage as `unscored`; this research evaluator preserves its neutral fallback.
