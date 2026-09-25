# Inputs and saved outputs

All examples use UTF-8. TSV files have a header. Use stable, unique sentence IDs when joining labels, tokens and predictions.

| Input | Required or useful columns |
|---|---|
| Sentence sample | sample_id, sentence_text, sentiment_label |
| Tokens | sample_id, token_index, token_text, upos, lemma |
| Saved WSD TSV | sentence_id, target_type, token_indices, target_text, lemma, upos, KBid, NumberOfSenses, Possible, Explanation, Origine |
| Lemma polarity CSV | lemma, polarity (−1, 0, +1) |
| Synset sentiment CSV | ID, POS, NEG; plus Lemme and Vrsta for avg2 |

`sentiment_label` accepts positive/negative/neutral or 1/−1/0. `sswe evaluate --gold labels.tsv` can join an external sample_id/sentiment_label file to saved scores. Missing gold labels are excluded from evaluation; N is reported separately for each configuration. A missing resource match is a coverage gap, not a missing gold label.

In saved WSD data, Possible is a semicolon-separated candidate list; KBid is the selected synset ID, and token_indices identifies the token(s) in the target. NEW_SENSE has no existing lexical match. When preparing this file, preserve all eligible target units, including unmatched units, so coverage denominators remain meaningful. The bundled file demonstrates both matching and unmatched targets.

## Public lexicons and the inventory join

Get S1–S7 from the [sentiment lexicon repository](https://github.com/sasa5linkar/Serbian-WordNet-Sentiment-Lexicon-Analysis). Internally save them as A1.csv through A7.csv in a synset_sentiment directory. S0 contains mapped SentiWordNet scores and must be prepared from the appropriate source resource; it is not an extra CSV promised by the lexicon repository.

Public score-only CSVs are sufficient for selected-sense scoring, but not for avg2. Export a permitted full SrpWN inventory as **ID,Lemme,Vrsta**, one row per synset, with comma-separated lemmas quoted as a CSV field. Join on the exact synset ID with tools/enrich_lexicon.py; the tool preserves all score rows and reports unmatched IDs. Do not join on surface text or substitute the WSD candidate list for the full inventory.

The evaluator expects the historical sentiment-table POS convention: NOUN→n, VERB→v, ADJ→a, ADV→b. A source inventory using r for adverbs must be mapped to b in Vrsta before the join. This changes only the local POS label, not synset IDs. Leave POS and NEG score columns untouched. POS in the score CSV means positive sentiment, not part of speech.

`prepare-resources` is available when you have the complete original input layout: recnikPolariteta.csv (semicolon-separated lemma/positive-flag/negative-flag), swn30_sentiment.csv, and srbsentiwordnet_a1.csv through srbsentiwordnet_a7.csv. It preserves Lemme and Vrsta if present; it does not obtain them from an external inventory automatically.

## Outputs

`score` writes sample_id, sentence_text, gold_label, config, score, label, covered_units, total_units, coverage and theta. `evaluate` writes per-configuration metrics and confusion counts. The report set has 16 configurations (L0, S0 and two methods for each of S1–S7); raw score output includes legacy controls as well.

The [fixed-threshold helper](../tools/evaluate_reported_thresholds.py) retains only configurations with recorded thresholds, relabels saved continuous scores, and evaluates them. It does not run WSD or choose a new threshold.

## Full analysis

`run-full-analysis --reuse-wsd saved_wsd.tsv` reuses saved selections while exporting coverage, score, threshold-sweep and comparison tables. It also requires sentence/token/annotation inputs and a combined gold workbook. Individual annotator worksheets are not needed for the public scoring workflow: use final sample IDs and labels.

Full research sentence inputs, final per-sentence labels and predictions are separate from this initial software release. [results](../results/README.md) contains the reported aggregate tables; [examples](../examples) contains only synthetic demonstration data.
