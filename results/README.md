# Reported sentence-level sentiment results

[reported_metrics.csv](reported_metrics.csv) transcribes the 16 configurations in the dissertation's section 5.11 result table. Values retain the reported four-decimal precision. [reported_significance.csv](reported_significance.csv) transcribes the five predefined paired comparisons. These are recorded aggregate results, not recomputed predictions or results from the four-sentence demo.

The evaluated sample has **2,536 labelled sentences**: 908 negative, 989 neutral and 639 positive. Configuration names are mapped to the internal labels and fixed thresholds in [reported_thresholds.json](../configs/reported_thresholds.json).

For S7-avg the reported macro-F1 is 0.4510; for S7-wsd it is 0.4441. The reported macro-F1 differences against S0 remain significant after Holm correction; the comparisons S7-avg vs L0, S7-wsd or S6-avg do not. A stored p-value of 0.0000 is the source table's rounded display value, not an exact zero.

The threshold-sweep implementation selects the best macro-F1 on the supplied labelled data. Such selected results require this qualification and should not be described as an independent test estimate. No threshold search was performed while preparing this release.

Research per-sentence scores, final labels and the corpus text are not bundled here. The software can evaluate these files when supplied, or run the included synthetic example immediately. No private annotator worksheets are required by the saved-score workflow.
