# Report Notes

## Configuration Names

For dissertation/report tables, use the cleaner methodological names below
instead of the internal code labels:

| Internal code label | Report label |
| --- | --- |
| `L0` | `L0` |
| `S0-wsd` | `S0` |
| `A1-wsd` ... `A7-wsd` | `S1-wsd` ... `S7-wsd` |
| `A1-avg2` ... `A7-avg2` | `S1-avg` ... `S7-avg` |

Do not use the old `A*-avg` rows as the main report control. They average only
over Elexis/WSD candidate IDs from `Possible`, so they still depend on the WSD
candidate repository.

In the report, `avg` should mean the current `avg2` implementation: for each
target lemma and POS, average `POS - NEG` over all matching synsets in the
selected sentiment table.

POS mapping for `avg`:

| Target UPOS | Sentiment table `Vrsta` |
| --- | --- |
| `NOUN` | `n` |
| `VERB` | `v` |
| `ADJ` | `a` |
| `ADV` | `b` |

Recommended report set:

- `L0`: lemma baseline from `Senti-pol-sr`.
- `S0`: synset baseline using WSD-selected synsets and mapped English
  SentiWordNet values.
- `S1-wsd` ... `S7-wsd`: enriched Serbian WordNet sentiment tables with
  WSD-selected synsets.
- `S1-avg` ... `S7-avg`: no-WSD control using lemma+POS averages over all
  matching synsets in each sentiment table.

## Coverage Interpretation

Coverage should be discussed separately from accuracy and macro-F1 because it
does not mean the same thing for all methods.

| Method | Coverage definition |
| --- | --- |
| `L0` | Share of content lemmas found in `Senti-pol-sr`. |
| `wsd` | Share of WSD units where the sense repository provides candidates and the WSD-selected synset is present in the synset sentiment resource. |
| `avg` | Share of WSD units where the synset sentiment resource contains at least one synset with the same lemma and POS. |

This means `wsd` coverage depends on two resources: the WSD/sense repository and
the synset sentiment table. The `avg` control depends only on the synset
sentiment table. Higher `avg` coverage therefore mainly shows that the current
sense repository should be expanded or better aligned with the sentiment
resource.

## Significance Tests

Use predefined paired comparisons only:

- `S7-avg` vs `L0`
- `S7-avg` vs `S0`
- `S7-wsd` vs `S0`
- `S7-avg` vs `S7-wsd`
- `S7-avg` vs `S6-avg`

For accuracy, use exact McNemar tests over paired sentence predictions. For
macro-F1, use approximate randomization over paired predictions. Correct the
macro-F1 p-values with Holm-Bonferroni across the predefined comparisons.
