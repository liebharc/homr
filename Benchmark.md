# Benchmarks

## External Benchmarks

Only one known public benchmark covers homr:

- [notes2tone](https://github.com/dasrinkana/notes2tone) — compares homr, oemer, and audiveris on a small set of scores.

The academic OMR community generally only benchmarks tools that have associated publications. Because homr has no paper, it has never been included in any research benchmark. This gap is the main motivation for maintaining an internal one.

## Internal Benchmark

### Why a Custom Benchmark

Comparing OMR output is harder than it looks. Music can be encoded in many ways — different file formats (MusicXML, Humdrum kern), different chord orderings, different representations of accidentals or ties — and two encodings that sound identical may not be textually identical. A naive string diff is therefore meaningless.

homr already has parsers that were built for training-data normalisation. These parsers convert both kern and MusicXML into the same internal token vocabulary (`EncodedSymbol`: rhythm, pitch, lift, articulation, slur). By routing all tool output through these parsers before comparison, every tool can write in whatever format it prefers and will be evaluated on equal terms.

### Metric: OMR-NED

The benchmark computes **Normalised Edit Distance (NED)** at the token level:

```
OMR-NED = edit_distance(expected_tokens, actual_tokens) / (N_ref + N_pred)
```

Tokens are compared independently for the upper and lower staves. The final NED is the sum of both edit distances divided by the total number of tokens in both the reference and the prediction. A NED of 0 % means perfect recognition; values approach 100 % as the output diverges from the reference.

The benchmark also reports five component-level NEDs — rhythm, pitch, lift (accidentals), articulation, and slur — by computing the edit distance on each field independently. This makes it possible to see, for example, that pitch accuracy is high while articulation is poor.

Detailed alignment data (which specific tokens were deleted, inserted, or substituted, and with what) is written to a SQLite database. This enables queries such as:

- How often was a particular clef missing?
- How often was pitch A5 confused with A4?
- Which symbol has the highest error rate overall?

### Why We Use Our Own Parsers

Rather than comparing raw files, the benchmark parses both the expected kern and the actual tool output into normalised token sequences before computing NED. This has one important benefit: **tools are not penalised for output format choices**. A tool that writes Humdrum kern is not at a disadvantage relative to one that writes MusicXML, because both pass through the same normalisation step.

External format converters (hum2xml, music21) exist but introduce their own errors when handling polyphonic music. Running them as a round-trip baseline — converting the reference kern to MusicXML and back, then scoring — reveals these conversion errors directly:

```
poetry run python -m validation.polish-scores --tool hum2xml
poetry run python -m validation.polish-scores --tool music21
```

Any NED above zero here is a conversion artefact, not a recognition error. You can confirm this by inspecting individual diffs with `show_diff.py`.

We also provide `--kern-parser music21` as an experimental alternative that parses the kern ground truth via music21 instead of our own parser. In testing on `btrkeks/polish-scores` it showed two apparent limitations. First, music21's Humdrum parser did not produce any slur spanners from kern `()`/`[]` notation -- the score's spanner bundle was empty in every sample -- so slur annotations were lost entirely from the ground truth. Second, some files with spine-split (`*^`/`*v`) constructs caused part content to be truncated or misassigned; sample 9 in that dataset has 13 such constructs and music21 produced roughly 65 fewer tokens in the bass part than our own parser. These may be limitations of music21's Humdrum support rather than fundamental problems, but we have not investigated further.

A music21-based MusicXML parser is also available for comparison (`compare_parsers.py --dataset lieder`). Several music21 behaviours cause systematic differences from the native parser: it includes articulations that are marked invisible in the score (`print-object="no"`); it silently discards arpeggio markings; it interprets tremolo rhythm values as subdivided notes rather than the written note value; and it collapses concurrent slurs into one when they share the same start and end notes.

### Risks of Using Our Own Parsers

The parsers were built to support homr's training vocabulary. This creates two risks:

**1. Systematic bias in homr's favour.**
If the parser silently drops a musical construct it does not support (an ornament, a complex tuplet, an unusual clef), that construct disappears from both the expected and actual token sequences. homr is never penalised for missing it because the benchmark never required it. Any other tool that does recognise the construct may also have it normalised away, receiving no credit.

**2. Silent parser bugs.**
A bug that causes tokens to be merged, reordered, or dropped tends to reduce the apparent edit distance rather than inflate it. The result looks like an artificially good score.

### Mitigations

Several properties of the design limit the impact of these risks:

- **homr writes MusicXML end-to-end.** When homr is under test it produces MusicXML from an image, then that file is parsed by the same tokeniser used on the reference. This makes it a genuine end-to-end test rather than a round-trip of the training data.

- **The parser produces symbols outside homr's vocabulary.** The tokeniser is not capped at what homr can output. If the reference contains a symbol homr was not trained on, it will appear in the expected sequence and be counted as a deletion, which correctly penalises homr.

- **The benchmark code can be reviewed and tested independently.** The comparison logic lives in `validation/ned_benchmark.py` and does not depend on any homr model code. Bugs there can be found through code review or by running the converter baselines on known-correct scores.

### Comparison with the transcoda benchmark

The transcoda paper reports an OMR-NED on the same `btrkeks/polish-scores` dataset using the same denominator formula. The remaining difference is the distance function: we apply Levenshtein distance to a flat sequence of symbolic tokens, while transcoda uses the [musicdiff](https://github.com/gregchapman-dev/musicdiff) library, which performs a semantic tree comparison via music21. Both measure edit distance between the reference and prediction but at a different level of abstraction, so the scores are not directly comparable.

Both approaches also pay a conversion cost before comparison. Our parsers convert everything to `EncodedSymbol` sequences; musicdiff converts everything to music21 object graphs via converter21. Neither is free of artefacts. The native parser's conversion errors are at least directly measurable — running hum2xml or music21 as a baseline tool surfaces them as NED above zero. musicdiff's conversion errors are harder to audit because they are inherited from a third-party library with no equivalent baseline.

## Setup

### HuggingFace Authentication (SMB dataset)

The SMB dataset (`PRAIG/SMB`) is access-restricted. Before running any SMB benchmark you need to:

1. Request access at https://huggingface.co/datasets/PRAIG/SMB
2. Create a token at https://huggingface.co/settings/tokens
3. Create a `.env` file in the project root:

```
HF_TOKEN=your_token_here
```

## How to Run

```bash
# Format-converter baseline (no images required)
poetry run python -m validation.smb --tool music21

# Full OMR benchmark (images extracted automatically from the dataset)
poetry run python -m validation.smb --tool homr

# Continue an interrupted run
poetry run python -m validation.smb --tool homr --continue

# transcoda on the first 10 samples of polish scores
poetry run python -m validation.polish-scores --tool transcoda --limit 10

# Generate an HTML report from a finished run
poetry run python -m validation.report smb_homr.db
```

Output databases are named `<dataset>_<tool>.db` by default (e.g. `smb_homr.db`, `polish-scores_transcoda.db`). Pass `--output <path>` to override.

## Investigating Results

The `show_diff.py` tool reads a benchmark database and shows a token-level diff for a single sample:

```bash
# List all samples with their NED scores
poetry run python validation/show_diff.py smb_homr.db --list

# Show the diff for sample at index 3
poetry run python validation/show_diff.py smb_homr.db --sample 3

# Export the expected kern and actual MusicXML to a directory
poetry run python validation/show_diff.py smb_homr.db --sample 3 --output /tmp/diff_out
```

The display uses `<` for deleted tokens (in reference, missing from output), `>` for inserted tokens (hallucinated), and `!` for substitutions.

## Benchmark results

All results are mean OMR-NED in % over the samples where the tool produced output. Means are only comparable between tools when coverage is similar.

## native parser

| dataset                         | homr           | transcoda     | oemer        | SMT with smt-fp-grandstaff weights |
| ------------------------------- | -------------- | ------------- | ------------ | ---------------------------------- |
| polish-scores                   | 24.8 (110/112) | 23.8 (98/112) |              | 78 (112/112)                       |
| smb                             | 23.4 (661/685) |               |              |                                    |
| smb, just the first 10 examples | 24.2 (10/10)   | 50.1 (9/10)   | 38.0 (13/15) | 75.6 (11/20)                       |

## musicdiff

| dataset                         | homr           | transcoda     | oemer       | SMT with smt-fp-grandstaff weights |
| ------------------------------- | -------------- | ------------- | ----------- | ---------------------------------- |
| polish-scores                   | 44.2 (110/112) | 52.6 (98/112) |             | musicdiff crashes                  |
| smb                             | 77.7 (661/685) |               |             |                                    |
| smb, just the first 10 examples | 75.5 (10/10)   | 101.9 (9/10)  | 90.8(13/15) | 104.8 (11/20)                      |
