# Data preparation and labeling

Start with the [project guide](../README.md) for the NIST download links and Google Drive folder layout. The labeling and preparation notebooks use the shared [`scripts/` package](../scripts/).

## Run in Colab

1. Open [Data_Overview.ipynb](../notebooks/Data%20Labeling/Data_Overview.ipynb). Clone the repository, mount Drive, inspect daily file counts and merge the extracted daily CSVs. Inputs are under `MyDrive/NIST/`; outputs are under `MyDrive/NIST_processed/`.
2. Open [Data_Labeling.ipynb](../notebooks/Data%20Labeling/Data_Labeling.ipynb). Set `SITE = "canopy"` or `"ground"` and `YEAR = 2015`, `2016`, `2017` or `2018`. Run cells in order to inspect every stage of the thesis methodology.
3. Inspect the final labels and export to `MyDrive/NIST_Labels/`. Final outputs are protected unless `OVERWRITE_FINAL` is enabled. Intermediate checkpoints are optional.

The detailed notebook calls named stages of `run_labeling`; the full pipeline and interactive walkthrough share the same implementation. Feature selection, cleaning, performance ratios, string analysis and inverter analysis remain separately explained and displayed. Reporting functions are in `scripts/reporting.py`.

The canonical, walkthrough, and batch notebooks default to `REVISED_INVERTER_POLICY`. Select `ORIGINAL_INVERTER_POLICY` to use the historical inverter rules.

## Quality checks

`scripts/qa.py` provides:

- `check_year_coverage(df, year)`: rejects wrong-year, missing or invalid timestamp indexes; reports expected/actual counts, duplicates, missing minutes and off-grid timestamps. Its reference calendar uses the index timezone. Partial years are reported rather than silently filled or rejected.
- `check_sentinels(df)`: fails on numeric or numeric-string fill codes such as `-999` and `-9999`. The explicit sentinel list is configurable; legitimate negative temperatures/status values and NaNs are not broadly rejected.
- `cross_check_with_inverter_flags(df)`: compares physics labels with inverter reports and separates power-only, status-only and overlapping shutdown flags using correctly grouped boolean expressions.

The notebook runs input QA before processing and displays the inverter cross-check after labeling. Missing minutes are reported only when they exist. No observations are imputed or removed automatically.

## Historical regression: all eight datasets

Saved outputs from the original annual notebooks were extracted into [regression/legacy_counts.json](../regression/legacy_counts.json). Each count has a source notebook, output-cell index and SHA-256 hash (with CRLF normalized to LF for Windows/Colab compatibility). The three label families have matching historical row totals within each dataset. There are 125 saved count records across all eight site/year combinations.

The checked-in [regression table](../regression/regression_report.md) is explicitly **NOT RUN** until actual annual data is supplied. Synthetic tests do not establish agreement with historical outputs.

Prepare all eight merged inputs, then run this in Colab after mounting Drive:

```python
%cd "/content/pv-timeseries-fault-forecasting"
!python -m scripts.regression --output /content/drive/MyDrive/NIST_regression
```

The checker processes one dataset at a time and always uses `ORIGINAL_INVERTER_POLICY` for a like-for-like comparison with the annual notebooks. It writes:

| File | Contents |
|---|---|
| `regression_table.csv` | Historical count, actual count, delta, status and explanation for every label category |
| `regression_report.md` | The complete comparison table for the PR |
| `coverage.csv` | Coverage diagnostics for successfully validated inputs |
| `result.json` | Pass/fail result and hashes of the baseline, table and script files |

An absent label category is compared as zero. New categories are included. Missing input files, sentinel failures and pipeline errors fail the run; the checker continues to report the remaining datasets.

Ground 2017 is a documented exception: its saved outputs contain 2016 timestamps, although its current loading cell names a 2017 file. Its count differences are marked `EXPECTED_WRONG_YEAR_BASELINE` only after the actual input passes 2017 validation. This exception does not establish correctness against a genuine historical 2017 run.

Every other nonzero difference starts as `UNEXPLAINED` and fails regression. Investigate it before accepting it. If it is intentional, supply a JSON list with the exact site, year, label family, label, historical count, actual count and explanation:

```json
[
  {
    "site": "canopy",
    "year": 2015,
    "label_type": "label_power",
    "label": "NORMAL",
    "historical_count": 129373,
    "actual_count": 129372,
    "explanation": "Illustrative format only: replace with the verified cause of this exact difference."
  }
]
```

The example is not an approved exception or an observed new result. Pass your reviewed file with `--reviews /path/to/reviews.json`. A waiver applies only to those exact counts, so later count changes require fresh investigation. Save the review file with the PR.

Copy the generated CSV and Markdown report into the repository's [`regression/` directory](../regression/) for the PR, replacing the pending artifacts. Include the coverage report and result manifest as validation evidence. The full dataset regression has not been run merely because a report template or baseline exists.

## Replace the legacy notebooks after a passing check

The old annual notebooks, `NIST_labeling.ipynb`, and the two superseded `Site_Year_Labeling*` notebooks remain temporarily available while regression is pending. Do not delete the historical evidence before validating the new workflow.

After a passing run, execute:

```python
!python -m scripts.cleanup_legacy /content/drive/MyDrive/NIST_regression
```

This checks all eight datasets and all three label families, requires no unresolved differences, and verifies that the scripts, baseline and comparison table have not changed since the run. It then removes the eight original annual notebooks, `NIST_labeling.ipynb`, and the two superseded labeling notebooks. It retains `Data_Labeling.ipynb`, `Data_Overview.ipynb`, the shared package and the extracted baseline evidence.

In Colab these deletions affect the runtime checkout only. Include the corresponding deletions in the GitHub PR after reviewing the regression table. The acceptance target is one active labeling notebook plus the preparation notebook, with no duplicated labeling implementation.

## Tests

From the repository root, run:

```text
pytest -q
```

Tests use synthetic measurements and the saved notebook outputs; annual Drive CSVs are not needed. They cover wrong-year rejection, timezone-aware normal/leap-year coverage, real gaps and duplicates, sentinel handling, boolean precedence, stage/full-pipeline equivalence, and historical count extraction/comparison.

To re-extract baselines deliberately while the original notebooks are present:

```text
python -m scripts.regression --extract-baselines
```

Keep baseline changes reviewable. Re-extraction must not be used to hide a regression mismatch.
