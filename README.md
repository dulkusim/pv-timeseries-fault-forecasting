# PV Time-Series Fault Forecasting

Diploma thesis project exploring photovoltaic (PV) fault forecasting and degradation trends using electrical and weather time series. This guide covers data preparation and rule-based labeling for the NIST Canopy and Ground installations, using one-minute measurements from 2015–2018.

The workflow runs in **Google Colab**, with input data and results stored in **Google Drive**. You do not need a local Python installation or a local copy of the datasets to run the notebooks.

## Workflow at a glance

1. Download the NIST one-minute data.
2. Extract and organize the daily CSVs in Google Drive.
3. Run **Data Overview** to inspect file counts and merge daily measurements into annual CSVs.
4. Run the **Labeling Walkthrough** to inspect each processing stage and export labeled data.
5. Optionally use the **batch runner** to process several site/year combinations.

## 1. Get the data

Open the [NIST Photovoltaic Data portal](https://pvdata.nist.gov/) and use **Bulk Download** to obtain the **one-minute (`onemin`) measurements** for:

| Installation | Years used in this project |
|---|---|
| Canopy | 2015, 2016, 2017, 2018 |
| Ground | 2015, 2016, 2017, 2018 |

For a first run, start with **Canopy 2015**. You can add the other datasets later. Roof, weather-station files, images and one-second measurements are not required by the current Canopy/Ground labeling pipeline.

The [NIST testbed page](https://www.nist.gov/el/beed/heat-transfer-alternative-energy-systems/photovoltaic-testbeds) provides background and links to file documentation and the data dictionary. These are useful when interpreting sensor names, units and inverter measurements.

## 2. Organize the files in Google Drive

In **My Drive**, create a folder named `NIST`. Extract the downloaded archives and place the daily CSVs inside a dataset/year folder, with one subfolder per month. Upload the extracted folders to Drive, or extract the archives there using your preferred method. The notebooks read extracted CSV files; they do not unpack archives.

The expected layout is:

```text
My Drive/
└── NIST/
    ├── onemin-Canopy-2015/
    │   ├── 01/
    │   │   ├── <daily-file-1>.csv
    │   │   └── <daily-file-2>.csv
    │   ├── 02/
    │   └── ...
    ├── onemin-Canopy-2016/
    ├── onemin-Canopy-2017/
    ├── onemin-Canopy-2018/
    ├── onemin-Ground-2015/
    ├── onemin-Ground-2016/
    ├── onemin-Ground-2017/
    └── onemin-Ground-2018/
```

The daily filenames above are placeholders; keep the downloaded CSV filenames. Use the dataset folder names shown, including capitalization, because those names become the merged input filenames. Avoid adding an extra archive folder between the dataset folder and its monthly folders.

For example, a daily file inside `My Drive/NIST/onemin-Canopy-2015/01/` is accessible in Colab under:

```text
/content/drive/MyDrive/NIST/onemin-Canopy-2015/01/<daily-file>.csv
```

Keep only the datasets you intend to merge under `NIST/`: Data Overview processes every dataset folder it finds there. Individual annual CSVs can reach hundreds of megabytes, so allow Drive space for the extracted inputs and generated outputs.

## 3. Prepare annual CSVs in Colab

Open [Data_Overview.ipynb](notebooks/Data%20Labeling/Data_Overview.ipynb), or [launch it in Google Colab](https://colab.research.google.com/github/dulkusim/pv-timeseries-fault-forecasting/blob/main/notebooks/Data%20Labeling/Data_Overview.ipynb).

Run the setup cells in order. They clone this repository into the Colab runtime, import the shared scripts and mount Google Drive. Authorize Drive access when Colab prompts you. The repository supplies the code; your Drive supplies the data.

**Check the output folder before running the merge cell.** The current overview notebook sets `output_path` to `NIST_Labeling`, while the labeling loader expects `NIST_processed`. Use the following paths in the overview's Drive/path cell to connect the two steps:

```python
os.environ["DATA_ROOT"] = "/content/drive/MyDrive"
root = get_data_root()
base_path = root / "NIST"
output_path = root / "NIST_processed"
```

The notebook counts daily CSVs in each monthly folder, then merges each dataset in sorted month/filename order. It creates outputs such as:

```text
My Drive/NIST_processed/onemin-Canopy-2015.csv
My Drive/NIST_processed/onemin-Ground-2015.csv
```

The output directory is created automatically. Rerunning the merge cell **overwrites the corresponding annual CSVs**. If you already merged into `NIST_Labeling`, you can move those annual CSVs into `NIST_processed` instead of merging again.

File counts help identify incomplete downloads, but do not verify all timestamps or file contents. Timestamp checks follow in the labeling walkthrough.

## 4. Inspect and label one dataset

Open [Site_Year_Labeling_Walkthrough.ipynb](notebooks/Data%20Labeling/Site_Year_Labeling_Walkthrough.ipynb), or [launch the walkthrough in Colab](https://colab.research.google.com/github/dulkusim/pv-timeseries-fault-forecasting/blob/main/notebooks/Data%20Labeling/Site_Year_Labeling_Walkthrough.ipynb).

Run its setup and Drive cells, then select your dataset:

```python
SITE_NAME = "canopy"  # "canopy" or "ground"
YEAR = 2015           # 2015, 2016, 2017 or 2018
SAVE_CHECKPOINTS = False
OVERWRITE_FINAL = False
```

Run the remaining cells from top to bottom. The notebook explains and displays each stage:

| Stage | What you inspect or calculate |
|---|---|
| Data Inspection | Sample records, types, statistics, shape, size, timestamp continuity and annual/monthly coverage |
| Feature Selection | Secondary and redundant feature removal, faulty RTD diagnostics, site-specific RTD removal and electrical correlations |
| Feature Engineering | Missing-value reports, the shared column-removal policy, physical bounds and operating mask |
| Performance Ratio | POA irradiance, module temperature, expected power, rolling PR features and power-performance labels |
| String Level Faults | Circuit statistics, voltage proxies, irradiance normalization, fitted baselines and string labels |
| Inverter Level Faults | Efficiency, DC discrepancy, shutdown indicators, inverter labels and comparisons with reported fault codes |
| Final Validation | Row/timestamp preservation, label summaries, cross-tabulations and export |

The calculations use the shared scripts rather than repeated annual notebook code. To inspect another dataset, change the site/year and rerun from the configuration cell onward. Restarting the runtime gives you a clean session when needed.

### Choose the inverter policy deliberately

The walkthrough currently selects `REVISED_INVERTER_POLICY`. The batch notebook selects `ORIGINAL_INVERTER_POLICY`. Set the same policy in both if you want comparable results.

| Rule | Original | Revised |
|---|---|---|
| Power-based shutdown | AC power below 0.05 kW | DC power below 1 kW |
| Invalid shutdown status | Missing or nonpositive | Also above 5 |
| Minimum DC input for efficiency-fault evaluation | Above 1 kW | Above 5 kW |
| Active reported fault code | Positive | Nonzero and nonmissing |

The walkthrough explains the remaining thresholds and rule precedence. These labels are heuristic supervision signals, not independently verified fault diagnoses. String baselines are fitted on the selected full year for exploratory labeling; forecasting evaluation should fit baselines on training data only.

## 5. Find your results

For Canopy 2015, the walkthrough writes:

```text
My Drive/
├── NIST/                                  # Extracted raw daily CSVs
├── NIST_processed/                        # Merged annual inputs
│   └── onemin-Canopy-2015.csv
├── NIST_Labels/                           # Final results
│   ├── canopy_2015.csv
│   ├── canopy_2015_label_summary.csv
│   ├── canopy_2015_stage_audit.csv
│   └── canopy_2015_methodology.json
└── NIST_labeling_stages/                   # Only when checkpoints are enabled
```

The final dataset contains the retained measurements, engineered features, a `TIMESTAMP` index column and three separate labels: `label_power`, `label_string` and `label_inverter`. The methodology JSON records the selected configuration, policy, fitted baselines and repository commit.

Final exports are protected against accidental replacement. Set `OVERWRITE_FINAL = True` when you intend to replace them. Optional checkpoints use `save_stage`, which always rejects an existing checkpoint file.

## 6. Optional: process several datasets

Once you have inspected the walkthrough, use [Site_Year_Labeling.ipynb](notebooks/Data%20Labeling/Site_Year_Labeling.ipynb) ([open in Colab](https://colab.research.google.com/github/dulkusim/pv-timeseries-fault-forecasting/blob/main/notebooks/Data%20Labeling/Site_Year_Labeling.ipynb)) for batch processing.

```python
SITE_NAMES = ("canopy", "ground")
YEARS = (2015, 2016, 2017, 2018)
OVERWRITE = False
```

All selected merged input files must exist. The notebook processes one site/year at a time, exports the final CSVs to `NIST_Labels/`, and writes combined `label_summary.csv` and `processing_summary.csv` reports. Its final dataset filenames are shared with the walkthrough, so check the selected policy before replacing earlier results.

## Common setup issues

| Problem | What to check |
|---|---|
| Raw dataset directory not found | Confirm that `NIST/` is directly under My Drive and Drive is mounted. |
| Merged input file not found | Check that the CSV is in `NIST_processed/`, with the exact site/year filename. See the overview output-path setting above. |
| No daily CSV files found | Check that archives are extracted and daily CSVs are inside monthly folders, without an extra nesting level. |
| Clone fails or scripts are missing | Confirm access to the public repository. Restart the runtime to clone updated scripts. |
| An output already exists | Choose whether to retain it or enable the appropriate final-output overwrite setting. Checkpoints have separate protection. |
| Colab runs out of RAM | A CSV's in-memory size can exceed its disk size substantially. Use a runtime with more RAM if necessary; keeping checkpoints off reduces Drive usage, not peak calculation memory. |

## Code organization

Shared code lives in [notebooks/Data Labeling/scripts/](notebooks/Data%20Labeling/scripts/):

| Module | Responsibility |
|---|---|
| `config.py` | Site configurations and shared thresholds |
| `io.py` | Data paths, merging, loading, timestamp/year validation and stage exports |
| `feature_selection.py`, `cleaning.py` | Feature groups, RTD/correlation diagnostics and missing-value policy |
| `operating.py`, `performance.py` | Operating features, expected power, PR and power labels |
| `strings.py` | Circuit features, baselines and string labels |
| `inverter.py` | Inverter features, policies and labels |
| `reporting.py` | Notebook reporting utilities |
| `labeling.py` | Complete pipeline and final label summaries |

The original annual notebooks are preserved in [legacy/](notebooks/Data%20Labeling/legacy/) for reference.

## Data attribution and license

When using the NIST data, cite the dataset as requested by its provider:

Boyd M, Chen T, Dougherty B (2017). *NIST Campus Photovoltaic (PV) Arrays and Weather Station Data Sets*. National Institute of Standards and Technology. [Dataset DOI](https://doi.org/10.18434/M3S67G).

Project code is available under the [MIT License](LICENSE). Refer to NIST's documentation for the source dataset's terms and attribution.
