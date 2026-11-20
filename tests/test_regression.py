import json

import pandas as pd
import pytest

from scripts.config import CANOPY, GROUND
from scripts.feature_selection import SECONDARY_FEATURES, REDUNDANT_FEATURES
from scripts.inverter import FAULT_FLAG_COLUMNS, ORIGINAL_INVERTER_POLICY
from scripts.labeling import STAGES, run_labeling
from scripts.regression import BASE, BASELINES, compare_counts, extract_baselines


@pytest.mark.parametrize("site", [CANOPY, GROUND])
def test_staged_and_full_pipeline_are_identical(site):
    columns = set(SECONDARY_FEATURES + REDUNDANT_FEATURES + FAULT_FLAG_COLUMNS)
    columns.update(f"RTD_C_Avg_{i}" for i in range(1, 10))
    columns.update(f"ShuntCurrent_A_Avg_{i}" for i in range(1, 8))
    columns.update(f"ShuntPDC_kW_Avg_{i}" for i in range(1, 8))
    columns.update(site.op_mask_columns + site.poa_columns)
    columns.update(("WindSpeedAve_ms", "InvPAC_kW_Avg", "InvPDC_kW_Avg", "InvOpStatus_Avg"))
    frame = pd.DataFrame(0.0, index=pd.date_range("2015-06-01 12:00", periods=90, freq="min"), columns=sorted(columns))
    for column in site.op_mask_columns + site.poa_columns:
        frame[column] = 800.0
    for i in range(1, 10):
        frame[f"RTD_C_Avg_{i}"] = 25.0
    for i in range(1, 8):
        frame[f"ShuntCurrent_A_Avg_{i}"] = 50.0
        frame[f"ShuntPDC_kW_Avg_{i}"] = 20.0
    frame["InvOpStatus_Avg"] = 1
    frame["InvPAC_kW_Avg"] = 150.0
    frame["InvPDC_kW_Avg"] = 160.0
    original = frame.copy()
    full = run_labeling(frame, site, inverter_policy=ORIGINAL_INVERTER_POLICY)
    staged = frame
    for stage in STAGES:
        staged = run_labeling(staged, site, stages=(stage,), inverter_policy=ORIGINAL_INVERTER_POLICY)
    pd.testing.assert_frame_equal(full, staged)
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError, match="pipeline order"):
        run_labeling(frame, site, stages=("redundant", "secondary"))


def test_extracted_baselines_match_saved_evidence(tmp_path):
    if not (BASE / "legacy" / "Canopy_2015.ipynb").exists():
        pytest.skip("Legacy sources were removed after a passing regression; frozen baselines remain")
    extracted = extract_baselines(tmp_path / "counts.json")
    assert extracted == json.loads(BASELINES.read_text(encoding="utf-8"))
    assert {(r["site"], r["year"]) for r in extracted} == {
        (site, year) for site in ("canopy", "ground") for year in range(2015, 2019)
    }


def test_frozen_baseline_families_have_consistent_totals():
    records = pd.DataFrame(json.loads(BASELINES.read_text(encoding="utf-8")))
    totals = records.groupby(["site", "year", "label_type"])["historical_count"].sum()
    assert len(totals) == 24
    assert totals.groupby(level=["site", "year"]).nunique().eq(1).all()


def test_digest_is_portable_across_windows_and_colab(tmp_path):
    from scripts.regression import digest

    windows = tmp_path / "windows.py"
    linux = tmp_path / "linux.py"
    windows.write_bytes(b"x = 1\r\ny = 2\r\n")
    linux.write_bytes(b"x = 1\ny = 2\n")
    assert digest(windows) == digest(linux)


def test_new_categories_and_unreviewed_differences_fail():
    old = [{"label_type": "label_power", "label": "NORMAL", "historical_count": 10}]
    actual = pd.DataFrame([
        {"label_type": "label_power", "label": "NORMAL", "count": 9},
        {"label_type": "label_power", "label": "MISSING_DATA", "count": 1},
    ])
    table = compare_counts(old, actual, "canopy", 2015)
    assert set(table["status"]) == {"UNEXPLAINED"}
    assert set(table["delta"]) == {-1, 1}
    ground = compare_counts(old, actual, "ground", 2017)
    assert set(ground["status"]) == {"EXPECTED_WRONG_YEAR_BASELINE"}


def test_review_only_applies_to_the_exact_counts():
    old = [{"label_type": "label_power", "label": "NORMAL", "historical_count": 10}]
    actual = pd.DataFrame([{"label_type": "label_power", "label": "NORMAL", "count": 9}])
    review = dict(site="canopy", year=2015, label_type="label_power", label="NORMAL",
                  historical_count=10, actual_count=9, explanation="Verified boundary rounding change")
    assert compare_counts(old, actual, "canopy", 2015, [review]).iloc[0]["status"] == "REVIEWED_DIFFERENCE"
    actual["count"] = 8
    assert compare_counts(old, actual, "canopy", 2015, [review]).iloc[0]["status"] == "UNEXPLAINED"


def test_all_dataset_errors_produce_failing_artifacts(tmp_path, monkeypatch):
    from scripts import regression

    def missing_input(*args):
        raise FileNotFoundError("Synthetic missing input")

    monkeypatch.setattr(regression, "load_site_year", missing_input)
    table, passed = regression.run_regression(tmp_path)
    assert not passed
    assert len(table) == 8
    assert set(table["status"]) == {"ERROR"}
    assert (tmp_path / "regression_report.md").exists()
    assert json.loads((tmp_path / "result.json").read_text())["passed"] is False


def test_cleanup_refuses_unpassed_regression(tmp_path):
    from scripts.cleanup_legacy import cleanup

    (tmp_path / "result.json").write_text(json.dumps({"passed": False, "datasets": 8}))
    with pytest.raises(ValueError, match="must pass regression"):
        cleanup(tmp_path)


def test_cleanup_only_removes_named_notebooks_after_verified_pass(tmp_path, monkeypatch):
    from scripts import cleanup_legacy
    from scripts.regression import digest

    repo = tmp_path / "repository"
    base = repo / "notebooks" / "Data Labeling"
    results = tmp_path / "results"
    (repo / "scripts").mkdir(parents=True)
    script = repo / "scripts" / "labeling.py"
    script.write_text("# Shared labeling code\n")
    (base / "legacy").mkdir(parents=True)
    results.mkdir()
    baseline = base / "counts.json"
    baseline.write_text("[]")
    retained = base / "Data_Labeling.ipynb"
    retained.write_text("canonical")
    targets = [base / "legacy" / f"{site}_{year}.ipynb"
               for site in ("Canopy", "Ground") for year in range(2015, 2019)]
    targets += [base / name for name in ("NIST_labeling.ipynb", "Site_Year_Labeling.ipynb",
                                         "Site_Year_Labeling_Walkthrough.ipynb")]
    for target in targets:
        target.write_text("superseded")
    table = pd.DataFrame([
        dict(site=site, year=year, label_type=family, status="MATCH")
        for site in ("canopy", "ground") for year in range(2015, 2019)
        for family in ("label_power", "label_string", "label_inverter")
    ])
    table.to_csv(results / "regression_table.csv", index=False)
    (results / "result.json").write_text(json.dumps({
        "passed": True, "datasets": 8, "policy": "ORIGINAL_INVERTER_POLICY",
        "baseline_sha256": digest(baseline), "scripts_sha256": {script.name: digest(script)},
        "table_sha256": digest(results / "regression_table.csv"),
    }))
    monkeypatch.setattr(cleanup_legacy, "REPO_ROOT", repo)
    monkeypatch.setattr(cleanup_legacy, "BASE", base)
    monkeypatch.setattr(cleanup_legacy, "BASELINES", baseline)
    cleanup_legacy.cleanup(results)
    assert retained.read_text() == "canonical"
    assert not any(target.exists() for target in targets)


def test_canonical_notebook_uses_each_shared_stage_once_and_compiles():
    import ast

    notebook = json.loads((BASE / "Data_Labeling.ipynb").read_text(encoding="utf-8"))
    stages = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, "Data_Labeling.ipynb", "exec")
        for call in ast.walk(ast.parse(source)):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "run_labeling":
                stages.extend(ast.literal_eval(next(k.value for k in call.keywords if k.arg == "stages")))
    assert tuple(stages) == STAGES
