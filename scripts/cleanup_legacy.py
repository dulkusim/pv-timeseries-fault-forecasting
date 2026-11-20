"""Delete superseded notebooks only after verifying a passing regression artifact.

python -m scripts.cleanup_legacy /path/to/NIST_regression
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from .regression import BASE, BASELINES, REPO_ROOT, digest


def cleanup(results):
    results = Path(results)
    manifest = json.loads((results / "result.json").read_text(encoding="utf-8"))
    if manifest.get("passed") is not True or manifest.get("datasets") != 8:
        raise ValueError("All eight datasets must pass regression before cleanup")
    if manifest.get("policy") != "ORIGINAL_INVERTER_POLICY":
        raise ValueError("Regression must use the historical inverter policy")
    if manifest["baseline_sha256"] != digest(BASELINES):
        raise ValueError("Baseline changed since regression")
    for record in json.loads(BASELINES.read_text(encoding="utf-8")):
        source = BASE / record["source"]
        if source.exists() and digest(source) != record["source_sha256"]:
            raise ValueError(
                f"Historical notebook changed since baseline extraction: {source.name}"
            )
    if manifest["table_sha256"] != digest(results / "regression_table.csv"):
        raise ValueError("Regression table changed since the run")
    scripts = {p.name: digest(p) for p in sorted((REPO_ROOT / "scripts").glob("*.py"))}
    if manifest["scripts_sha256"] != scripts:
        raise ValueError("Scripts changed since regression; rerun the check")
    table = pd.read_csv(results / "regression_table.csv")
    expected = {
        (site, year) for site in ("canopy", "ground") for year in range(2015, 2019)
    }
    if set(zip(table["site"], table["year"], strict=False)) != expected:
        raise ValueError("Regression table does not cover all eight datasets")
    allowed = {"MATCH", "EXPECTED_WRONG_YEAR_BASELINE", "REVIEWED_DIFFERENCE"}
    if not table["status"].isin(allowed).all():
        raise ValueError("Unresolved regression rows remain")
    expected_families = {
        (s, y, label)
        for s, y in expected
        for label in ("label_power", "label_string", "label_inverter")
    }
    if (
        set(zip(table["site"], table["year"], table["label_type"], strict=False))
        != expected_families
    ):
        raise ValueError("Regression must cover all three label families")
    targets = [
        BASE / "legacy" / f"{site}_{year}.ipynb"
        for site in ("Canopy", "Ground")
        for year in range(2015, 2019)
    ]
    targets += [
        BASE / name
        for name in (
            "NIST_labeling.ipynb",
            "Site_Year_Labeling.ipynb",
            "Site_Year_Labeling_Walkthrough.ipynb",
        )
    ]
    resolved_base = BASE.resolve()
    # Resolve and validate every target before deleting anything.
    for target in targets:
        target.resolve().relative_to(resolved_base)
    for target in targets:
        if target.exists():
            target.unlink()
            print(f"Removed {target.relative_to(BASE)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    cleanup(parser.parse_args().results)


if __name__ == "__main__":
    main()
