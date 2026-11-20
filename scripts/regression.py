"""Historical label-count regression. Run from the repository root.

python -m scripts.regression --output /content/drive/MyDrive/NIST_regression
python -m scripts.regression --extract-baselines
"""

import argparse
import gc
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from .config import CANOPY, GROUND
from .inverter import ORIGINAL_INVERTER_POLICY
from .io import load_site_year
from .labeling import run_labeling, summarize
from .qa import check_year_coverage, check_sentinels

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "notebooks" / "Data Labeling"
LABELS = ("label_power", "label_string", "label_inverter")
BASELINES = REPO_ROOT / "regression" / "legacy_counts.json"


def digest(path):
    # Text artifacts are checked out with different line endings on Windows
    # and Colab. Normalize CRLF so the gate detects content changes, not Git's
    # newline conversion when moving a validated report between platforms.
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def extract_baselines(destination=BASELINES):
    """Extract only the full count distributions, never diagnostic subsets/%s."""
    records = []
    for site in (CANOPY, GROUND):
        for year in range(2015, 2019):
            path = BASE / "legacy" / f"{site.name.title()}_{year}.ipynb"
            notebook = json.loads(path.read_text(encoding="utf-8"))
            totals = []
            for label_type in LABELS:
                candidates = []
                for index, cell in enumerate(notebook["cells"]):
                    for output in cell.get("outputs", []):
                        text = "".join(output.get("text", output.get("data", {}).get("text/plain", [])))
                        marker = f"{label_type} distribution:"
                        # Ignore 'Corresponding label_power distribution' subsets.
                        match = re.search(r"(?m)^" + re.escape(marker) + r"\s*$", text)
                        if not match:
                            continue
                        block = text[match.end():].split("Name:", 1)[0]
                        counts = dict((name, int(count)) for name, count in re.findall(
                            r"(?m)^([A-Z][A-Z_]+)\s+(\d+)\s*$", block
                        ))
                        if counts:
                            candidates.append((index, counts))
                if len(candidates) != 1:
                    raise ValueError(f"Expected one full count output for {path.name}/{label_type}: {len(candidates)}")
                cell, counts = candidates[0]
                totals.append(sum(counts.values()))
                for label, count in counts.items():
                    records.append({
                        "site": site.name, "year": year, "label_type": label_type,
                        "label": label, "historical_count": count,
                        "source": path.relative_to(BASE).as_posix(), "source_cell": cell,
                        "source_sha256": digest(path),
                        "baseline_note": (
                            "Saved outputs contain 2016 timestamps; not a valid 2017 baseline."
                            if site.name == "ground" and year == 2017 else ""
                        ),
                    })
            if len(set(totals)) != 1:
                raise ValueError(f"Historical label families have inconsistent totals: {path.name}: {totals}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def compare_counts(historical, actual, site, year, reviews=()):
    """Outer comparison includes zero-count and newly introduced categories.

    Reviewed exceptions must pin both counts and a concrete explanation so a
    later change cannot inherit an unrelated waiver. Ground 2017 is the one
    predefined historical wrong-year exception, after input-year QA succeeds.
    """
    keys = ["label_type", "label"]
    old = pd.DataFrame(historical)[keys + ["historical_count"]]
    new = actual[keys + ["count"]].rename(columns={"count": "actual_count"})
    table = old.merge(new, on=keys, how="outer").fillna(0)
    table[["historical_count", "actual_count"]] = table[["historical_count", "actual_count"]].astype(int)
    table.insert(0, "year", year)
    table.insert(0, "site", site)
    table["delta"] = table["actual_count"] - table["historical_count"]
    table["status"] = "MATCH"
    table["explanation"] = ""
    for index, row in table.iterrows():
        if row["delta"] == 0:
            continue
        table.loc[index, "status"] = "UNEXPLAINED"
        if site == "ground" and year == 2017:
            table.loc[index, "status"] = "EXPECTED_WRONG_YEAR_BASELINE"
            table.loc[index, "explanation"] = (
                "Historical Ground_2017 outputs processed 2016; current input passed 2017 year validation. "
                "Counts are not a like-for-like regression baseline."
            )
        for review in reviews:
            fields = ("site", "year", "label_type", "label", "historical_count", "actual_count")
            if all(review.get(key) == row[key] for key in fields) and review.get("explanation", "").strip():
                table.loc[index, "status"] = "REVIEWED_DIFFERENCE"
                table.loc[index, "explanation"] = review["explanation"]
    return table


def write_report(table, output, passed):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "regression_table.csv", index=False)
    # Avoid requiring the optional tabulate dependency in Colab.
    columns = list(table.columns)
    lines = ["# Historical label-count regression", "", f"Result: {'PASS' if passed else 'FAIL / INCOMPLETE'}", "",
             "Policy: ORIGINAL_INVERTER_POLICY. Ground 2017 is a documented wrong-year baseline exception.", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    (output / "regression_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_regression(output, reviews_path=None):
    """Run all eight annual datasets sequentially, retaining only small reports."""
    records = json.loads(BASELINES.read_text(encoding="utf-8"))
    reviews = json.loads(Path(reviews_path).read_text(encoding="utf-8")) if reviews_path else []
    tables, coverage_records = [], []
    for site in (CANOPY, GROUND):
        for year in range(2015, 2019):
            raw = labeled = None
            historical = [r for r in records if r["site"] == site.name and r["year"] == year]
            try:
                if not historical:
                    raise ValueError("Missing historical baseline")
                raw = load_site_year(site, year)
                coverage = check_year_coverage(raw, year)
                check_sentinels(raw)
                coverage_records.append({k: v for k, v in coverage.items() if k not in ("missing_minutes", "off_grid_timestamps")}
                                        | {"site": site.name})
                labeled = run_labeling(raw, site, inverter_policy=ORIGINAL_INVERTER_POLICY)
                tables.append(compare_counts(historical, summarize(labeled), site.name, year, reviews))
            except Exception as exc:
                tables.append(pd.DataFrame([{
                    "site": site.name, "year": year, "label_type": "INPUT_OR_PIPELINE_ERROR",
                    "label": "", "historical_count": "", "actual_count": "", "delta": "",
                    "status": "ERROR", "explanation": f"{type(exc).__name__}: {exc}",
                }]))
            finally:
                del raw, labeled
                gc.collect()
    table = pd.concat(tables, ignore_index=True)
    passed = bool(table["status"].isin(["MATCH", "EXPECTED_WRONG_YEAR_BASELINE", "REVIEWED_DIFFERENCE"]).all())
    write_report(table, output, passed)
    output = Path(output)
    pd.DataFrame(coverage_records).to_csv(output / "coverage.csv", index=False)
    manifest = {
        "passed": passed, "policy": "ORIGINAL_INVERTER_POLICY",
        "datasets": 8, "baseline_sha256": digest(BASELINES),
        "table_sha256": digest(output / "regression_table.csv"),
        "scripts_sha256": {p.name: digest(p) for p in sorted((REPO_ROOT / "scripts").glob("*.py"))},
    }
    (output / "result.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return table, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-baselines", action="store_true")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "regression" / "results")
    parser.add_argument("--reviews", type=Path, help="JSON list of count-specific reviewed mismatch explanations")
    args = parser.parse_args()
    if args.extract_baselines:
        print(f"Extracted {len(extract_baselines())} historical count records")
        return
    _, passed = run_regression(args.output, args.reviews)
    print(f"{'PASS' if passed else 'FAIL'}: {args.output / 'regression_report.md'}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
