import json
import subprocess
from pathlib import Path, PurePosixPath


def test_tracked_notebooks_have_no_saved_outputs():
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "*.ipynb"], cwd=root
    ).decode("utf-8")
    violations = []
    for name in filter(None, tracked.split("\0")):
        path = PurePosixPath(name)
        if path == PurePosixPath(
            "notebooks/Data Labeling/Site_Year_Labeling_Walkthrough.ipynb"
        ) or path.is_relative_to("notebooks/Data Labeling/legacy"):
            continue
        notebook = json.loads((root / path).read_text(encoding="utf-8"))
        for number, cell in enumerate(notebook["cells"], start=1):
            if cell["cell_type"] == "code" and (
                cell.get("outputs") or cell.get("execution_count") is not None
            ):
                violations.append(f"{name}, cell {number}")

    assert not violations, "Run nbstripout --keep-id on:\n" + "\n".join(violations)
