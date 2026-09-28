import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sensitivity_confirmation_supervisor.ps1"


def test_confirmation_supervisor_dry_run_lists_every_matrix_value(tmp_path: Path):
    if shutil.which("powershell.exe") is None:
        pytest.skip("Windows PowerShell is required for this supervisor contract test")
    screening = tmp_path / "screening"
    output = tmp_path / "confirmation"
    screening.mkdir()
    summary = {
        "execution_complete": True,
        "status": "complete",
        "candidate_matrix": {
            "time_mixer_channels": [1, 2, 4],
            "lamb_bias_correction": [True, False],
        },
        "candidate_decisions": [
            {"variable": "time_mixer_channels", "value": 2, "decision": "baseline"},
            {"variable": "time_mixer_channels", "value": 1, "decision": "rejected"},
            {"variable": "time_mixer_channels", "value": 4, "decision": "rejected"},
            {"variable": "lamb_bias_correction", "value": True, "decision": "baseline"},
            {"variable": "lamb_bias_correction", "value": False, "decision": "rejected"},
        ],
    }
    (screening / "sensitivity_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-ScreeningDir",
            str(screening),
            "-RootOutputDir",
            str(output),
            "-SelectionMode",
            "all",
            "-DryRun",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    jobs = json.loads(result.stdout)
    values = {job["variable"]: job["values"] for job in jobs}
    assert values["time_mixer_channels"] == "1,2,4"
    assert values["lamb_bias_correction"] == "true,false"
