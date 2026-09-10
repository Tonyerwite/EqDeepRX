from pathlib import Path

import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_safe_smoke_and_explicit_full_training():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "scripts/preflight.py --tiny" in readme
    assert "scripts/train.py --confirm-full-run" in readme
    assert "uncoded BER" in readme
    assert "LDPC" in readme


def test_audit_and_search_docs_contain_paper_and_external_search_evidence():
    audit = (ROOT / "docs" / "paper_audit.md").read_text(encoding="utf-8")
    search = (ROOT / "docs" / "open_source_search.md").read_text(encoding="utf-8")
    assert "DenoiseNN" in audit and "RZF" in audit and "LMMSE" in audit
    assert "115,456" in audit
    assert "github.com/Tonyerwite/EqDeepRX" in search
    assert "No author or third-party public implementation" in search


def test_declared_runtime_versions_cover_sionna_2_1_requirements():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert project["requires-python"] == ">=3.11"
    assert "torch>=2.9.1" in project["dependencies"]
    assert "numpy>=2.2.6" in project["dependencies"]
    assert "matplotlib>=3.10.8" in project["dependencies"]
    assert "sionna==2.1.0" in project["optional-dependencies"]["standard"]
