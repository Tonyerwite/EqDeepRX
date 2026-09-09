from pathlib import Path


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
    assert "118,740" in audit
    assert "github.com/Tonyerwite/EqDeepRX" in search
    assert "没有发现" in search or "未发现" in search

