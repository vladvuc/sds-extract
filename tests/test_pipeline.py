"""Tests run against whatever is in data/, whatever the file names are.

- Every PDF found in data/ (by content, any name, any subfolder) must be readable and sliceable.
- Hand-verified values live in tests/expected.json, keyed by the SHA-256 of the PDF, so renaming
  or moving a file never breaks a test. A new PDF without an entry still gets the generic checks.
- The LLM is replaced by a stub, so no Ollama is needed.
"""

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import pytest

from sds_extract.llm import LlmResult
from sds_extract.pdf import SdsSections, load_sections
from sds_extract.pipeline import discover_inputs, load_latest, new_run_id, read_history, run
from sds_extract.validate import (
    HAZARD_CODE,
    LlmExtraction,
    find_hazard_codes,
    find_revision_date,
    normalize_date,
    split_codes,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXPECTED: dict[str, dict] = json.loads((Path(__file__).parent / "expected.json").read_text(encoding="utf-8"))["documents"]
PDFS, _ = discover_inputs(DATA) if DATA.exists() else ([], [])


def _id(p: Path) -> str:
    return p.relative_to(DATA).as_posix()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@cache
def _sections(p: Path) -> SdsSections:
    return load_sections(p)


@cache
def _expected_by_text() -> dict[str, dict]:
    return {_sections(p).full_text: EXPECTED[_sha(p)] for p in PDFS if _sha(p) in EXPECTED}


def _all_codes(exp: dict) -> list[str]:
    """H and EUH codes of one expected record, as the model returns them: one list."""
    return [c for f in ("hazard_statements", "supplemental_hazard_statements") for c in exp[f].split(", ") if c]


def _fields(d: dict) -> tuple:
    return tuple(d[f] for f in ("trade_name", "revision_date", "hazard_statements", "supplemental_hazard_statements"))


def fake_model(sections: SdsSections) -> LlmResult:
    """Stub LLM: answers with the hand-verified values when it knows the document, else with nothing."""
    exp = _expected_by_text().get(sections.full_text)
    parsed = LlmExtraction(
        trade_name=exp["trade_name"] if exp else None,
        revision_date=exp["revision_date"] if exp else None,
        hazard_codes=_all_codes(exp) if exp else [],
    )
    return LlmResult(parsed=parsed, raw=parsed.model_dump_json(), duration_s=0.0)


def test_data_folder_contains_pdfs():
    assert PDFS, f"no PDFs found under {DATA}"


@pytest.mark.parametrize("pdf", PDFS, ids=_id)
def test_every_pdf_in_data_folder_is_supported(pdf):
    s = _sections(pdf)
    assert s.full_text.strip(), "no extractable text, probably a scanned PDF"
    assert s.found() == {"header": True, "1.1": True, "2.2": True}, "section heading not recognised"
    assert find_revision_date(s), "no labelled revision date found; add the label to validate.REVISION_LABEL"
    assert all(HAZARD_CODE.fullmatch(c) for c in find_hazard_codes(s.label_elements))


@pytest.mark.parametrize("pdf", PDFS, ids=_id)
def test_rule_checks_match_hand_verified_values(pdf):
    exp = EXPECTED.get(_sha(pdf))
    if exp is None:
        pytest.skip("no hand-verified values yet; add this PDF's SHA-256 to tests/expected.json")
    s = _sections(pdf)
    assert _squash(exp["trade_name"]) in _squash(s.product_identifier)
    assert find_revision_date(s) == exp["revision_date"]
    h_codes, euh_codes = split_codes(find_hazard_codes(s.label_elements))
    assert (", ".join(h_codes), ", ".join(euh_codes)) == (exp["hazard_statements"], exp["supplemental_hazard_statements"])


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("15.07.2026", "15/07/2026"),
        ("21-7-2026", "21/07/2026"),
        ("4.3.2025", "04/03/2025"),
        ("2026-04-20", "20/04/2026"),
        ("31.02.2026", None),
        ("July", None),
    ],
)
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


def test_validation_flags_model_mistakes():
    header = "Afdrukdatum: 08.07.2026 herzien op: 20.04.2026 Versienummer 43"
    product = "· 1.1 Productidentificatie\n· Handelsnaam: BISON KIT TRANSPARANT"
    labels = "· 2.2 Etiketteringselementen\nH225 Licht ontvlambaar.\nH319 Oogirritatie.\nH336 Slaperigheid.\nEUH066 Droge huid."
    section3 = "RUBRIEK 3\nAceton Flam. Liq. 2, H225; Skin Sens. 1, H317"
    s = SdsSections(header, product, labels, "\n".join([header, "BIS CONTACT KIT TR BD 100ML*6 NLFR", product, labels, section3]))
    bad = LlmExtraction(
        trade_name="BIS CONTACT KIT TR BD 100ML*6 NLFR",  # internal article line, not the trade name
        revision_date="08.07.2026",  # print date
        hazard_codes=["H225", "H319", "H317", "P210"],  # H317 is an ingredient code; H336 and EUH066 missed
    )
    out = validate(bad, s)
    assert out.hazard_statements == ["H225", "H319"]
    assert out.supplemental_hazard_statements == []
    text = " ".join(out.issues)
    assert "not found verbatim" in text
    assert "20/04/2026" in text
    assert "dropped 'H317'" in text and "dropped 'P210'" in text
    assert "missed H336, EUH066" in text

    good = LlmExtraction(trade_name="BISON KIT TRANSPARANT", revision_date="20.04.2026", hazard_codes=["H225", "H319", "H336", "EUH066"])
    out = validate(good, s)
    assert (out.hazard_statements, out.supplemental_hazard_statements, out.issues) == (["H225", "H319", "H336"], ["EUH066"], [])


def test_pipeline_over_data_folder(tmp_path):
    result = run(DATA, tmp_path, fake_model, {"llm": {"model": "fake"}})

    docs = result["documents"]
    assert len(docs) == len(PDFS)
    assert result["run"]["counts"]["error"] == 0
    rows = {r["source_file"]: r for r in csv.DictReader((Path(result["run"]["run_dir"]) / "results.csv").open(encoding="utf-8"))}
    for d in docs:
        exp = EXPECTED.get(d["sha256"])
        if exp is None:
            continue  # unknown PDF: stub returns nothing, so needs_review is expected
        assert d["status"] == "ok", d["issues"]
        assert _fields(rows[d["source_file"]]) == _fields(exp)
    assert read_history(tmp_path)[-1]["counts"] == result["run"]["counts"]
    assert (Path(result["run"]["run_dir"]) / "run.log").exists()


def test_run_id_format():
    run_id = new_run_id(datetime(2026, 11, 5, 12, 24, 0, tzinfo=timezone.utc))
    assert re.fullmatch(r"2026-11-05T12-24-00Z_[0-9a-f]{6}", run_id)


def test_run_folders_and_history(tmp_path):
    """Each run gets its own folder; history.jsonl gains one line per finished run only."""
    pdf = PDFS[0]
    inbox, out = tmp_path / "inbox", tmp_path / "out"
    inbox.mkdir()
    shutil.copy(pdf, inbox / pdf.name)

    first = run(inbox, out, fake_model, {"llm": {"model": "fake"}})
    second = run(inbox, out, fake_model, {"llm": {"model": "fake"}})

    history = read_history(out)
    assert [h["run_id"] for h in history] == [first["run"]["run_id"], second["run"]["run_id"]]
    assert history[0]["model"] == "fake" and history[0]["documents"][0]["source_file"] == pdf.name
    assert "sections" not in history[0]["documents"][0]  # full records live in the run folder
    for r in (first, second):
        run_dir = out / "runs" / r["run"]["run_id"]
        assert sorted(p.name for p in run_dir.iterdir()) == ["config.json", "results.csv", "results.json", "run.log"]
        assert json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["llm"] == {"model": "fake"}
        assert json.loads((run_dir / "results.json").read_text(encoding="utf-8"))["documents"] == r["documents"]
    assert load_latest(out)["run"]["run_id"] == second["run"]["run_id"]

    def crash(_sections):
        raise KeyboardInterrupt  # not caught per document, so the run stops before finishing

    with pytest.raises(KeyboardInterrupt):
        run(inbox, out, crash)
    assert len(read_history(out)) == 2  # the interrupted run has a folder with config.json, but no history line


def test_file_names_do_not_matter(tmp_path):
    """Odd names, missing extensions, subfolders and duplicates are processed; non-PDFs are skipped."""
    known = [p for p in PDFS if _sha(p) in EXPECTED][:2]
    if len(known) < 2:
        pytest.skip("needs two PDFs with hand-verified values")
    inbox, out = tmp_path / "inbox", tmp_path / "out"
    copies = {
        "SDS final (2).PDF": known[0],
        "nested/deeper/ümlaut näme": known[1],  # no extension
        "copy-of-first.pdf": known[0],  # same content, different name
    }
    for name, src in copies.items():
        (inbox / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, inbox / name)
    (inbox / "notes.txt").write_text("not a pdf")
    (inbox / "fake.pdf").write_text("named .pdf but plain text")
    (inbox / ".DS_Store").write_bytes(b"\0")

    result = run(inbox, out, fake_model)

    assert {d["source_file"] for d in result["documents"]} == set(copies)
    assert result["run"]["skipped_files"] == ["fake.pdf", "notes.txt"]
    for d in result["documents"]:
        exp = EXPECTED[d["sha256"]]
        assert d["status"] == "ok", d["issues"]
        assert _fields(d) == _fields(exp)
