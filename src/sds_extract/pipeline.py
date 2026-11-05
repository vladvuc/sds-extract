"""Run the extraction over a folder of PDFs.

Each run writes output/runs/<run_id>/ (config.json, results.json, results.csv, run.log)
and, once finished, appends one line to output/history.jsonl.
"""

import csv
import hashlib
import json
import logging
import platform
import secrets
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from .llm import PROMPT_VERSION, Extractor, LlmUnavailable
from .pdf import load_sections
from .validate import validate

log = logging.getLogger("sds_extract")

CSV_FIELDS = [
    "source_file",
    "trade_name",
    "revision_date",
    "hazard_statements",
    "supplemental_hazard_statements",
    "status",
    "issues",
]


def is_pdf(path: Path) -> bool:
    """Detect PDFs by their signature, not by name or extension."""
    with path.open("rb") as f:
        return b"%PDF-" in f.read(1024)


def discover_inputs(folder: Path) -> tuple[list[Path], list[Path]]:
    """Every PDF under `folder`, including subfolders, whatever its file name.

    Returns (pdfs, skipped). Hidden files and folders are ignored; other non-PDF files are reported as skipped.
    """
    pdfs, skipped = [], []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        if any(part.startswith(".") for part in path.relative_to(folder).parts):
            continue
        (pdfs if is_pdf(path) else skipped).append(path)
    return pdfs, skipped


def process_file(path: Path, extract: Extractor, root: Path | None = None) -> dict:
    record: dict = {
        # Relative path keeps files with the same name in different subfolders apart.
        "source_file": path.relative_to(root).as_posix() if root else path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    try:
        sections = load_sections(path)
        log.info("%s: sections found %s", path.name, sections.found())
        result = extract(sections)
        log.info(
            "%s: model answered in %.1fs: %s", path.name, result.duration_s, result.raw
        )
        checked = validate(result.parsed, sections)
        record |= {
            "trade_name": checked.trade_name,
            "revision_date": checked.revision_date,
            "hazard_statements": ", ".join(checked.hazard_statements),
            "supplemental_hazard_statements": ", ".join(
                checked.supplemental_hazard_statements
            ),
            "status": "needs_review" if checked.issues else "ok",
            "issues": checked.issues,
            "llm": {"raw": result.raw, "duration_s": result.duration_s},
            "sections": {
                "header": sections.header,
                "1.1": sections.product_identifier,
                "2.2": sections.label_elements,
            },
        }
        for issue in checked.issues:
            log.warning("%s: %s", path.name, issue)
    except LlmUnavailable:
        raise  # not a document problem: stop the run instead of failing every remaining PDF
    except Exception as exc:  # one bad document must not stop the batch
        log.exception("%s: extraction failed", path.name)
        record |= {
            "trade_name": None,
            "revision_date": None,
            "hazard_statements": "",
            "supplemental_hazard_statements": "",
            "status": "error",
            "issues": [f"error: {exc}"],
        }
    log.info("%s: status=%s", path.name, record["status"])
    return record


def _write_json(path: Path, data: dict) -> None:
    """Write via a temp file so an interrupted write never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _write_csv(path: Path, documents: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for d in documents:
            writer.writerow({**d, "issues": " | ".join(d["issues"])})


HISTORY_FILE = "history.jsonl"
# Fields copied into history.jsonl; the full records stay in the run folder.
HISTORY_DOC_FIELDS = ["source_file", "sha256", *CSV_FIELDS[1:-1]]


def new_run_id(now: datetime) -> str:
    """UTC timestamp plus a random suffix, e.g. 2026-11-05T12-24-00Z_a1b2c3. Dashes keep it a valid folder name."""
    return f"{now:%Y-%m-%dT%H-%M-%SZ}_{secrets.token_hex(3)}"


def read_history(out_dir: Path) -> list[dict]:
    """One entry per finished run, oldest first. Unreadable lines are skipped with a warning."""
    path = out_dir / HISTORY_FILE
    if not path.exists():
        return []
    entries = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("%s line %d is not valid JSON; skipped", path, n)
    return entries


def load_run(out_dir: Path, run_id: str) -> dict:
    """A run folder as {"run": config + summary, "documents": [...]}."""
    run_dir = out_dir / "runs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"run {run_id} is listed in {out_dir / HISTORY_FILE} but {run_dir} does not exist; "
            f"delete {HISTORY_FILE} together with the runs folder"
        )
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    documents = results.pop("documents")
    return {"run": {**config, **results}, "documents": documents}


def load_latest(out_dir: Path) -> dict:
    """The newest finished run, or an empty result when there is none."""
    history = read_history(out_dir)
    if not history:
        return {"run": {}, "documents": []}
    return load_run(out_dir, history[-1]["run_id"])


def run(
    pdf_dir: Path, out_dir: Path, extract: Extractor, run_info: dict | None = None
) -> dict:
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"input folder {pdf_dir} does not exist")
    started_at = datetime.now(timezone.utc)
    run_id = new_run_id(started_at)
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)

    try:
        pdfs, skipped = discover_inputs(pdf_dir)
        # Written before processing, so an interrupted run still shows what was attempted.
        config = {
            "run_id": run_id,
            "started_at": started_at.isoformat(timespec="seconds"),
            "input_dir": str(pdf_dir),
            "prompt_version": PROMPT_VERSION,
            "package_version": version("sds-extract"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **(run_info or {}),
        }
        _write_json(run_dir / "config.json", config)
        log.info("run %s started: %d PDF(s) in %s", run_id, len(pdfs), pdf_dir)
        log.info("run settings: %s", json.dumps(config))

        started = time.perf_counter()
        skipped_files = [p.relative_to(pdf_dir).as_posix() for p in skipped]
        for name in skipped_files:
            log.warning("%s: skipped, not a PDF", name)
        documents = [process_file(p, extract, pdf_dir) for p in pdfs]
        summary = {
            "run_id": run_id,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "duration_s": round(time.perf_counter() - started, 2),
            "counts": {
                s: sum(d["status"] == s for d in documents)
                for s in ("ok", "needs_review", "error")
            },
            "skipped_files": skipped_files,
        }
        _write_json(run_dir / "results.json", {**summary, "documents": documents})
        _write_csv(run_dir / "results.csv", documents)
        log.info(
            "run %s finished in %.1fs: %s", run_id, summary["duration_s"], summary["counts"]
        )

        # Appended last, so only finished runs reach the history.
        entry = {
            **summary,
            "started_at": config["started_at"],
            "model": (run_info or {}).get("llm", {}).get("model"),
            "prompt_version": PROMPT_VERSION,
            "documents": [{f: d.get(f) for f in HISTORY_DOC_FIELDS} for d in documents],
        }
        with (out_dir / HISTORY_FILE).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log.info("wrote %s and appended to %s", run_dir, out_dir / HISTORY_FILE)
        return {"run": {**config, **summary, "run_dir": str(run_dir)}, "documents": documents}
    finally:
        log.removeHandler(handler)
        handler.close()
