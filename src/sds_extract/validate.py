"""Deterministic guardrails around the LLM output.

The model proposes values; this module checks every value against the source text,
normalises formats and records any disagreement as an issue for human review.
"""

import re
from datetime import date

from pydantic import BaseModel, Field

from .pdf import SdsSections

HAZARD_CODE = re.compile(r"\b(EUH\d{3}|H\d{3}[A-Za-z]{0,2})\b")
# Day-first or ISO date. Single spaces between digits are tolerated because some PDFs
# extract as "23.07.20 25".
_DATE = r"(\d{1,2} ?[./-] ?\d{1,2} ?[./-] ?\d(?: ?\d){3}|\d{4}-\d{2}-\d{2})"
# Revision-date labels seen in NL/EN/DE SDSs, matched with or without spaces ("Überarbeitetam").
# Print date ("Afdrukdatum", "Druckdatum", "print date"), issue date ("Datum van uitgave") and the
# replaced version's date ("Replacing version dated", "Ersetzt") are excluded on purpose.
_LABELS = [
    "herzien op", "herziening van", "datum herziening", "revisiedatum", "herzieningsdatum",
    "revision date / version", "revision date", "date of revision", "revised on",
    "überarbeitet am", "überarbeitungsdatum", "stand",
]
REVISION_LABEL = re.compile(
    r"\b(" + "|".join(lbl.replace(" ", r"\s*") for lbl in _LABELS) + r")\s*:?\s*" + _DATE,
    re.IGNORECASE,
)


class LlmExtraction(BaseModel):
    """Schema the model must fill. Field descriptions are sent to the model as part of the JSON schema."""

    trade_name: str | None = Field(
        description="Commercial product name from section 1.1, copied exactly as written. No UFI, product code or article number."
    )
    revision_date: str | None = Field(
        description="Revision date copied exactly as written in the document. Not the print date, issue date or superseded version date."
    )
    hazard_codes: list[str] = Field(
        description="All H-codes and EUH-codes listed in section 2.2, e.g. H225 or EUH066. No P-codes. Empty list if none."
    )


class Extraction(BaseModel):
    trade_name: str | None
    revision_date: str | None  # dd/mm/yyyy
    hazard_statements: list[str]  # H-codes only, as the assignment asks
    supplemental_hazard_statements: list[str]  # EUH-codes, CLP "supplemental hazard information"
    issues: list[str]


def split_codes(codes: list[str]) -> tuple[list[str], list[str]]:
    """Split hazard codes into (H-codes, EUH-codes), keeping order."""
    return [c for c in codes if not c.startswith("EUH")], [c for c in codes if c.startswith("EUH")]


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_date(raw: str | None) -> str | None:
    """Return dd/mm/yyyy, or None if the value is not a recognisable day-first or ISO date.

    Extra text around a single date is tolerated, e.g. "10.07.2015 / 0003" copied from
    "Revision date / version". A value with two or more dates is ambiguous and returns None.
    """
    if not raw:
        return None
    compact = re.sub(r"\s+", "", raw)
    try:
        if m := re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", compact):
            y, mo, d = map(int, m.groups())
        elif m := re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", compact):
            d, mo, y = map(int, m.groups())
        else:
            found = re.findall(_DATE, raw)
            return normalize_date(found[0]) if len(found) == 1 and found[0] != raw else None
        return date(y, mo, d).strftime("%d/%m/%Y")
    except ValueError:
        return None


def find_revision_date(sections: SdsSections) -> str | None:
    """Rule-based revision date, used only to cross-check the model."""
    for text in (sections.header, sections.full_text):
        if m := REVISION_LABEL.search(text):
            return normalize_date(m.group(2))
    return None


def find_hazard_codes(text: str) -> list[str]:
    """H/EUH codes in order of first appearance."""
    return list(dict.fromkeys(m.group(1).upper() for m in HAZARD_CODE.finditer(text)))


def validate(llm: LlmExtraction, sections: SdsSections) -> Extraction:
    issues: list[str] = []

    # Trade name: must be present verbatim (whitespace-insensitive) in section 1.1.
    name = _squash(llm.trade_name) if llm.trade_name else None
    source_11 = _squash(sections.product_identifier or sections.full_text)
    if not name:
        issues.append("trade_name: model returned no value")
    elif name not in source_11:
        issues.append(f"trade_name: '{name}' not found verbatim in section 1.1")
    if name and re.search(r"\bUFI\b", name, re.IGNORECASE):
        issues.append("trade_name: contains a UFI")

    # Revision date: normalise, then compare with the labelled date found by regex.
    revision = normalize_date(llm.revision_date)
    if llm.revision_date and not revision:
        issues.append(f"revision_date: cannot parse '{llm.revision_date}'")
    elif not llm.revision_date:
        issues.append("revision_date: model returned no value")
    rule_date = find_revision_date(sections)
    if rule_date and revision and rule_date != revision:
        issues.append(
            f"revision_date: model says {revision}, labelled revision date in document is {rule_date}"
        )
    if rule_date and not revision:
        revision = rule_date
        issues.append(f"revision_date: filled from labelled date {rule_date}")

    # Hazard codes: keep only codes that exist in section 2.2, report codes the model missed.
    scope = sections.label_elements or sections.full_text
    in_source = find_hazard_codes(scope)
    proposed = [c.strip().upper().replace(" ", "") for c in llm.hazard_codes]
    kept = [c for c in in_source if c in proposed]  # source order, deduplicated
    for c in dict.fromkeys(proposed):
        if c not in in_source:
            issues.append(
                f"hazard_statements: dropped '{c}', not present in section 2.2"
            )
    missed = [c for c in in_source if c not in proposed]
    if missed:
        issues.append(
            f"hazard_statements: model missed {', '.join(missed)} found in section 2.2"
        )
    if not sections.label_elements:
        issues.append(
            "sections: section 2.2 not found, hazard codes checked against the whole document"
        )
    if not sections.product_identifier:
        issues.append("sections: section 1.1 not found")

    h_codes, euh_codes = split_codes(kept)
    return Extraction(
        trade_name=name,
        revision_date=revision,
        hazard_statements=h_codes,
        supplemental_hazard_statements=euh_codes,
        issues=issues,
    )
