"""PDF text extraction and SDS section slicing.

pypdf is used on purpose: it returns text in content-stream order. Layout-based
extractors (pdfplumber, pdftotext -layout) merge the two overlapping text layers
that some suppliers draw in the trade-name field into unreadable text.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

# A heading line starts with optional bullets/asterisks, e.g. "· 2.2 Etiketteringselementen" or "2.2. Label elements".
_HEADING = r"^[\W_]*{num}\.?\s"
_SECTION_3 = r"^[\W_]*(RUBRIEK|SECTION|ABSCHNITT|RUBRIQUE)\s*3\b"
HEADER_MAX_CHARS = 2500


@dataclass(frozen=True)
class SdsSections:
    header: str  # text before section 1.1; holds the revision date
    product_identifier: str  # section 1.1
    label_elements: str  # section 2.2
    full_text: str

    def found(self) -> dict[str, bool]:
        return {
            "header": bool(self.header.strip()),
            "1.1": bool(self.product_identifier.strip()),
            "2.2": bool(self.label_elements.strip()),
        }


def read_pdf_text(path: Path) -> list[str]:
    """Return the text of each page."""
    return [page.extract_text() or "" for page in PdfReader(path).pages]


def _find(pattern: str, text: str, start: int = 0) -> re.Match[str] | None:
    return re.compile(pattern, re.MULTILINE | re.IGNORECASE).search(text, start)


def _slice(text: str, start: re.Match[str] | None, end_pats: list[str]) -> str:
    """Text from the heading match `start` up to the first of `end_pats` (or the end of the text)."""
    if not start:
        return ""
    ends = [m.start() for p in end_pats if (m := _find(p, text, start.end()))]
    end = min(ends) if ends else len(text)
    return text[start.start() : end].strip()


def split_sections(pages: list[str]) -> SdsSections:
    full = "\n".join(pages)
    start_11 = _find(_HEADING.format(num=r"1\.1"), full)
    product = _slice(full, start_11, [_HEADING.format(num=r"1\.2")])
    labels = _slice(
        full,
        _find(_HEADING.format(num=r"2\.2"), full),
        [_HEADING.format(num=r"2\.3"), _SECTION_3],
    )

    # Header = everything before the first 1.1 heading. Searching the full text, not only page 1,
    # handles cover pages added by web shops. Keep the last part only, closest to section 1.
    header = full[: start_11.start()][-HEADER_MAX_CHARS:] if start_11 else full[:HEADER_MAX_CHARS]

    return SdsSections(
        header=header.strip(),
        product_identifier=product,
        label_elements=labels,
        full_text=full,
    )


def load_sections(path: Path) -> SdsSections:
    return split_sections(read_pdf_text(path))
