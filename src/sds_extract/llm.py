"""Ollama call with a JSON-schema constrained response."""

from dataclasses import asdict, dataclass, field
from typing import Protocol

from .pdf import SdsSections
from .validate import LlmExtraction

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You extract product master data from European Safety Data Sheets (SDS).
The SDS can be in any EU language (for example Dutch, English, German, French).
You receive three excerpts: the document HEADER, SECTION 1.1 (product identifier) and SECTION 2.2 (label elements).

Rules:
- trade_name: the value of "Trade name", "Product name", "Handelsnaam" or equivalent in SECTION 1.1.
  Copy it exactly as written, same language, same capitalisation. Never include UFI, product code, article number or packaging.
- revision_date: the date labelled "Revision date", "Date of revision", "Revisiedatum", "herzien op", "Datum herziening" or equivalent.
  Copy it exactly as written. Ignore the print date ("Afdrukdatum"), issue date ("Datum van uitgave") and the date of the replaced version.
- hazard_codes: every H-code (H followed by 3 digits) and EUH-code (EUH followed by 3 digits) that appears in SECTION 2.2,
  including codes listed under supplementary information ("Aanvullende gegevens"). Do not include P-codes.
  If the hazard statements are "not applicable" and no codes appear, return an empty list.
- Use null when a value is not present. Never invent values."""


def build_user_prompt(sections: SdsSections) -> str:
    return (
        f"### HEADER\n{sections.header or '(not found)'}\n\n"
        f"### SECTION 1.1\n{sections.product_identifier or '(not found)'}\n\n"
        f"### SECTION 2.2\n{sections.label_elements or '(not found)'}"
    )


@dataclass
class LlmSettings:
    model: str = "qwen2.5:7b"
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    seed: int = 56
    num_ctx: int = 8192
    think: bool | None = (
        None  # only sent when set; use False for qwen3 models to skip the reasoning phase
    )
    timeout_s: float = 600.0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class LlmResult:
    parsed: LlmExtraction
    raw: str
    duration_s: float


class Extractor(Protocol):
    def __call__(self, sections: SdsSections) -> LlmResult: ...


class OllamaExtractor:
    def __init__(self, settings: LlmSettings):
        from ollama import (
            Client,  # imported lazily so tests and the UI do not need a running Ollama
        )

        self.settings = settings
        self.client = Client(host=settings.host, timeout=settings.timeout_s)

    def server_version(self) -> str | None:
        import httpx  # dependency of the ollama client

        try:
            response = httpx.get(f"{self.settings.host}/api/version", timeout=5)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, httpx.InvalidURL, ValueError):
            return None
        version = data.get("version") if isinstance(data, dict) else None
        return version if isinstance(version, str) else None

    def __call__(self, sections: SdsSections) -> LlmResult:
        s = self.settings
        response = self.client.chat(
            model=s.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(sections)},
            ],
            format=LlmExtraction.model_json_schema(),
            options={
                "temperature": s.temperature,
                "seed": s.seed,
                "num_ctx": s.num_ctx,
                **s.extra,
            },
            # None is dropped by the client (exclude_none), so it is only sent when set
            think=s.think,
        )
        raw = response.message.content or ""
        duration = (response.total_duration or 0) / 1e9
        return LlmResult(
            parsed=LlmExtraction.model_validate_json(raw),
            raw=raw,
            duration_s=round(duration, 2),
        )
