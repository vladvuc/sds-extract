"""Ollama call with a JSON-schema constrained response."""

from dataclasses import asdict, dataclass, field
from typing import Protocol

from jinja2 import Environment, PackageLoader, StrictUndefined

from .pdf import SdsSections
from .validate import LlmExtraction

# Prompts live in src/sds_extract/prompts/ as Jinja2 templates:
#   system_<version>.j2  system prompt, one file per version (older versions are kept to reproduce old runs)
#   user.j2              user message with the three SDS excerpts
PROMPT_VERSION = "v2"

_templates = Environment(
    loader=PackageLoader("sds_extract", "prompts"),
    undefined=StrictUndefined,  # a missing variable is an error, never a silently empty prompt
    autoescape=False,  # plain text, not HTML
)


def render_system_prompt(version: str = PROMPT_VERSION) -> str:
    return _templates.get_template(f"system_{version}.j2").render()


def build_user_prompt(sections: SdsSections) -> str:
    return _templates.get_template("user.j2").render(
        header=sections.header,
        product_identifier=sections.product_identifier,
        label_elements=sections.label_elements,
    )


SYSTEM_PROMPT = render_system_prompt()


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
