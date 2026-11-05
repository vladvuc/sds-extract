"""Ollama call with a JSON-schema constrained response."""

from dataclasses import asdict, dataclass, field
from typing import Protocol

import httpx  # dependency of the ollama client
from jinja2 import Environment, PackageLoader, StrictUndefined
from ollama import Client, ResponseError

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


class LlmUnavailable(Exception):
    """Ollama cannot serve any request (server down, model not pulled, server error).

    Not a problem of one document, so the pipeline stops the whole run instead of
    recording the same error for every remaining PDF.
    """


class OllamaExtractor:
    def __init__(self, settings: LlmSettings):
        self.settings = settings
        self.client = Client(host=settings.host, timeout=settings.timeout_s)

    def server_version(self) -> str:
        """Version of the Ollama server, used as the reachability check before a run."""
        host = self.settings.host
        try:
            response = httpx.get(f"{host}/api/version", timeout=5)
            response.raise_for_status()
            version = response.json()["version"]
        except httpx.ConnectError as exc:
            raise LlmUnavailable(
                f"Ollama is not reachable at {host}. Start it with: ollama serve"
            ) from exc
        except httpx.HTTPError as exc:  # timeout, non-2xx, protocol error
            raise LlmUnavailable(f"Ollama at {host} did not answer /api/version: {exc}") from exc
        except (ValueError, KeyError, TypeError) as exc:  # not the JSON Ollama returns, e.g. a proxy page
            raise LlmUnavailable(f"{host}/api/version is not an Ollama server: {exc!r}") from exc
        if not isinstance(version, str):
            raise LlmUnavailable(f"{host}/api/version returned an unexpected version value: {version!r}")
        return version

    def __call__(self, sections: SdsSections) -> LlmResult:
        s = self.settings
        try:
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
        except httpx.ConnectError as exc:
            raise LlmUnavailable(f"Ollama at {s.host} is no longer reachable: {exc}") from exc
        except ResponseError as exc:  # non-2xx, e.g. 404 when the model is not pulled
            raise LlmUnavailable(f"Ollama rejected the request for model {s.model!r}: {exc}") from exc
        raw = response.message.content or ""
        duration = (response.total_duration or 0) / 1e9
        return LlmResult(
            parsed=LlmExtraction.model_validate_json(raw),
            raw=raw,
            duration_s=round(duration, 2),
        )
