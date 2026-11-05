"""Command-line entry point. Every extraction, including the ones started from the UI, runs through `sds-extract extract`."""

import argparse
import logging
import sys
from importlib.metadata import version
from pathlib import Path

from .llm import LlmSettings, LlmUnavailable, OllamaExtractor

log = logging.getLogger("sds_extract")


def _settings(args: argparse.Namespace) -> LlmSettings:
    return LlmSettings(
        model=args.model, host=args.host, num_ctx=args.num_ctx, think=args.think
    )


def _extract(args: argparse.Namespace) -> int:
    from .pipeline import run

    settings = _settings(args)
    extractor = OllamaExtractor(settings)
    try:
        run_info = {"llm": settings.as_dict(), "ollama_version": extractor.server_version()}
        result = run(args.pdf_dir, args.out, extractor, run_info)
    except LlmUnavailable as exc:
        log.error("%s", exc)
        return 2
    return 1 if result["run"]["counts"]["error"] else 0


def _sections(args: argparse.Namespace) -> int:
    from .pdf import load_sections

    s = load_sections(args.pdf)
    for title, text in (
        ("HEADER", s.header),
        ("SECTION 1.1", s.product_identifier),
        ("SECTION 2.2", s.label_elements),
    ):
        print(f"===== {title}\n{text or '(not found)'}\n")
    return 0


def _ui(args: argparse.Namespace) -> int:
    try:
        from .ui import serve
    except ImportError:
        print(
            "The UI needs the 'ui' extra: uv run --extra ui sds-extract ui",
            file=sys.stderr,
        )
        return 2
    serve(pdf_dir=args.pdf_dir, out_dir=args.out, port=args.port, settings=_settings(args))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sds-extract", description="Extract master data from Safety Data Sheets."
    )
    p.add_argument(
        "--version", action="version", version=f"%(prog)s {version('sds-extract')}"
    )
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        """Options shared by `extract` and `ui`; the UI forwards them to the `extract` subprocess."""
        sp.add_argument(
            "--pdf-dir",
            type=Path,
            default=Path("data"),
            help="folder with SDS PDFs (default: data)",
        )
        sp.add_argument(
            "--out",
            type=Path,
            default=Path("output"),
            help="output folder (default: output)",
        )
        sp.add_argument(
            "--model",
            default=LlmSettings.model,
            help=f"Ollama model (default: {LlmSettings.model})",
        )
        sp.add_argument("--host", default=LlmSettings.host, help="Ollama URL")
        sp.add_argument(
            "--num-ctx",
            type=int,
            default=LlmSettings.num_ctx,
            help="context window in tokens",
        )
        sp.add_argument(
            "--think",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="enable/disable reasoning for thinking models such as qwen3 (default: not sent)",
        )

    ex = sub.add_parser(
        "extract", help="run the pipeline into output/runs/<run_id>/ and append to output/history.jsonl"
    )
    common(ex)
    ex.set_defaults(func=_extract)

    se = sub.add_parser(
        "sections", help="print the text slices sent to the model, no LLM call"
    )
    se.add_argument("pdf", type=Path)
    se.set_defaults(func=_sections)

    ui = sub.add_parser(
        "ui", help="open the review UI; its Run button calls 'sds-extract extract'"
    )
    common(ui)
    ui.add_argument("--port", type=int, default=8080)
    ui.set_defaults(func=_ui)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # font-encoding notices, not actionable
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
