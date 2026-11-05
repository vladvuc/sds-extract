"""Command-line entry point. Every extraction, including the ones started from the UI, runs through `sds-extract extract`."""

import argparse
import logging
import sys
from pathlib import Path

from .llm import LlmSettings


def _extract(args: argparse.Namespace) -> int:
    from .llm import OllamaExtractor
    from .pipeline import run

    settings = LlmSettings(
        model=args.model, host=args.host, num_ctx=args.num_ctx, think=args.think
    )
    extractor = OllamaExtractor(settings)
    run_info = {"llm": settings.as_dict(), "ollama_version": extractor.server_version()}
    if run_info["ollama_version"] is None:
        logging.getLogger("sds_extract").error(
            "Ollama is not reachable at %s. Start it with: ollama serve", args.host
        )
        return 2
    result = run(args.pdf_dir, args.out, extractor, run_info)
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
    serve(pdf_dir=args.pdf_dir, out_dir=args.out, port=args.port, model=args.model)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sds-extract", description="Extract master data from Safety Data Sheets."
    )
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
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

    ex = sub.add_parser(
        "extract", help="run the pipeline into output/runs/<run_id>/ and append to output/history.jsonl"
    )
    common(ex)
    ex.add_argument("--host", default=LlmSettings.host, help="Ollama URL")
    ex.add_argument(
        "--num-ctx",
        type=int,
        default=LlmSettings.num_ctx,
        help="context window in tokens",
    )
    ex.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable reasoning for thinking models such as qwen3 (default: not sent)",
    )
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
