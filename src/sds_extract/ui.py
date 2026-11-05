"""Review UI (optional). It only reads the latest run listed in output/history.jsonl and starts `sds-extract extract` as a subprocess.

Delete this file and the `ui` extra to drop the UI; the pipeline does not depend on it.
"""

import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import quote

from nicegui import app, ui

from .llm import LlmSettings
from .pipeline import load_latest

STATUS_STYLE = {
    "ok": ("OK", "bg-emerald-100 text-emerald-800"),
    "needs_review": ("Review", "bg-amber-100 text-amber-800"),
    "error": ("Error", "bg-rose-100 text-rose-800"),
}
# CLP numbering: H2xx physical, H3xx health, H4xx environment.
HAZARD_STYLE = {
    "H2": "bg-orange-100 text-orange-800",
    "H3": "bg-rose-100 text-rose-800",
    "H4": "bg-teal-100 text-teal-800",
}


def _badge(status: str) -> None:
    text, cls = STATUS_STYLE.get(status, (status, "bg-slate-100 text-slate-700"))
    ui.label(text).classes(f"{cls} text-xs font-medium px-2 py-0.5 rounded-full")


def _field(label: str, value: str | None, mono: bool = False) -> None:
    with ui.column().classes("gap-1"):
        ui.label(label).classes("text-xs uppercase tracking-wide text-slate-500")
        ui.label(value or "—").classes(
            ("font-mono " if mono else "") + "text-lg text-slate-900"
        )


def _codes(label: str, joined: str | None) -> None:
    """Comma-separated hazard codes as coloured chips; EUH codes get the neutral style."""
    codes = [c for c in (joined or "").split(", ") if c]
    with ui.column().classes("gap-1"):
        ui.label(label).classes("text-xs uppercase tracking-wide text-slate-500")
        with ui.row().classes("gap-1"):
            if not codes:
                ui.label("none").classes("text-lg text-slate-400")
            for c in codes:
                cls = HAZARD_STYLE.get(c[:2], "bg-slate-200 text-slate-800")
                ui.label(c).classes(f"{cls} font-mono text-sm px-2 py-0.5 rounded")


def serve(pdf_dir: Path, out_dir: Path, port: int, settings: LlmSettings) -> None:
    pdf_dir, out_dir = pdf_dir.resolve(), out_dir.resolve()
    app.add_static_files("/pdfs", pdf_dir)

    @ui.page("/")
    def index() -> None:
        state = {"data": load_latest(out_dir), "selected": 0, "running": False}
        ui.query("body").classes("bg-slate-50")
        ui.colors(primary="#0f172a")

        @ui.refreshable
        def run_summary() -> None:
            run = state["data"]["run"]
            if not run:
                ui.label("No results yet. Press Run extraction.").classes(
                    "text-sm text-slate-500"
                )
                return
            llm = run.get("llm", {})
            parts = [
                f"run {run['run_id']}",
                f"model {llm.get('model')}",
                f"temperature {llm.get('temperature')}",
                f"Ollama {run.get('ollama_version')}",
                f"{run.get('duration_s')}s",
            ]
            ui.label(" · ".join(parts)).classes("text-xs text-slate-500")

        @ui.refreshable
        def doc_list() -> None:
            for i, d in enumerate(state["data"]["documents"]):
                active = (
                    "ring-2 ring-slate-900"
                    if i == state["selected"]
                    else "hover:bg-slate-100"
                )
                with (
                    ui.card()
                    .tight()
                    .classes(
                        f"w-full cursor-pointer shadow-none border border-slate-200 {active}"
                    )
                    .on("click", lambda i=i: select(i))
                ):
                    with ui.column().classes("p-3 gap-1 w-full"):
                        with ui.row().classes(
                            "w-full items-center justify-between no-wrap"
                        ):
                            ui.label(d.get("trade_name") or "(no trade name)").classes(
                                "font-medium text-slate-900 truncate"
                            )
                            _badge(d["status"])
                        ui.label(d["source_file"]).classes(
                            "text-xs text-slate-500 font-mono"
                        )

        @ui.refreshable
        def detail() -> None:
            docs = state["data"]["documents"]
            if not docs:
                return
            d = docs[state["selected"]]
            with ui.card().classes("w-full shadow-none border border-slate-200"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(d["source_file"]).classes(
                        "font-mono text-sm text-slate-500"
                    )
                    _badge(d["status"])
                with ui.grid(columns=2).classes("w-full gap-6 pt-2"):
                    _field("Trade name", d.get("trade_name"))
                    _field("Revision date", d.get("revision_date"), mono=True)
                    _codes("Hazard statements (H)", d.get("hazard_statements"))
                    _codes(
                        "Supplemental hazard statements (EUH)",
                        d.get("supplemental_hazard_statements"),
                    )
                for issue in d.get("issues", []):
                    with ui.row().classes(
                        "items-start gap-2 bg-amber-50 text-amber-900 text-sm rounded p-2 w-full no-wrap"
                    ):
                        ui.icon("warning", size="xs").classes("mt-0.5")
                        ui.label(issue)

            with ui.tabs().classes("text-slate-700") as tabs:
                t_pdf, t_input, t_raw = (
                    ui.tab("Document"),
                    ui.tab("Model input"),
                    ui.tab("Model output"),
                )
            with ui.tab_panels(tabs, value=t_pdf).classes("w-full bg-transparent"):
                with ui.tab_panel(t_pdf).classes("p-0"):
                    # source_file can contain spaces, umlauts or subfolders; encode it for the URL.
                    ui.element("iframe").props(
                        f'src="/pdfs/{quote(d["source_file"])}"'
                    ).classes("w-full h-[75vh] rounded border border-slate-200")
                with ui.tab_panel(t_input).classes("p-0"):
                    for name, text in (d.get("sections") or {}).items():
                        ui.label(
                            f"Section {name}" if name != "header" else "Header"
                        ).classes("text-xs uppercase text-slate-500 pt-2")
                        ui.code(text or "(not found)", language="text").classes(
                            "w-full"
                        )
                with ui.tab_panel(t_raw).classes("p-0"):
                    raw = (d.get("llm") or {}).get("raw") or "{}"
                    try:
                        raw = json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
                    except json.JSONDecodeError:
                        pass
                    ui.code(raw, language="json").classes("w-full")
                    ui.label(
                        f"model time: {(d.get('llm') or {}).get('duration_s')}s"
                    ).classes("text-xs text-slate-500")

        def select(i: int) -> None:
            state["selected"] = i
            doc_list.refresh()
            detail.refresh()

        async def run_extraction() -> None:
            if state["running"]:
                return
            state["running"] = True
            run_button.props("loading")
            try:
                log_panel.clear()
                log_box.open()
                # Same options the UI was started with, so a non-default host/num_ctx/think reaches the run.
                cmd = [
                    sys.executable,
                    "-m",
                    "sds_extract",
                    "extract",
                    "--pdf-dir",
                    str(pdf_dir),
                    "--out",
                    str(out_dir),
                    "--model",
                    model_input.value,
                    "--host",
                    settings.host,
                    "--num-ctx",
                    str(settings.num_ctx),
                ]
                if settings.think is not None:
                    cmd.append("--think" if settings.think else "--no-think")
                log_panel.push("$ sds-extract " + " ".join(cmd[4:]))
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                assert proc.stdout
                async for line in proc.stdout:
                    log_panel.push(line.decode(errors="replace").rstrip())
                code = await proc.wait()
                state["data"], state["selected"] = load_latest(out_dir), 0
            finally:  # whatever failed, the button must become usable again
                state["running"] = False
                run_button.props(remove="loading")
            run_summary.refresh()
            doc_list.refresh()
            detail.refresh()
            ui.notify(
                f"Extraction finished with exit code {code}",
                type="positive" if code == 0 else "warning",
            )

        with ui.header().classes(
            "bg-white text-slate-900 border-b border-slate-200 px-6 py-3 items-center justify-between"
        ):
            with ui.column().classes("gap-0"):
                ui.label("SDS extraction review").classes("text-lg font-semibold")
                run_summary()
            with ui.row().classes("items-center gap-2"):
                model_input = (
                    ui.input("Ollama model", value=settings.model)
                    .props("dense outlined")
                    .classes("w-48")
                )
                run_button = ui.button(
                    "Run extraction", icon="play_arrow", on_click=run_extraction
                ).props("unelevated no-caps")

        with ui.row().classes("w-full no-wrap gap-6 p-6 items-start"):
            with ui.column().classes("w-80 shrink-0 gap-2"):
                doc_list()
            with ui.column().classes("grow min-w-0 gap-4"):
                with ui.expansion("Run log", icon="terminal").classes(
                    "w-full bg-white border border-slate-200 rounded"
                ) as log_box:
                    log_panel = ui.log(max_lines=1000).classes("w-full h-64 text-xs")
                detail()

    # Loopback only: the page serves the PDF folder and can start extraction runs.
    ui.run(
        host="127.0.0.1",
        port=port,
        title="SDS extraction review",
        reload=False,
        show=False,
        favicon="📁",
    )
