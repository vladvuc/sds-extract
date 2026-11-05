# SDS extraction pipeline

Extracts the **trade name**, **revision date**, and **hazard statements** from Safety Data Sheet PDFs for import into a master data management system (MDM). A local Qwen model running in Ollama proposes the values. Deterministic checks then verify every value against the PDF text.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).

Please copy the documents from the assignment into `data/` folder, they are being treated as confidential files.

```bash
# one-time setup
brew install ollama             # or the installer from ollama.com
ollama serve                    # keep running in its own terminal
ollama pull qwen2.5:7b          # installing Qwen with 7 billion parameters
uv sync --extra ui

# run the pipeline (the main entry point)
uv run sds-extract extract                                  # reads data/*.pdf, writes output/
uv run sds-extract extract --model qwen3:8b --no-think

# inspect what is sent to the model for one PDF, no LLM call
uv run sds-extract sections data/<FILENAME>.pdf

# optional review UI on http://localhost:8080
uv run --extra ui sds-extract ui

# tests (run over every PDF in data/ with a stubbed model, so Ollama is not needed)
uv run pytest
```

## Output

| File                                | Content                                                                                                                                          |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `output/runs/<run_id>/config.json`  | Run settings recorded before processing: model/Ollama settings, prompt version, package/Python versions, platform, input folder.                 |
| `output/runs/<run_id>/results.json` | Run summary and one full record per PDF: fields, status, issues, model input, raw answer, timings.                                               |
| `output/runs/<run_id>/results.csv`  | MDM import file: one row per PDF with trade name, revision date (dd/mm/yyyy), comma-separated H-codes and EUH-codes, status, and issues.         |
| `output/runs/<run_id>/run.log`      | Full log of the run, including model answers and every validation issue.                                                                         |
| `output/history.jsonl`              | One line per finished run, oldest first: run id, times, model, counts and the extracted fields per PDF. A run is appended only when it finishes. |

`<run_id>` is the UTC time plus a random suffix (`2026-11-05T12-24-00Z_a1b2c3`)

### Output Labels

- `status` is `ok` when every check passed
- `needs_review` when a check found a disagreement
- `error` when the document could not be processed

## How it works

```
PDF ─► pypdf text ─► slice header, section 1.1, section 2.2 ─► Qwen (JSON schema, temperature 0) ─► validate ─► JSON / CSV / log
```

**Input discovery** (`pipeline.py`): All files are checked recursively for the PDF signature; names and extensions are ignored. Real PDFs are processed even with odd names or no extension; files without a PDF signature are skipped and logged; hidden files are ignored. Records store relative path and SHA-256.

1. **Text extraction:** `pypdf` reads content-stream order, avoiding layout-merger issues; AES-encrypted PDFs without passwords work via `pypdf[crypto]` (`cryptography` ≥50.0.2).
2. **Section slicing** (`pdf.py`): Regex locates the header above 1.1, section 1.1, and section 2.2, so the model never sees section 3 (where H-codes could be confused with mixture codes) and input stays small/fast.
3. **LLM extraction** (`llm.py`): Ollama structured output is constrained by a Pydantic schema. The model returns the date as written and Python formats it.
4. **Validation** (`validate.py`): Model values are checked against source: trade name verbatim in 1.1; revision date via regex labels; hazard codes absent from 2.2 are dropped, missed ones reported. Disagreement sets `needs_review`; nothing is silently corrected except a missing date, which is filled from the labelled date and flagged.

The UI in `ui.py` is optional. It shows the latest run listed in `output/history.jsonl` and starts `sds-extract extract` as a subprocess, so the CLI remains the only way to run the pipeline.

- To remove the UI, delete `ui.py` and the `ui` extra.

## Decisions and assumptions

- **Confidentiality clause** - The documents in the assignment are treated as confidential, so please copy the required documents in the `data/` folder.
- **H-codes and EUH-codes are separate attributes** - The assignment asks for "the list of H-codes" but defines the format as `H\d{3}` or `EUH\d{3}`. To stay within the requirement without losing data, `hazard_statements` holds H-codes and `supplemental_hazard_statements` holds the EUH-codes. The name comes from section 2.2, including codes under "Aanvullende gegevens" (supplementary information). The Griffon silicone states "Gevarenaanduidingen: Niet van toepassing" but lists EUH208 and EUH210, so its hazard statements are empty and its supplemental hazard statements are `EUH208, EUH210`.
- **Only mixture codes from section 2.2** - Ingredient classifications in section 3 and section 16 are ignored.
- **Code suffixes are kept** - Codes such as `H360Fd` or `H361f` keep their letters. They are valid codes for Classificaiton, Labelling and Packaging, and dropping the suffix would lose meaning.
- **Hazard codes follow document order** and are deduplicated.
- **The trade name is copied exactly** - Original case and language are kept. The page-header article line, for example `BIS CONTACT KIT TR BD 100ML*6 NLFR`, is not the trade name.
- **Revision date**: used the labelled revision date, ignored print, issue, and replaced-version dates. Ambiguous numeric dates are day first.
- **An empty hazard list with status `ok`** means the product has no label hazard codes. It does not mean extraction failed.
- **File names are never used as data** - The model does not see them, and the tests identify documents by content hash.
- **Local model** - The assignment allows local models, and SDS content stays on the machine.
- **Installed Cryptography dependency** - some suppliers SDSs documents are encrypted with AES, and pypdf cannot read them without the package.

## Questions for stakeholders before production

1. Are these documents and information classified/confidential or am I freely able to use it, and send it to non-local models?
2. Should EUH codes, particularly EUH208 and EUH210, remain in their own attribute as now, or should they be combined with the H-codes?
3. Should codes from section 2.1 (classification) be used when section 2.2 is empty or says "not applicable"?
4. How are multilingual SDSs, or one SDS covering several products, handled? Which language wins for the trade name?
5. Should the hazard attribute be a comma-separated string or a list in the MDM import?
6. Any other standards which I need to follow for the MDM solution?
7. Who reviews `needs_review` records, and should those records block import?
8. Which SDS version is authoritative when the supplier's newest revision differs from the one in the MDM?

## Test data

The pipeline was run on two datasets, in this order:

1. **Download data.** 18 publicly available SDSs downloaded from the internet, in Dutch, German and English. They are stored in `dl-data`.

- `sources.csv` in the same folder lists each file's source URL.

2. **The provided assessment documents.** The three SDSs supplied with the assignment, they should be placed in `data/`.

### Usage

Default: read `data/`, write to `output/`.

```bash
uv run sds-extract extract
```

Custom, for a specific dataset: pick the input folder and keep its runs apart.

```bash
uv run sds-extract extract --pdf-dir dl-data --out output/dl-data
uv run sds-extract extract --pdf-dir data --out output/provided
```

- Each dataset gets its own `runs/` folder and `history.jsonl`.

## Known limitations

- **Text-based PDFs only** - Scanned SDSs need OCR, which is not implemented.
- **Section headings must follow the standard numbering** - Slicing expects "1.1" and "2.2". When a section is not found, the model receives `(not found)` for it, the checks fall back to the whole text, and the record is flagged.
- **Revision labels cover NL, EN and DE** - Other languages still work through the model, but without the date cross-check.
- **Tested on 21 SDSs only** - See [Test data](#test-data). Other supplier layouts may need new heading or label patterns.
- **Password-protected PDFs are not supported.** Encrypted PDFs open only when no user password is needed.
- **Speed depends on hardware.** A 7B model on a laptop takes several seconds per document.
- **The hidden text layer** that pypdf skips is an observation from these PDFs, not a guarantee for every supplier.

## Run details

Filled from `config.json` in the run folder that produced the submitted output.

| Setting            | Value                                                              |
| ------------------ | ------------------------------------------------------------------ |
| Model              | `qwen2.5:7b` (Ollama tag)                                          |
| Ollama version     | see `ollama_version` in `config.json`                              |
| Temperature / seed | 0 / 56                                                             |
| Context window     | 8192 tokens; inputs are about 1 to 2 thousand tokens               |
| Output format      | JSON schema from the `LlmExtraction` Pydantic model                |
| Prompt             | `src/sds_extract/prompts/system_v2.j2` and `user.j2`, version `v2` |
| Platform           | see `run.platform` and `run.python`                                |

**Prompt design.** The system prompt lists multilingual labels, names the dates to ignore (print, issue, replaced-version), requires exact copying, and allows `null` instead of guessing. The user message contains only the three labelled excerpts.

**Prompt templates.** Jinja2 templates live in `src/sds_extract/prompts/`: one system prompt per version used, plus `user.j2`.
`PROMPT_VERSION` in `llm.py` selects the system prompt and is recorded in `config.json`.

- `system_v1.j2` keeps v1 reproducible (`PROMPT_VERSION = "v1"`)
- `system_v2.j2` adds German/English labels, more ignored dates, and a rule for section 1.1 without a trade-name label.

## Use of AI assistants

**Claude Code** helped explore the PDFs, compare text extractors. **Pi Coding Agent** and **Claude Code** help write the code, tests and review.

The output was verified in these ways:

- Expected values were checked by hand against the PDFs and stored in `tests/expected.json`.
- A test feeds deliberately wrong model output, such as the print date, an ingredient code and the article line, and asserts that validation catches each mistake.
- The CLI, the section slicing and the "Ollama not reachable" path were run manually.

### Total time

- Research and implementation ~ 1 hour
- Fine tuning, bug fix and documentation ~ 1.5 hour

## Tests and adding documents

`uv run pytest` runs over every PDF it finds in `data/`, whatever the names are:

- **Every PDF must be supported.** Text must be extractable, sections 1.1 and 2.2 and a labelled revision date must be found, and every hazard code must have a valid format. A failure names the file and the missing piece.
- **Expected values are matched by content.** `tests/expected.json` holds the hand-checked values, keyed by the SHA-256 of each PDF. Renaming or moving a file does not break anything. A new PDF without an entry still gets the generic checks, and its value comparison is skipped.
- **File names do not matter.** A test copies PDFs under odd names, without an extension, into subfolders and as duplicates. It also adds non-PDF files, and checks that only the real PDFs are processed.
- **Validation catches model mistakes.** A test feeds wrong answers and checks that each one is flagged.

To add a document, drop it into `data/` and run the tests. Then add its SHA-256 and the values checked by hand to `tests/expected.json`.

## Versioning and releases

The version follows [Semantic Versioning](https://semver.org) and is set by
[python-semantic-release](https://python-semantic-release.readthedocs.io) from
[Conventional Commits](https://www.conventionalcommits.org). Never edit the version by hand.

## Next steps

1. Score each model run against `tests/expected.json` to measure accuracy per field and per model, for example `qwen2.5:7b` against `qwen3:8b`.
2. Test results and output logs on the frontier labs models.
3. Add OCR fallback for scanned PDFs and more heading and label patterns for FR, DE and other languages.
4. Add a retry loop: when validation fails, send the issue back to the model once before flagging.
5. Output an MDM-specific import schema with product ID mapping, and track SDS revisions over time.
