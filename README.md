# Docs-to-Anki

Convert exported Google Docs (markdown) lessons into Anki importable cards using OpenAI (gpt-5-mini).

Quick overview

- Export your Google Doc as Markdown.
- Run the processor with `python -m src.main --input <file.md>`.
- The script splits the markdown into lesson sections using separator lines containing a date (e.g. lots of slashes and `08/09/2025`).
- New lessons (after the last checkpoint) are sent to OpenAI to produce cards. Cards are aggregated into a single TSV file for Anki import. Optionally produce a `.apkg` if `genanki` is installed.

Assumptions

- Date format in the separator is parsed as `MM/DD/YYYY`.

Files

- `src/processor.py` - markdown splitting & checkpointing
- `src/openai_client.py` - OpenAI interaction
- `src/anki_writer.py` - write TSV and optional .apkg
- `src/main.py` - CLI
- `.env.example` - example env vars

Usage

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and optional `OPENAI_MODEL`.
2. Install dependencies:

```pwsh
pip install -r requirements.txt
```

3. Run:

```pwsh
python -m src.main --input lessons.md --out cards.tsv
```

This will create `cards.tsv` and update `checkpoint.json`.

Notes

- The tool tries to be token-efficient: it asks the model to return a compact JSON array of cards only.
- Checkpointing avoids reprocessing lessons that are already processed.
- If you want `.apkg` generation, install `genanki` and run with `--genanki` flag.
