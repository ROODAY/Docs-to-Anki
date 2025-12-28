import argparse
import os
import logging
from dotenv import load_dotenv
from src.processor import find_sections
import asyncio
from src.openai_client import call_openai_for_section, OUTPUT_DIR
from src.anki_writer import write_tsv, build_apkg
from tqdm import tqdm
import glob
import json
import pathlib
import os

load_dotenv()


async def process_file(
    input_path: str,
    out_tsv: str,
    checkpoint_file: str,
    genanki_flag: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    mode: str = "process",
    max_completion_tokens: int = 4000,
):
    with open(input_path, "r", encoding="utf-8") as f:
        md = f.read()

    sections = find_sections(md)

    logger = logging.getLogger("docs_to_anki.main")
    logger.info(
        "Starting processing; mode=%s, limit=%s, max_completion_tokens=%d",
        mode,
        (limit if limit is not None else "no limit"),
        max_completion_tokens,
    )

    # Ensure outputs dir exists
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if mode == "assemble":
        # Read all JSON outputs and aggregate into cards, then write TSV / apkg
        files = glob.glob(os.path.join(OUTPUT_DIR, "*.json"))
        all_cards = []
        for fn in files:
            try:
                with open(fn, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                cards = payload.get("cards", [])
                all_cards.extend(cards)
            except Exception as e:
                logger.warning("Failed to read output file %s: %s", fn, e)

        if all_cards:
            write_tsv(all_cards, out_tsv)
            if genanki_flag:
                try:
                    build_apkg(
                        all_cards, out_path=os.path.splitext(out_tsv)[0] + ".apkg"
                    )
                except Exception as e:
                    print("genanki failed:", e)
        else:
            print("No output JSON files found to assemble.")
        return

    # mode == 'process'
    # Determine which sections already have outputs and skip them
    existing = set()
    for p in glob.glob(os.path.join(OUTPUT_DIR, "*.json")):
        name = os.path.splitext(os.path.basename(p))[0]
        # convert back to date format with slashes
        existing.add(name.replace("-", "/"))

    to_process = [
        s for s in sections if (s.get("date_str") or "unknown") not in existing
    ]
    logger.info(
        "Sections to process: %d (skipping %d existing)",
        len(to_process),
        len(sections) - len(to_process),
    )

    all_cards = []
    processed = 0
    for s in tqdm(to_process, desc="Sections"):
        date_str = s.get("date_str") or "unknown"
        if limit is not None and processed >= limit:
            logger.info("Reached processing limit (%d); stopping", limit)
            break
        if dry_run:
            print(f"[dry-run] would process section: {date_str}")
            continue
        # Call OpenAI
        try:
            cards = await call_openai_for_section(
                s.get("content", ""),
                date_str,
                max_retries=3,
                max_completion_tokens=max_completion_tokens,
            )
            # normalize
            for c in cards:
                if "tags" not in c:
                    c["tags"] = f"tamil,{date_str}"
            # write per-section success to outputs folder immediately
            safe_date = (date_str or "unknown").replace("/", "-")
            out_path = os.path.join(OUTPUT_DIR, f"{safe_date}.json")
            try:
                payload = {"date": date_str, "cards": cards}
                with open(out_path, "w", encoding="utf-8") as _f:
                    json.dump(payload, _f, ensure_ascii=False, indent=2)
                logger.info("Wrote successful section output to %s", out_path)
            except Exception as _e:
                logger.warning(
                    "Failed to write section output for %s: %s", date_str, _e
                )

            all_cards.extend(cards)
            processed += 1
        except Exception as e:
            print(f"Failed processing section {date_str}: {e}")

    # In process mode we persist per-section outputs only; do not write aggregate TSV/APKG here.
    logger.info("Processing complete. Processed %d new sections.", processed)
    if processed == 0:
        print("No new sections were processed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input markdown file")
    p.add_argument("--out", default=os.getenv("OUTPUT_TSV", "cards.tsv"))
    p.add_argument(
        "--checkpoint", default=os.getenv("CHECKPOINT_FILE", "checkpoint.json")
    )
    p.add_argument(
        "--genanki", action="store_true", help="Also build .apkg (requires genanki)"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse file and show sections without calling OpenAI",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of new lessons to process",
    )
    p.add_argument(
        "--mode",
        choices=["process", "assemble"],
        default="process",
        help="Operation mode: 'process' calls OpenAI and writes per-section outputs; 'assemble' builds the final TSV/APKG from outputs/",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Max completion tokens to request from the model (default: 4000)",
    )
    args = p.parse_args()
    asyncio.run(
        process_file(
            args.input,
            args.out,
            args.checkpoint,
            args.genanki,
            dry_run=args.dry_run,
            limit=args.limit,
            mode=args.mode,
            max_completion_tokens=args.max_tokens,
        )
    )
