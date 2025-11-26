import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import os
import csv
from src.processor import find_sections, Checkpoint
from src.openai_client import call_openai_for_section


def process_limited(input_path: str, out_tsv: str, checkpoint_file: str, limit: int = 5):
    with open(input_path, "r", encoding="utf-8") as f:
        md = f.read()
    sections = find_sections(md)
    cp = Checkpoint(checkpoint_file)

    all_cards = []
    processed = 0
    for s in sections:
        if processed >= limit:
            break
        date_str = s.get("date_str") or "unknown"
        if cp.is_processed(date_str):
            continue
        try:
            cards = call_openai_for_section(s.get("content", ""), date_str)
            for c in cards:
                if "tags" not in c:
                    c["tags"] = f"tamil,{date_str}"
            all_cards.extend(cards)
            cp.mark_processed(date_str)
            processed += 1
        except Exception as e:
            print(f"Failed processing section {date_str}: {e}")

    if all_cards:
        with open(out_tsv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            for c in all_cards:
                front = c.get("front", "")
                back = c.get("back", "")
                tags = c.get("tags", "tamil")
                if isinstance(tags, list):
                    tags = ",".join(tags)
                writer.writerow([front, back, tags])
        print(f"Wrote {len(all_cards)} cards to {out_tsv}")
    else:
        print("No new cards written.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", default=os.getenv("OUTPUT_TSV", "cards.tsv"))
    p.add_argument("--checkpoint", default=os.getenv("CHECKPOINT_FILE", "checkpoint.json"))
    p.add_argument("--limit", type=int, default=5)
    args = p.parse_args()
    process_limited(args.input, args.out, args.checkpoint, limit=args.limit)
