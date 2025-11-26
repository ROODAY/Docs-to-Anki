import argparse
import os
from dotenv import load_dotenv
from src.processor import find_sections, Checkpoint
from src.openai_client import call_openai_for_section
from src.anki_writer import write_tsv, build_apkg
from tqdm import tqdm

load_dotenv()


def process_file(input_path: str, out_tsv: str, checkpoint_file: str, genanki_flag: bool = False, dry_run: bool = False):
    with open(input_path, "r", encoding="utf-8") as f:
        md = f.read()

    sections = find_sections(md)
    cp = Checkpoint(checkpoint_file)

    all_cards = []
    for s in tqdm(sections, desc="Sections"):
        date_str = s.get("date_str") or "unknown"
        if cp.is_processed(date_str):
            continue
        if dry_run:
            print(f"[dry-run] would process section: {date_str}")
            continue
        # Call OpenAI
        try:
            cards = call_openai_for_section(s.get("content", ""), date_str)
            # normalize
            for c in cards:
                if "tags" not in c:
                    c["tags"] = f"tamil,{date_str}"
            all_cards.extend(cards)
            cp.mark_processed(date_str)
        except Exception as e:
            print(f"Failed processing section {date_str}: {e}")

    if all_cards:
        write_tsv(all_cards, out_tsv)
        if genanki_flag:
            try:
                build_apkg(all_cards, out_path=os.path.splitext(out_tsv)[0] + ".apkg")
            except Exception as e:
                print("genanki failed:", e)
    else:
        print("No new cards to write.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input markdown file")
    p.add_argument("--out", default=os.getenv("OUTPUT_TSV", "cards.tsv"))
    p.add_argument("--checkpoint", default=os.getenv("CHECKPOINT_FILE", "checkpoint.json"))
    p.add_argument("--genanki", action="store_true", help="Also build .apkg (requires genanki)")
    p.add_argument("--dry-run", action="store_true", help="Parse file and show sections without calling OpenAI")
    args = p.parse_args()
    process_file(args.input, args.out, args.checkpoint, args.genanki, dry_run=args.dry_run)
