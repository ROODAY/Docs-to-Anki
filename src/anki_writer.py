from typing import List, Dict
import csv

try:
    import genanki
    HAVE_GENANKI = True
except Exception:
    HAVE_GENANKI = False


def write_tsv(cards: List[Dict], out_path: str = "cards.tsv"):
    # Cards expected: list of dicts with 'front', 'back', optional 'tags'
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for c in cards:
            front = c.get("front", "")
            back = c.get("back", "")
            tags = c.get("tags", "tamil")
            writer.writerow([front, back, tags])


def build_apkg(cards: List[Dict], deck_name: str = "Docs-to-Anki", out_path: str = "cards.apkg"):
    if not HAVE_GENANKI:
        raise RuntimeError("genanki not installed. Install it or use TSV output.")
    my_deck = genanki.Deck(2059400110, deck_name)
    my_model = genanki.Model(
        1607392319,
        'Simple Model',
        fields=[
            {'name': 'Front'},
            {'name': 'Back'},
            {'name': 'Tags'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '{{Front}}',
                'afmt': '{{Front}}<hr id="answer">{{Back}}',
            },
        ])
    for c in cards:
        f = c.get('front', '')
        b = c.get('back', '')
        tags = [t.strip() for t in (c.get('tags') or '').split(',') if t.strip()]
        note = genanki.Note(model=my_model, fields=[f, b, ",".join(tags)])
        my_deck.add_note(note)
    genanki.Package(my_deck).write_to_file(out_path)
