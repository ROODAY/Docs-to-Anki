import re
from datetime import datetime
from typing import List, Tuple, Optional
import json

SEP_DATE_RE = re.compile(r"(\d{1,2}/\d{1,2}/\d{4})")


def find_sections(markdown_text: str) -> List[dict]:
    """Split markdown into sections using separator lines that contain a date.

    A separator line is any line that contains a substring matching MM/DD/YYYY.
    We assume MM/DD/YYYY. Each section dict contains:
      - date_str: matched date string (as found)
      - date: parsed datetime (assumes MM/DD/YYYY)
      - content: markdown content of the section (excluding separator lines)

    Returns sections in original order.
    """
    lines = markdown_text.splitlines()
    sections = []
    current_lines = []
    current_date = None

    def flush():
        nonlocal current_date, current_lines
        if current_date or current_lines:
            content = "\n".join(current_lines).strip()
            sections.append({
                "date_str": current_date,
                "date": _parse_date(current_date) if current_date else None,
                "content": content,
            })
            current_lines = []
            current_date = None

    for ln in lines:
        m = SEP_DATE_RE.search(ln)
        if m:
            # separator encountered
            # flush prior block as a section
            flush()
            current_date = m.group(1)
            continue
        current_lines.append(ln)
    flush()
    # Filter out empty sections
    sections = [s for s in sections if s.get("content")]
    return sections


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    # try MM/DD/YYYY
    for fmt in ("%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except Exception:
            continue
    return None


# Checkpointing

class Checkpoint:
    def __init__(self, path: str = "checkpoint.json"):
        self.path = path
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception:
            self.data = {"last_date": None, "processed": []}

    def last_date(self) -> Optional[str]:
        return self.data.get("last_date")

    def is_processed(self, date_str: str) -> bool:
        return date_str in self.data.get("processed", [])

    def mark_processed(self, date_str: str):
        if "processed" not in self.data:
            self.data["processed"] = []
        if date_str not in self.data["processed"]:
            self.data["processed"].append(date_str)
            # update last_date to the newest by parsing
            try:
                dates = [d for d in self.data["processed"] if d]
                parsed = [_parse_date(d) for d in dates]
                parsed = [p for p in parsed if p]
                if parsed:
                    latest = max(parsed)
                    self.data["last_date"] = latest.strftime("%m/%d/%Y")
            except Exception:
                pass
            self._save()

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
