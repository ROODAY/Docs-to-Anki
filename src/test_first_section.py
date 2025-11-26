import sys
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ensure project root is on sys.path so `from src...` imports work when run as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.processor import find_sections
from src.openai_client import call_openai_for_section, make_prompt_for_section, FALLBACK_MODELS
import openai
import os

# ensure openai api key is set in this process
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.test_first_section <markdown-file>")
        sys.exit(1)
    path = sys.argv[1]
    with open(path, 'r', encoding='utf-8') as f:
        md = f.read()
    sections = find_sections(md)
    if not sections:
        print("No sections found")
        return
    first = sections[0]
    date_str = first.get('date_str') or 'unknown'
    print(f"Processing first section: {date_str}")
    try:
        cards = call_openai_for_section(first.get('content',''), date_str)
        print(json.dumps(cards, ensure_ascii=False, indent=2))
        result = cards
    except Exception as e:
        print('Error calling OpenAI:', e)
        print('\n--- Raw response follow-up attempt ---')
        # Attempt a raw call to inspect the exact response content
        messages = make_prompt_for_section(first.get('content',''), date_str)
        configured = os.getenv('OPENAI_MODEL', 'gpt5-mini')
        models_to_try = [configured] + FALLBACK_MODELS
        succeeded = False
        for m in models_to_try:
            print(f"Trying model: {m}")
            try:
                if hasattr(openai, 'chat') and hasattr(openai.chat, 'completions'):
                    resp = openai.chat.completions.create(model=m, messages=messages, max_tokens=1200, temperature=0.2)
                else:
                    resp = openai.ChatCompletion.create(model=m, messages=messages, max_tokens=1200, temperature=0.2)

                # try attribute access
                raw = None
                try:
                    choices = getattr(resp, 'choices', None)
                    if choices:
                        raw = getattr(choices[0].message, 'content', None)
                except Exception:
                    pass
                if not raw:
                    try:
                        raw = resp['choices'][0]['message']['content']
                    except Exception:
                        raw = None

                print('RAW CONTENT:')
                print(raw)
                print('\nAttempting JSON parse of raw content...')
                # Clean common markdown code fences if present
                cleaned = raw
                if cleaned is None:
                    cleaned = ''
                # remove leading/trailing triple-backtick fences
                if cleaned.strip().startswith('```'):
                    # remove first line (the fence) and last fence if present
                    parts = cleaned.splitlines()
                    # drop first line
                    parts = parts[1:]
                    # drop last line if it's a fence
                    if parts and parts[-1].strip().startswith('```'):
                        parts = parts[:-1]
                    cleaned = '\n'.join(parts)
                # also try to find first '[' and last ']' substring
                start = cleaned.find('[')
                end = cleaned.rfind(']')
                if start != -1 and end != -1:
                    cleaned = cleaned[start:end+1]
                try:
                    parsed = json.loads(cleaned)
                    print(json.dumps(parsed, ensure_ascii=False, indent=2))
                    succeeded = True
                    break
                except Exception as e2:
                    print('JSON parse failed after cleaning:', e2)
            except Exception as e3:
                print('Raw call failed:', e3)
        if not succeeded:
            print('All model attempts failed or returned unparseable content.')
            result = None
        else:
            result = parsed

    # If we have result cards, write a TSV for testing import into Anki
    if result:
        out_path = ROOT / 'cards_test.tsv'
        import csv
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            for c in result:
                front = c.get('front') if isinstance(c, dict) else ''
                back = c.get('back') if isinstance(c, dict) else ''
                tags = c.get('tags') if isinstance(c, dict) else ''
                # normalize tags: ensure comma-separated string
                if isinstance(tags, list):
                    tags = ','.join(tags)
                writer.writerow([front or '', back or '', tags or 'tamil'])
        print(f"Wrote TSV preview to: {out_path}")

if __name__ == '__main__':
    main()
