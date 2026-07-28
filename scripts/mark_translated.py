#!/usr/bin/env python3
"""手作業で翻訳したページを「翻訳済み」として `translation/state.json` に記録する。

    python scripts/mark_translated.py getting_started/quickstart.md usage/faq.md
    python scripts/mark_translated.py --by "手動" 'features/*.md'
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
STATE_FILE = ROOT_DIR / "translation" / "state.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="docs/ からの相対パス（glob 可）")
    parser.add_argument("--by", default="manual", help="翻訳者の記録 (translated_by)")
    args = parser.parse_args()

    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    pages: dict[str, dict] = state["pages"]

    matched: list[str] = []
    for pattern in args.paths:
        hits = [p for p in pages if p == pattern or fnmatch.fnmatch(p, pattern)]
        if not hits:
            print(f"該当なし: {pattern}")
        matched.extend(hits)

    for path in sorted(set(matched)):
        entry = pages[path]
        entry["status"] = "translated"
        entry["source_sha"] = entry["upstream_sha"]
        entry["translated_at"] = date.today().isoformat()
        entry["translated_by"] = args.by
        print(f"翻訳済みとして記録: {path}")

    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
