#!/usr/bin/env python3
"""Claude API を使って未翻訳・陳腐化したページの下訳を作る。

生成された訳は必ず人手でレビューしてからマージすること。

    export ANTHROPIC_API_KEY=...
    python scripts/translate.py --limit 3
    python scripts/translate.py --only 'features/*'
    python scripts/translate.py --dry-run --limit 1   # 対象選定だけ確認する

依存: `pip install -r requirements-tools.txt`
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
STATE_FILE = ROOT_DIR / "translation" / "state.json"
GLOSSARY_FILE = ROOT_DIR / "translation" / "glossary.md"
UPSTREAM_DOCS = ROOT_DIR / "upstream" / "docs"
DOCS_DIR = ROOT_DIR / "docs"

DEFAULT_MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
あなたは vLLM（LLM 推論・サービングエンジン）の公式ドキュメントを日本語に翻訳する
技術翻訳者です。入力された Markdown を日本語に翻訳し、翻訳後の Markdown のみを出力
してください。前置き・後書き・コードフェンスでの囲みは一切不要です。

守るべきルール:

1. 文体は敬体（です・ます調）。技術文書として簡潔に訳す。
2. Markdown の構造（見出しレベル、箇条書き、表、引用、admonition `!!! note` など）を
   原文どおり保つ。行数や段落の対応関係を大きく崩さない。
3. 次のものは翻訳せず原文のまま残す:
   - コードブロックの中身、インラインコード、コマンド、環境変数、フラグ
   - YAML front matter のキーと値
   - リンクの URL、画像の URL、`--8<--` スニペット指示、HTML タグと属性
   - 見出しの明示的なアンカー指定（`{ #anchor }` など）
4. 見出しは日本語に訳してよいが、他ページから参照されうるため、原文の見出しを
   `<!-- 原文: Original Heading -->` のようなコメントで残さないこと（不要）。
5. リンクテキストは訳す。リンク先が英語ページであっても URL は変更しない。
6. 用語は後述の用語集に従う。用語集にない専門用語は、初出時に
   「日本語訳（English）」の形で併記してもよい。
7. 訳しにくい固有名詞・製品名・API 名は原語のままにする。
"""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def classify(entry: dict) -> str:
    if entry.get("status") != "translated":
        return "untranslated"
    if entry.get("source_sha") != entry.get("upstream_sha"):
        return "stale"
    return "translated"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="対象を絞る glob (例: 'features/*')")
    parser.add_argument("--limit", type=int, default=None, help="翻訳するページ数の上限")
    parser.add_argument(
        "--include-stale",
        action="store_true",
        help="陳腐化した翻訳済みページも訳し直す（既存の訳は上書きされる）",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="使用するモデル ID")
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument(
        "--dry-run", action="store_true", help="API を呼ばず対象一覧だけ表示する"
    )
    args = parser.parse_args()

    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    pages: dict[str, dict] = state.get("pages", {})

    targets: list[str] = []
    for path, entry in sorted(pages.items()):
        status = classify(entry)
        if status == "translated":
            continue
        if status == "stale" and not args.include_stale:
            continue
        if args.only and not fnmatch.fnmatch(path, args.only):
            continue
        targets.append(path)
    if args.limit is not None:
        targets = targets[: args.limit]

    if not targets:
        print("翻訳対象のページはありません。")
        return 0

    print(f"翻訳対象: {len(targets)} ページ")
    for path in targets:
        print(f"  - {path}")

    if args.dry_run:
        print("\n--dry-run のため API は呼び出しませんでした。")
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY が設定されていません。", file=sys.stderr)
        return 1

    try:
        import anthropic
    except ImportError:
        print(
            "anthropic パッケージが必要です: pip install -r requirements-tools.txt",
            file=sys.stderr,
        )
        return 1

    glossary = GLOSSARY_FILE.read_text(encoding="utf-8")
    system = f"{SYSTEM_PROMPT}\n\n# 用語集\n\n{glossary}"
    client = anthropic.Anthropic()

    for path in targets:
        source = (UPSTREAM_DOCS / path).read_text(encoding="utf-8")
        print(f"翻訳中: {path} ({len(source)} 文字)")
        message = client.messages.create(
            model=args.model,
            max_tokens=args.max_tokens,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"次の vLLM ドキュメント (`docs/{path}`) を日本語に翻訳して"
                        f"ください。\n\n<document>\n{source}\n</document>"
                    ),
                }
            ],
        )
        translated = "".join(
            block.text for block in message.content if block.type == "text"
        ).strip()
        if not translated:
            print(f"  警告: 空の応答でした。スキップします: {path}", file=sys.stderr)
            continue

        target_file = DOCS_DIR / path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(translated + "\n", encoding="utf-8")

        entry = pages[path]
        entry["status"] = "translated"
        entry["source_sha"] = sha256(source)
        entry["upstream_sha"] = sha256(source)
        entry["translated_at"] = date.today().isoformat()
        entry["translated_by"] = f"script:{args.model}"
        STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"  完了: docs/{path}")

    print("\n下訳が完了しました。必ず内容をレビューしてからコミットしてください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
