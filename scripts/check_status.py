#!/usr/bin/env python3
"""翻訳カバレッジを集計し、`docs/ja/status.md` を生成する。

    python scripts/check_status.py              # 集計してページを更新
    python scripts/check_status.py --print      # 標準出力に表示するだけ
    python scripts/check_status.py --fail-on-stale   # 陳腐化ページがあれば終了コード 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
STATE_FILE = ROOT_DIR / "translation" / "state.json"
OUTPUT = ROOT_DIR / "docs" / "ja" / "status.md"

SECTION_LABELS = {
    ".": "トップページ",
    "getting_started": "はじめに",
    "usage": "使い方",
    "serving": "推論とサービング",
    "deployment": "デプロイ",
    "training": "学習",
    "configuration": "設定",
    "models": "モデル",
    "features": "機能",
    "contributing": "開発者ガイド",
    "design": "設計ドキュメント",
    "benchmarking": "ベンチマーク",
    "cli": "CLI リファレンス",
    "community": "コミュニティ",
    "governance": "ガバナンス",
}


def classify(entry: dict) -> str:
    if entry.get("status") != "translated":
        return "untranslated"
    if entry.get("source_sha") != entry.get("upstream_sha"):
        return "stale"
    return "translated"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", action="store_true", help="ファイルを書かず表示する")
    parser.add_argument(
        "--fail-on-stale",
        action="store_true",
        help="陳腐化したページがあれば終了コード 1 を返す",
    )
    args = parser.parse_args()

    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    pages: dict[str, dict] = state.get("pages", {})
    ref = state.get("upstream", {}).get("ref", "unknown")
    synced_at = state.get("upstream", {}).get("synced_at", "")

    sections: dict[str, dict[str, int]] = {}
    stale_pages: list[str] = []
    translated_pages: list[str] = []
    for path, entry in sorted(pages.items()):
        section = path.split("/")[0] if "/" in path else "."
        counts = sections.setdefault(
            section, {"translated": 0, "stale": 0, "untranslated": 0}
        )
        status = classify(entry)
        counts[status] += 1
        if status == "stale":
            stale_pages.append(path)
        elif status == "translated":
            translated_pages.append(path)

    total = len(pages)
    done = sum(c["translated"] for c in sections.values())
    stale = sum(c["stale"] for c in sections.values())

    lines = [
        "# 翻訳状況",
        "",
        f"追随中の上流バージョン: **{ref}**"
        + (f"（最終同期: {synced_at}）" if synced_at else ""),
        "",
        f"全 {total} ページ中 **{done} ページ**を翻訳済みです"
        f"（{done / total * 100:.1f}%）。"
        + (f"うち {stale} ページは原文の更新に追随できていません。" if stale else ""),
        "",
        "未翻訳のページは英語原文をそのまま掲載しています。",
        "翻訳の追加・修正は [GitHub リポジトリ]"
        "(https://github.com/tatsuya-fukuoka/vllm-docs-ja) へ Pull Request を歓迎します。",
        "",
        "## セクション別",
        "",
        "| セクション | 翻訳済み | 要更新 | 未翻訳 | 合計 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for section in sorted(sections, key=lambda s: (s != ".", s)):
        counts = sections[section]
        label = SECTION_LABELS.get(section, section)
        section_total = sum(counts.values())
        lines.append(
            f"| {label} | {counts['translated']} | {counts['stale']} "
            f"| {counts['untranslated']} | {section_total} |"
        )
    lines += [
        f"| **合計** | **{done}** | **{stale}** "
        f"| **{total - done - stale}** | **{total}** |",
    ]

    if translated_pages:
        lines += ["", "## 翻訳済みのページ", ""]
        lines += [f"- `{path}`" for path in translated_pages]
    if stale_pages:
        lines += ["", "## 原文の更新に追随できていないページ", ""]
        lines += [f"- `{path}`" for path in stale_pages]

    content = "\n".join(lines) + "\n"
    if args.print:
        print(content)
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(content, encoding="utf-8")
        print(f"{OUTPUT.relative_to(ROOT_DIR)} を更新しました "
              f"(翻訳済み {done}/{total}, 要更新 {stale})")

    if args.fail_on_stale and stale_pages:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
