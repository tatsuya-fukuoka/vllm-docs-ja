"""各ページの冒頭に翻訳ステータスのバナーを自動挿入する MkDocs フック。

`translation/state.json` に登録されているページに対して、次の 3 状態を出し分ける。

- translated   : 日本語訳済み（原文へのリンクを添える）
- untranslated : 未翻訳（英語原文をそのまま表示していることを明示する）
- stale        : 翻訳済みだが原文が更新されている（内容が古い可能性を警告する）

`state.json` に存在しないページ（日本語版独自のページなど）にはバナーを挿入しない。
"""

import json
from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.pages import Page

ROOT_DIR = Path(__file__).parent.parent.resolve()
STATE_FILE = ROOT_DIR / "translation" / "state.json"

_state: dict | None = None


def _load_state() -> dict:
    global _state
    if _state is None:
        if STATE_FILE.exists():
            _state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        else:
            _state = {"upstream": {}, "pages": {}}
    return _state


def _upstream_page_url(config: MkDocsConfig, page: Page) -> str:
    """対応する上流（英語）ページの URL を返す。"""
    base = config["extra"].get("upstream_docs_url", "https://docs.vllm.ai/en/latest/")
    return f"{base.rstrip('/')}/{page.file.url}"


def _upstream_source_url(config: MkDocsConfig, src_uri: str) -> str:
    """対応する上流（英語）原文 Markdown の GitHub URL を返す。"""
    ref = config["extra"].get("upstream_ref", "main")
    return f"https://github.com/vllm-project/vllm/blob/{ref}/docs/{src_uri}"


def _banner(status: str, page_url: str, source_url: str, ref: str) -> str:
    links = f"[原文（英語）を表示]({page_url}) ・ [原文の Markdown]({source_url})"
    if status == "untranslated":
        return (
            '!!! warning "このページはまだ翻訳されていません"\n'
            "    以下は vLLM 公式ドキュメント "
            f"{ref} の英語原文です。翻訳の協力を歓迎します。\n"
            f"    {links}\n"
        )
    if status == "stale":
        return (
            '!!! warning "原文が更新されています"\n'
            "    このページの日本語訳は、上流の更新に追随できていません。"
            "最新の情報は原文を参照してください。\n"
            f"    {links}\n"
        )
    return (
        '!!! info "非公式日本語訳"\n'
        f"    このページは vLLM 公式ドキュメント {ref} の非公式日本語訳です。\n"
        f"    {links}\n"
    )


def on_page_markdown(
    markdown: str, *, page: Page, config: MkDocsConfig, files
) -> str:
    state = _load_state()
    entry = state.get("pages", {}).get(page.file.src_uri)
    if entry is None:
        return markdown

    status = entry.get("status", "untranslated")
    if status == "translated" and entry.get("source_sha") != entry.get("upstream_sha"):
        status = "stale"

    ref = state.get("upstream", {}).get("ref", "latest")
    banner = _banner(
        status,
        _upstream_page_url(config, page),
        _upstream_source_url(config, page.file.src_uri),
        ref,
    )

    # H1 がページ先頭にある場合はその直後に、なければ冒頭に挿入する。
    lines = markdown.split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if line.startswith("# "):
            return "\n".join(lines[: i + 1] + ["", banner] + lines[i + 1 :])
        break
    return f"{banner}\n{markdown}"
