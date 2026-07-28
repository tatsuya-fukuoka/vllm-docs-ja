#!/usr/bin/env python3
"""上流 vLLM のドキュメントを取り込み、日本語版サイトに反映する。

やること:

1. `vllm-project/vllm` を指定タグ (既定: mkdocs.yml の `extra.upstream_ref`) で
   浅くクローンし、`docs/` を取得する
2. 原文スナップショットを `upstream/docs/` に保存する（差分検知の基準）
3. `--8<--` で参照される上流ソース (`vllm/**/*.py`, `README.md`) を
   `upstream/src/` に取得する
4. 未翻訳ページには変換後の英語原文を `docs/` に配置する
   （翻訳済みページは上書きしない）
5. 自動生成される `docs/generated/**.inc.md` のスタブを作る
6. `translation/state.json` を更新する

使い方:

    python scripts/sync_upstream.py                 # 既定のタグで同期
    python scripts/sync_upstream.py --ref v0.27.0   # 追随先のタグを変更
    python scripts/sync_upstream.py --check-only    # 変更せず差分だけ報告
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
CACHE_DIR = ROOT_DIR / ".cache" / "vllm-upstream"
UPSTREAM_DOCS = ROOT_DIR / "upstream" / "docs"
UPSTREAM_SRC = ROOT_DIR / "upstream" / "src"
DOCS_DIR = ROOT_DIR / "docs"
STATE_FILE = ROOT_DIR / "translation" / "state.json"
MKDOCS_FILE = ROOT_DIR / "mkdocs.yml"

REPO_URL = "https://github.com/vllm-project/vllm.git"
RAW_BASE = "https://raw.githubusercontent.com/vllm-project/vllm"

# 日本語版に取り込まないディレクトリ（vLLM 本体のインストールが必要な生成物）
EXCLUDE_DIRS = ("api/", "examples/", "mkdocs/")
EXCLUDE_SUFFIXES = (".template.md",)

# 上流からそのまま持ち込むアセット
VERBATIM_ASSETS = {
    "docs/mkdocs/stylesheets/extra.css": "docs/stylesheets/extra.css",
    "docs/mkdocs/javascript/mathjax.js": "docs/javascript/mathjax.js",
    "docs/assets/logos/vllm-logo-only-light.ico": (
        "docs/assets/logos/vllm-logo-only-light.ico"
    ),
    "docs/assets/logos/vllm-logo-text-light.png": (
        "docs/assets/logos/vllm-logo-text-light.png"
    ),
    "docs/assets/logos/vllm-logo-text-dark.png": (
        "docs/assets/logos/vllm-logo-text-dark.png"
    ),
}

SNIPPET_RE = re.compile(r'--8<--\s+"(?P<target>[^"]+)"')
SNIPPET_LINE_RE = re.compile(
    r'^(?P<indent>[ \t]*)--8<--\s+"(?P<target>[^"]+)"[ \t]*$', re.MULTILINE
)
MARKER_RE = re.compile(r"--8<--\s*\[start:(?P<name>[^\]]+)\]")
# 相対パスで書かれた docs/assets/... への参照
ASSET_RE = re.compile(r'(?<=[("\'])((?:\.{1,2}/)+assets/[^)"\'\s]+)')


def run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout


def ensure_clone(ref: str) -> Path:
    """指定タグの上流リポジトリ (docs/ のみ) を用意する。"""
    if not (CACHE_DIR / ".git").exists():
        CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
        print(f"上流をクローンしています: {REPO_URL}")
        subprocess.run(
            [
                "git", "clone", "--filter=blob:none", "--sparse", "--no-checkout",
                REPO_URL, str(CACHE_DIR),
            ],
            check=True,
        )
        run_git(["sparse-checkout", "set", "docs"], CACHE_DIR)

    print(f"上流の {ref} を取得しています")
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", f"refs/tags/{ref}:refs/tags/{ref}"],
        cwd=CACHE_DIR,
        check=False,
        capture_output=True,
    )
    run_git(["checkout", "--force", ref], CACHE_DIR)
    return CACHE_DIR


def upstream_commit(repo: Path) -> str:
    return run_git(["rev-parse", "HEAD"], repo).strip()


def read_upstream_file(repo: Path, ref: str, path: str) -> bytes | None:
    """sparse-checkout の外にあるファイルも含めて上流の内容を取得する。"""
    full = repo / path
    if full.exists():
        return full.read_bytes()
    try:
        result = subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def in_scope(rel_path: str) -> bool:
    """日本語版サイトに取り込む Markdown かどうか。"""
    if not rel_path.endswith(".md"):
        return False
    if rel_path.startswith(EXCLUDE_DIRS):
        return False
    if rel_path.endswith(EXCLUDE_SUFFIXES):
        return False
    return True


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def rewrite_assets(text: str, src_uri: str, ref: str) -> str:
    """docs/assets への相対参照を上流の raw URL に置き換える（画像を同梱しないため）。"""
    page_dir = Path(src_uri).parent

    def repl(match: re.Match) -> str:
        rel = match.group(1)
        resolved = (page_dir / rel).as_posix()
        # 正規化 (a/b/../c -> a/c)
        parts: list[str] = []
        for part in resolved.split("/"):
            if part in ("", "."):
                continue
            if part == "..":
                if parts:
                    parts.pop()
                continue
            parts.append(part)
        normalized = "/".join(parts)
        if not normalized.startswith("assets/"):
            return match.group(0)
        return f"{RAW_BASE}/{ref}/docs/{normalized}"

    return ASSET_RE.sub(repl, text)


EXCLUDED_LINK_RE = re.compile(
    r"\]\((?!https?://|#|mailto:)(?P<path>[^)\s]+?\.md)(?P<fragment>#[^)\s]*)?\)"
)


def _normalize(path: str) -> str:
    parts: list[str] = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def rewrite_excluded_links(text: str, src_uri: str, ref: str) -> str:
    """日本語版に取り込まないページ (api/, examples/) へのリンクを英語版に向ける。"""
    page_dir = Path(src_uri).parent

    def repl(match: re.Match) -> str:
        target = _normalize((page_dir / match.group("path")).as_posix())
        if not target.startswith(EXCLUDE_DIRS):
            return match.group(0)
        slug = target.removesuffix(".md")
        slug = slug.removesuffix("/README") if slug.endswith("/README") else slug
        if slug == "README":
            slug = ""
        fragment = match.group("fragment") or ""
        url = f"https://docs.vllm.ai/en/{ref}/{slug}/{fragment}".replace("//#", "/#")
        return f"]({url})"

    return EXCLUDED_LINK_RE.sub(repl, text)


API_REF_RE = re.compile(
    r"\[(?P<text>[^\[\]]+)\]\[(?P<ident>vllm\.[A-Za-z0-9_.]+)\]"
)


def rewrite_api_refs(text: str, ref: str) -> str:
    """mkdocstrings のクロスリファレンスを英語版 API リファレンスへのリンクにする。

    上流は `[LLM][vllm.LLM]` のような記法で API リファレンスを参照しているが、
    日本語版は mkdocstrings を使わないため、そのままではリンクにならない。
    api-autonav が生成する英語版のページ URL に変換する。
    """

    def repl(match: re.Match) -> str:
        label = match.group("text")
        ident = match.group("ident")
        parts = ident.split(".")
        # 先頭が大文字の要素（クラス）より前がモジュールパス。
        # すべて小文字なら最後の要素を関数とみなす。
        upper = next((i for i, p in enumerate(parts) if p[:1].isupper()), None)
        module_parts = parts[:upper] if upper else parts[:-1]
        if not module_parts:
            module_parts = ["vllm"]
        module_path = "/".join(module_parts)
        url = f"https://docs.vllm.ai/en/{ref}/api/{module_path}/#{ident}"
        if not label.startswith("`"):
            label = f"`{label}`"
        return f"[{label}]({url})"

    return API_REF_RE.sub(repl, text)


def build_snippet_index() -> dict[str, set[str]]:
    """`upstream/src/` にあるファイルとそこで定義されるスニペット区間の一覧。"""
    index: dict[str, set[str]] = {}
    for path in UPSTREAM_SRC.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(UPSTREAM_SRC).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        index[rel] = {m.group("name") for m in MARKER_RE.finditer(text)}
    return index


def fix_missing_snippets(
    text: str, ref: str, index: dict[str, set[str]]
) -> tuple[str, list[str]]:
    """解決できない `--8<--` を上流ソースへのリンクに置き換える。

    上流のタグによっては、ドキュメントが参照しているスニペット区間が
    ソース側に存在しないことがある（上流の docs は main でビルドされるため）。
    そのままではビルドが失敗するので、該当行を参照リンクに退避させる。
    """
    missing: list[str] = []

    def repl(match: re.Match) -> str:
        indent = match.group("indent")
        target = match.group("target")
        path, _, section = target.partition(":")
        if path.startswith("docs/"):
            return match.group(0)
        sections = index.get(path)
        if sections is not None and (not section or section in sections):
            return match.group(0)
        missing.append(target)
        url = f"https://github.com/vllm-project/vllm/blob/{ref}/{path}"
        return f"{indent}# このコードは上流のソースを参照してください: {url}"

    return SNIPPET_LINE_RE.sub(repl, text), missing


def transform(
    text: str, src_uri: str, ref: str, index: dict[str, set[str]]
) -> tuple[str, list[str]]:
    """未翻訳ページとして docs/ に置くための変換。"""
    text = rewrite_assets(text, src_uri, ref)
    text = rewrite_excluded_links(text, src_uri, ref)
    text = rewrite_api_refs(text, ref)
    return fix_missing_snippets(text, ref, index)


def patch_root_nav(text: str, ref: str) -> str:
    """API リファレンスと Examples を英語版への外部リンクに差し替える。"""
    docs_base = f"https://docs.vllm.ai/en/{ref}"
    text = text.replace(
        "      - Examples: examples\n",
        f"      - サンプル集 (英語): {docs_base}/examples/\n",
    )
    text = re.sub(
        r"  - API Reference:\n    - api/README\.md\n    - api/vllm\n",
        f"  - API リファレンス (英語): {docs_base}/api/\n",
        text,
    )
    return text


def collect_snippet_targets(docs_root: Path) -> set[str]:
    targets: set[str] = set()
    for md in docs_root.rglob("*.md"):
        for match in SNIPPET_RE.finditer(md.read_text(encoding="utf-8")):
            target = match.group("target").split(":", 1)[0]
            targets.add(target)
    return targets


def generated_stub(target: str, including_pages: list[str], ref: str) -> str:
    docs_base = f"https://docs.vllm.ai/en/{ref}"
    if including_pages:
        page = including_pages[0]
        url = f"{docs_base}/{page.removesuffix('.md').removesuffix('/README')}/"
    else:
        url = f"{docs_base}/"
    return (
        '!!! info "自動生成される内容"\n'
        "    この一覧は vLLM 本体のソースから自動生成されるため、日本語版サイトには\n"
        "    含まれていません。最新の内容は\n"
        f"    [英語版の該当ページ]({url})を参照してください。\n"
    )


def fetch_verbatim_assets(repo: Path, ref: str) -> None:
    """ロゴやスタイルシートなど、上流からそのまま持ち込むファイルを取得する。"""
    for src_rel, dst_rel in VERBATIM_ASSETS.items():
        data = read_upstream_file(repo, ref, src_rel)
        if data is None:
            print(f"  警告: {src_rel} を取得できませんでした", file=sys.stderr)
            continue
        dst = ROOT_DIR / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)


def fetch_snippet_sources(repo: Path, ref: str, upstream_docs: Path) -> None:
    """`--8<--` が参照する上流ソース (vllm/**/*.py, README.md) を取得する。"""
    targets = collect_snippet_targets(upstream_docs)
    for target in sorted(t for t in targets if not t.startswith("docs/")):
        data = read_upstream_file(repo, ref, target)
        if data is None:
            print(f"  警告: スニペット {target} を取得できませんでした", file=sys.stderr)
            continue
        dst = UPSTREAM_SRC / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)


def write_generated_stubs(upstream_docs: Path, ref: str) -> None:
    """vLLM 本体から自動生成される `docs/generated/**.inc.md` のスタブを作る。"""
    including: dict[str, list[str]] = {}
    for md in upstream_docs.rglob("*.md"):
        rel = md.relative_to(upstream_docs).as_posix()
        for match in SNIPPET_RE.finditer(md.read_text(encoding="utf-8")):
            target = match.group("target").split(":", 1)[0]
            if target.startswith("docs/generated/"):
                including.setdefault(target, []).append(rel)
    for target, pages_using in sorted(including.items()):
        dst = ROOT_DIR / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(
            generated_stub(target, sorted(pages_using), ref), encoding="utf-8"
        )


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"upstream": {}, "nav_files": {}, "pages": {}}


def default_ref() -> str:
    match = re.search(r"upstream_ref:\s*(\S+)", MKDOCS_FILE.read_text(encoding="utf-8"))
    return match.group(1) if match else "v0.26.0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default=None, help="追随する上流のタグ (既定: mkdocs.yml)")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="ファイルを変更せず、差分の要約だけを表示する",
    )
    parser.add_argument("--report", default=None, help="Markdown 形式の要約を書き出す先")
    args = parser.parse_args()

    ref = args.ref or default_ref()
    repo = ensure_clone(ref)
    commit = upstream_commit(repo)
    upstream_docs = repo / "docs"

    state = load_state()
    pages = state.setdefault("pages", {})
    nav_files = state.setdefault("nav_files", {})

    added: list[str] = []
    updated: list[str] = []
    now_stale: list[str] = []
    removed: list[str] = []
    missing_snippets: set[str] = set()

    if not args.check_only:
        # ページの変換より先に、スニペットが参照する上流ソースを用意する
        fetch_verbatim_assets(repo, ref)
        fetch_snippet_sources(repo, ref, upstream_docs)
        write_generated_stubs(upstream_docs, ref)
    snippet_index = build_snippet_index()

    seen: set[str] = set()
    for path in sorted(upstream_docs.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(upstream_docs).as_posix()
        if not in_scope(rel):
            continue
        seen.add(rel)

        source_text = path.read_text(encoding="utf-8")
        upstream_sha = sha256(source_text)
        entry = pages.get(rel)

        if entry is None:
            added.append(rel)
            entry = {
                "status": "untranslated",
                "source_sha": None,
                "upstream_sha": upstream_sha,
            }
            pages[rel] = entry
        elif entry.get("upstream_sha") != upstream_sha:
            updated.append(rel)
            entry["upstream_sha"] = upstream_sha
            if entry.get("status") == "translated":
                now_stale.append(rel)

        if args.check_only:
            continue

        # 原文スナップショットは常に最新に保つ
        snapshot = UPSTREAM_DOCS / rel
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(source_text, encoding="utf-8")

        # 未翻訳ページは変換した英語原文を配置する（翻訳済みは触らない）
        if entry["status"] != "translated":
            target = DOCS_DIR / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            content, missing = transform(source_text, rel, ref, snippet_index)
            missing_snippets.update(missing)
            target.write_text(content, encoding="utf-8")

    for rel in sorted(set(pages) - seen):
        removed.append(rel)

    if not args.check_only:
        # ナビゲーション定義
        for nav_rel in (
            ".nav.yml",
            "cli/.nav.yml",
            "getting_started/installation/.nav.yml",
        ):
            src = upstream_docs / nav_rel
            if not src.exists():
                continue
            text = src.read_text(encoding="utf-8")
            (UPSTREAM_DOCS / nav_rel).parent.mkdir(parents=True, exist_ok=True)
            (UPSTREAM_DOCS / nav_rel).write_text(text, encoding="utf-8")
            nav_files[nav_rel] = sha256(text)
            target = DOCS_DIR / nav_rel
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                patched = patch_root_nav(text, ref) if nav_rel == ".nav.yml" else text
                target.write_text(patched, encoding="utf-8")

        state["upstream"] = {
            "repo": "vllm-project/vllm",
            "ref": ref,
            "commit": commit,
            "synced_at": date.today().isoformat(),
        }
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    lines = [
        f"# 上流同期レポート ({ref})",
        "",
        f"- 上流コミット: `{commit}`",
        f"- 対象ページ数: {len(seen)}",
        f"- 新規ページ: {len(added)}",
        f"- 原文が更新されたページ: {len(updated)}",
        f"- 翻訳が陳腐化したページ: {len(now_stale)}",
        f"- 上流から消えたページ: {len(removed)}",
        f"- 解決できず退避したスニペット: {len(missing_snippets)}",
    ]
    for title, items in (
        ("新規ページ", added),
        ("翻訳が陳腐化したページ", now_stale),
        ("上流から消えたページ", removed),
        ("解決できず退避したスニペット", sorted(missing_snippets)),
    ):
        if items:
            lines += ["", f"## {title}", ""]
            lines += [f"- `{item}`" for item in items]
    report = "\n".join(lines) + "\n"

    print(report)
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")

    if args.check_only and (added or updated or removed):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
