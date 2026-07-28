"""リンク書き換えのための MkDocs フック + Markdown 拡張。

上流 vLLM の `docs/mkdocs/hooks/url_schemes.py`
(https://github.com/vllm-project/vllm, Apache License 2.0) をもとに、
日本語版サイト（vLLM 本体のソースツリーを持たない）向けに改変したもの。

対応する内容:

- `docs/` の外を指す相対リンク（`../../examples/foo.py` など）を、
  追随中のタグに固定した GitHub の URL に変換する
- GitHub の Issue / Pull Request / Project へのリンクにアイコンと既定のタイトルを付ける
- `https://docs.vllm.ai/en/<ver>/...` 形式の絶対リンクを、日本語版サイト内の
  相対リンクに変換する（対応するページが日本語版に存在する場合のみ）

pymdownx.snippets (priority 32) で取り込まれた内容にも適用されるよう、
Markdown プリプロセッサ（priority 25）として動作する。
"""

import posixpath

import regex as re
from markdown import Extension
from markdown.preprocessors import Preprocessor
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page

gh_icon = ":octicons-mark-github-16:"

# 正規表現の部品
TITLE = r"(?P<title>[^\[\]<>]+?)"
REPO = r"(?P<repo>.+?/.+?)"
TYPE = r"(?P<type>issues|pull|projects)"
NUMBER = r"(?P<number>\d+)"
VERSION = r"[^/\s]+"
# 日本語の本文は単語間に空白がないため、`)` や `]` を明示的に除外しないと
# パスやフラグメントが後続の文をまるごと飲み込んでしまう（上流の英語前提の実装からの変更点）。
PATH = r"(?P<path>[^\s)]+?)"
FRAGMENT = r"(?P<fragment>#[^\s)\]]+)?"
URL_GITHUB = f"https://github.com/{REPO}/{TYPE}/{NUMBER}{FRAGMENT}"
RELATIVE = rf"(?!(https?|ftp)://|#|mailto:){PATH}{FRAGMENT}"
URL_DOCS = f"https://docs.vllm.ai/en/{VERSION}/{PATH}{FRAGMENT}"

TITLES = {"issues": "Issue ", "pull": "Pull Request ", "projects": "Project "}

github_link = re.compile(rf"(\[{TITLE}\]\(|<){URL_GITHUB}(\)|>)")
relative_link = re.compile(rf"\[{TITLE}\]\({RELATIVE}\)")
docs_link = re.compile(rf"\[{TITLE}\]\({URL_DOCS}\)")


class UrlSchemesPreprocessor(Preprocessor):
    def __init__(self, md, ext):
        super().__init__(md)
        self.ext = ext

    def run(self, lines):
        page = self.ext.page
        files = self.ext.files
        config = self.ext.mkdocs_config
        if page is None or config is None:
            return lines

        ref = config["extra"].get("upstream_ref", "main")

        def replace_relative_link(match: re.Match) -> str:
            """docs/ の外を指す相対リンクを上流 GitHub の URL に置き換える。"""
            title = match.group("title")
            path = match.group("path")
            fragment = match.group("fragment") or ""

            # docs/ をルートとみなして正規化する
            target = posixpath.normpath(
                posixpath.join(posixpath.dirname(page.file.src_uri), path)
            )
            if not target.startswith("../"):
                # docs/ の中を指しているのでそのまま
                return match.group(0)

            # docs/ の外 = 上流リポジトリのルートからの相対パス
            upstream_path = target.lstrip("./")
            while upstream_path.startswith("../"):
                upstream_path = upstream_path[3:]

            slug = "tree" if path.endswith("/") else "blob"
            url = (
                f"https://github.com/vllm-project/vllm/{slug}/{ref}/"
                f"{upstream_path}{fragment}"
            )
            return f"[{gh_icon} {title}]({url})"

        def replace_github_link(match: re.Match) -> str:
            """Issue / PR / Project へのリンクを読みやすい形に整える。"""
            repo = match.group("repo")
            type = match.group("type")
            number = match.group("number")
            title = match.group("title") or ""
            fragment = match.group("fragment") or ""

            if not title:
                title = TITLES[type]
                if "vllm-project" not in repo:
                    title += repo
                title += f"#{number}"

            url = f"https://github.com/{repo}/{type}/{number}{fragment}"
            return f"[{gh_icon} {title}]({url})"

        def replace_docs_link(match: re.Match) -> str:
            """docs.vllm.ai への絶対リンクを日本語版サイト内の相対リンクにする。"""
            title = match.group("title")
            path = match.group("path").rstrip("/")
            fragment = match.group("fragment") or ""

            src = f"{path.removesuffix('.html')}.md"
            if files is None or files.get_file_from_path(src) is None:
                # 日本語版に存在しないページ（API リファレンス等）は原文へのリンクのまま
                return match.group(0)
            rel = posixpath.relpath(src, posixpath.dirname(page.file.src_uri))
            if title.startswith("http"):
                title = path.removesuffix(".html")
            return f"[{title}]({rel}{fragment})"

        markdown = "\n".join(lines)
        markdown = github_link.sub(replace_github_link, markdown)
        markdown = relative_link.sub(replace_relative_link, markdown)
        markdown = docs_link.sub(replace_docs_link, markdown)
        return markdown.split("\n")


class UrlSchemesExtension(Extension):
    def __init__(self, **kwargs):
        self.page = None
        self.files = None
        # `config` は markdown.Extension が内部で使う属性名のため衝突を避ける
        self.mkdocs_config = None
        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        # pymdownx.snippets (priority 32) の後に実行する
        md.preprocessors.register(UrlSchemesPreprocessor(md, self), "url_schemes", 25)


_ext = UrlSchemesExtension()


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    config["markdown_extensions"].append(_ext)
    _ext.mkdocs_config = config
    return config


def on_page_markdown(
    markdown: str, *, page: Page, config: MkDocsConfig, files: Files
) -> str:
    _ext.page = page
    _ext.files = files
    _ext.mkdocs_config = config
    return markdown
