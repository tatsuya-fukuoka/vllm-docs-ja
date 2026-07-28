# vLLM ドキュメント日本語版

[vLLM](https://github.com/vllm-project/vllm) 公式ドキュメントの**非公式**日本語訳です。

- 公開サイト: <https://tatsuya-fukuoka.github.io/vllm-docs-ja/>
- 追随している上流バージョン: **v0.26.0**（`translation/state.json` が正）
- 原文: <https://docs.vllm.ai/en/v0.26.0/>

> [!NOTE]
> 翻訳の正確性は保証されません。内容に疑義がある場合は英語原文を参照してください。
> 本リポジトリは vLLM プロジェクトの公式なものではありません。

## 仕組み

上流のドキュメントは MkDocs Material で書かれており、API リファレンス・CLI の引数一覧・
サンプル集は vLLM 本体を import してビルド時に生成されます。本リポジトリはそれらの
生成系を除いた「散文ページだけ」を取り込み、独立した MkDocs サイトとして構築します。

```
docs/          公開されるページ（日本語訳、未翻訳ページは英語原文のまま）
upstream/docs/ 追随中のタグにおける英語原文のスナップショット（差分検知用）
upstream/src/  --8<-- で参照される上流ソース（vllm/**/*.py, README.md）
hooks/         翻訳ステータスのバナー挿入、リンク書き換え
scripts/       上流同期・翻訳・状況集計のスクリプト
translation/   翻訳状態 (state.json) と用語集 (glossary.md)
```

未翻訳のページも英語のままサイトに含めているため、ページ間のリンクは常に解決します。
各ページの冒頭には翻訳ステータス（翻訳済み／未翻訳／要更新）と原文へのリンクが
自動で挿入されます。

## ローカルでのビルド

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-docs.txt
mkdocs serve          # http://127.0.0.1:8000
mkdocs build --strict # 本番と同じビルド
```

## 上流への追随

```bash
# 現在の追随先で差分だけ確認する
python scripts/sync_upstream.py --check-only

# 新しい安定版タグへ更新する（mkdocs.yml の extra.upstream_ref も更新すること）
python scripts/sync_upstream.py --ref v0.27.0

# 翻訳カバレッジのページを更新する
python scripts/check_status.py
```

`sync_upstream.py` は翻訳済みページを上書きしません。原文が更新された場合は
`translation/state.json` の `upstream_sha` が変わり、当該ページは「要更新」として
サイト上に警告が表示されます。

## 翻訳を追加する

```bash
# 未翻訳ページの下訳を Claude API で作る（要 ANTHROPIC_API_KEY）
pip install -r requirements-tools.txt
export ANTHROPIC_API_KEY=...
python scripts/translate.py --only 'features/*' --limit 3
```

下訳は**必ず人手でレビュー**してからコミットしてください。詳細は
[CONTRIBUTING.md](./CONTRIBUTING.md) と [用語集](./translation/glossary.md) を参照。

## ライセンス

原著作物と同じ Apache License 2.0 です。詳細は [LICENSE](./LICENSE) と
[NOTICE](./NOTICE) を参照してください。
