# 翻訳に参加する

## 翻訳の流れ

1. [翻訳状況](https://tatsuya-fukuoka.github.io/vllm-docs-ja/ja/status/) で未翻訳・
   要更新のページを確認する
2. ブランチを切る（例: `translate/features-lora`）
3. 原文 `upstream/docs/<path>` を見ながら `docs/<path>` を日本語に書き換える
   - Claude API で下訳する場合: `python scripts/translate.py --only '<glob>'`
4. `translation/state.json` の該当エントリを更新する

   ```json
   "features/lora.md": {
     "status": "translated",
     "source_sha": "<upstream_sha と同じ値を入れる>",
     "upstream_sha": "<変更しない>",
     "translated_at": "2026-07-28"
   }
   ```

   `source_sha` を `upstream_sha` に合わせることで「最新の原文に基づく翻訳」として
   扱われます。
5. `python scripts/check_status.py` で翻訳状況ページを更新する
6. `mkdocs build --strict` が通ることを確認して Pull Request を出す

## 翻訳のルール

[translation/glossary.md](./translation/glossary.md) に用語集と文体のルールがあります。
特に重要な点:

- 文体は敬体（です・ます調）
- コード・コマンド・フラグ・環境変数・API 名・パスは翻訳しない
- Markdown の構造（見出しレベル、表、admonition、タブ）を原文どおり保つ
- リンクの URL は変更しない（相対リンクもそのまま）
- `--8<--` のスニペット指示行は絶対に変更しない
- 英数字と日本語の間には半角スペースを入れる

## やってはいけないこと

- `upstream/` 配下の編集（同期スクリプトが上書きします）
- `docs/generated/` 配下の編集（自動生成のスタブです）
- 原文にない情報の追加（補足したい場合は訳注として `!!! note "訳注"` を使う）

## 上流のバージョンを上げる

1. `mkdocs.yml` の `extra.upstream_ref` と `extra.upstream_docs_url` を更新
2. `python scripts/sync_upstream.py --ref <新しいタグ>`
3. 「要更新」になったページを訳し直す
4. `python scripts/check_status.py`
