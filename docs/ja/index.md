# この日本語版について

このサイトは [vLLM](https://github.com/vllm-project/vllm) 公式ドキュメントの
**非公式**な日本語訳です。vLLM プロジェクトが公開・保守しているものではありません。

- 原文: <https://docs.vllm.ai/en/v0.26.0/>
- 翻訳リポジトリ: <https://github.com/tatsuya-fukuoka/vllm-docs-ja>
- 追随している上流バージョン: **v0.26.0**

!!! warning "免責"
    翻訳の正確性・最新性は保証されません。設定値やコマンドの挙動など、正確さが
    求められる場面では必ず英語原文を参照してください。

## 翻訳の進め方

- 上流の安定版タグ（現在は `v0.26.0`）に固定して翻訳しています。
  上流の `main` を毎日追いかけることはしません。
- **未翻訳のページも英語のまま掲載**しています。ページ間のリンクを切らさず、
  どのページからでも目的の情報にたどり着けるようにするためです。
- 各ページの冒頭には、そのページが「翻訳済み」「未翻訳」「原文の更新に
  追随できていない」のどれであるかを示すバナーが表示されます。
- 現在の進捗は[翻訳状況](./status.md)を参照してください。

## 日本語版に含まれないもの

次のコンテンツは vLLM 本体をインストールしてビルド時に自動生成されるため、
日本語版サイトには含めず、英語版へのリンクにしています。

| コンテンツ | 英語版へのリンク |
| --- | --- |
| API リファレンス | <https://docs.vllm.ai/en/v0.26.0/api/> |
| サンプル集（Examples） | <https://docs.vllm.ai/en/v0.26.0/examples/> |
| CLI の引数一覧 | 各 CLI ページに英語版へのリンクを掲載 |
| メトリクス一覧 | <https://docs.vllm.ai/en/v0.26.0/usage/metrics/> |

## 翻訳への参加

誤訳の指摘、未翻訳ページの翻訳、用語の統一など、どんな規模でも歓迎します。

- [Issue を立てる](https://github.com/tatsuya-fukuoka/vllm-docs-ja/issues)
- [Pull Request を送る](https://github.com/tatsuya-fukuoka/vllm-docs-ja/pulls)
- 翻訳の進め方は
  [CONTRIBUTING.md](https://github.com/tatsuya-fukuoka/vllm-docs-ja/blob/main/CONTRIBUTING.md)、
  訳語は
  [用語集](https://github.com/tatsuya-fukuoka/vllm-docs-ja/blob/main/translation/glossary.md)
  を参照してください。

## ライセンス

原著作物である vLLM のドキュメントは Apache License 2.0 で公開されています。
本日本語訳も同じ Apache License 2.0 に従います。

```text
Copyright contributors to the vLLM project
Licensed under the Apache License, Version 2.0
```
