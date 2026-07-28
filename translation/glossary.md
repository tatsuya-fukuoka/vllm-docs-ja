# 用語集（訳語統一のためのルール）

翻訳時はこの表に従ってください。表にない専門用語は、初出時に「日本語訳（English）」の
形で併記して構いません。判断に迷う場合は原語のままにし、Pull Request で相談してください。

## 基本方針

- 文体は敬体（です・ます調）。
- コード、コマンド、フラグ、環境変数、API 名、ファイルパス、YAML/JSON は翻訳しない。
- 見出しは訳すが、**原文の見出しから生成されるアンカーを明示的に付ける**。
  例: `## Offline Batched Inference` → `## オフラインバッチ推論 { #offline-batched-inference }`
  （他ページや外部サイトからのリンクを壊さないため。既に `{ #id }` が付いている場合は変更しない）
- 英数字と日本語の間に半角スペースを入れる（例: `vLLM の推論`）。
- 括弧は全角「（）」、句読点は「、」「。」を使う。

## 訳語

| 原語 | 訳語 | 備考 |
| --- | --- | --- |
| inference | 推論 | |
| serving | サービング | 「配信」とは訳さない |
| online serving | オンラインサービング | |
| offline inference | オフライン推論 | |
| throughput | スループット | |
| latency | レイテンシ | 「遅延」も可だが統一はレイテンシ |
| batch / batching | バッチ / バッチ処理 | |
| continuous batching | 連続バッチング | |
| scheduler | スケジューラ | |
| engine | エンジン | |
| worker | ワーカー | |
| request | リクエスト | |
| prompt | プロンプト | |
| token | トークン | |
| embedding | 埋め込み | 文脈により「エンベディング」 |
| attention | Attention | 原語のまま |
| KV cache | KV キャッシュ | |
| prefix caching | プレフィックスキャッシュ | |
| paged attention | PagedAttention | 固有名詞として原語のまま |
| quantization | 量子化 | |
| speculative decoding | 投機的デコーディング | |
| draft model | ドラフトモデル | |
| structured output | 構造化出力 | |
| guided decoding | ガイド付きデコーディング | |
| tool calling | ツール呼び出し | |
| reasoning | 推論（思考） | LLM の思考過程。inference と紛らわしいので文脈で補う |
| multimodal | マルチモーダル | |
| tensor parallelism | テンソル並列 | |
| pipeline parallelism | パイプライン並列 | |
| data parallelism | データ並列 | |
| expert parallelism | エキスパート並列 | |
| distributed serving | 分散サービング | |
| disaggregated prefill | Prefill 分離 | 原語を併記する |
| prefill / decode | Prefill / Decode | 原語のまま |
| chunked prefill | チャンク化 Prefill | |
| eager mode | eager モード | |
| CUDA graph | CUDA グラフ | |
| compilation | コンパイル | |
| backend | バックエンド | |
| deployment | デプロイ | |
| container | コンテナ | |
| cluster | クラスタ | |
| node | ノード | |
| GPU memory utilization | GPU メモリ使用率 | 設定値の説明では引数名を併記 |
| supported models | 対応モデル | |
| generative model | 生成モデル | |
| pooling model | プーリングモデル | |
| adapter | アダプタ | |
| fine-tuning | ファインチューニング | |
| checkpoint | チェックポイント | |
| troubleshooting | トラブルシューティング | |
| benchmark | ベンチマーク | |
| profiling | プロファイリング | |
| plugin | プラグイン | |
| contribution | コントリビューション | |
| upstream | 上流 | |
| release | リリース | |
| deprecated | 非推奨 | |
| experimental | 実験的機能 | |
| known issue | 既知の問題 | |
| best practice | ベストプラクティス | |
| out of memory (OOM) | メモリ不足（OOM） | |
| trade-off | トレードオフ | |
| overhead | オーバーヘッド | |
| warm-up | ウォームアップ | |

## admonition の訳

| 原文 | 訳語 |
| --- | --- |
| `!!! note` | メモ |
| `!!! tip` | ヒント |
| `!!! warning` | 注意 |
| `!!! danger` | 警告 |
| `!!! important` | 重要 |
| `!!! info` | 情報 |
| `!!! example` | 例 |

`!!! note` のようなキーワード自体は英語のまま残し、その後ろのタイトル文字列がある場合だけ
訳してください（例: `!!! note "Supported models"` → `!!! note "対応モデル"`）。
