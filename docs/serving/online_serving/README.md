# オンラインサービング { #online-serving }

vLLM は、多くのインターフェイスと互換性のある HTTP サーバーを提供します。

## OpenAI 互換サーバー { #openai-compatible-server }

現在サポートしている OpenAI API は次のとおりです。

- [Completions API](./openai_compatible_server.md#completions-api) (`/v1/completions`)
    - [テキスト生成モデル](../../models/generative_models.md)にのみ適用できます。
    - *注意: `suffix` パラメータはサポートされていません。*
- [Chat Completions API](./openai_compatible_server.md#chat-api) (`/v1/chat/completions`)
    - [チャットテンプレート](./openai_compatible_server.md#chat-template)を持つ[テキスト生成モデル](../../models/generative_models.md)にのみ適用できます。
    - *注意: `user` パラメータは無視されます。*
    - *注意:* `parallel_tool_calls` パラメータを `false` にすると、vLLM は 1 リクエストにつきツール呼び出しを 0 個または 1 個しか返しません。`true`（既定値）にすると 1 リクエストで複数のツール呼び出しを返せます。ただし `true` にしても複数返ることは保証されません。この挙動はモデルに依存し、すべてのモデルが並列のツール呼び出しに対応しているわけではないためです。
- [Chat Completions batch API](./openai_compatible_server.md#chat-api) (`/v1/chat/completions/batch`)
- [Responses API](./openai_compatible_server.md#responses-api) (`/v1/responses`, `/v1/responses/{response_id}`, `/v1/responses/{response_id}/cancel`)
    - [テキスト生成モデル](../../models/generative_models.md)にのみ適用できます。
- [Embeddings API](../../models/pooling_models/embed.md#openai-compatible-embeddings-api) (`/v1/embeddings`)
    - [埋め込みモデル](../../models/pooling_models/embed.md)にのみ適用できます。
- [Transcriptions API](./speech_to_text.md#transcriptions-api) (`/v1/audio/transcriptions`)
    - [自動音声認識 (ASR) モデル](../../models/supported_models.md#transcription)にのみ適用できます。
- [Translation API](./speech_to_text.md#translations-api) (`/v1/audio/translations`)
    - [自動音声認識 (ASR) モデル](../../models/supported_models.md#transcription)にのみ適用できます。

## Anthropic API { #anthropic-apis }

- Anthropic messages API (`/v1/messages`, `/v1/messages/count_tokens`)

## Cohere API { #cohere-apis }

- [Cohere Embed API](../../models/pooling_models/embed.md#cohere-embed-api) (`/v2/embed`)
    - [Cohere の Embed API](https://docs.cohere.com/reference/embed) と互換性があります
    - マルチモーダルモデルを含む任意の[埋め込みモデル](../../models/pooling_models/embed.md#supported-models)で利用できます。
- [Cohere Rerank API](../../models/pooling_models/scoring.md#rerank-api) (`/rerank`, `/v1/rerank`, `/v2/rerank`)
    - [Jina AI の v1 rerank API](https://jina.ai/reranker/) を実装しています
    - [Cohere の v1 / v2 rerank API](https://docs.cohere.com/v2/reference/rerank) と互換性があります

## プーリング API { #pooling-apis }

プーリングモデルの詳細は[このページ](../../models/pooling_models/README.md)を参照してください。

- [分類の使い方](../../models/pooling_models/classify.md)
    - [Classification API](../../models/pooling_models/classify.md#online-serving) (`/classify`)
    - [分類モデル](../../models/pooling_models/classify.md)にのみ適用できます。
- [埋め込みの使い方](../../models/pooling_models/embed.md)
    - [Cohere Embed API](../../models/pooling_models/embed.md#cohere-embed-api) (`/v2/embed`)
    - [OpenAI-compatible Embeddings API](../../models/pooling_models/embed.md#openai-compatible-embeddings-api) (`/v1/embeddings`)
    - [埋め込みモデル](../../models/pooling_models/embed.md)にのみ適用できます。
- [スコアリングの使い方](../../models/pooling_models/scoring.md)
    - [Score API](../../models/pooling_models/scoring.md#score-api) (`/score`, `/v1/score`)
    - [Cohere Rerank API](../../models/pooling_models/scoring.md#rerank-api) (`/rerank`, `/v1/rerank`, `/v2/rerank`)
    - [スコアモデル](../../models/pooling_models/scoring.md)（cross-encoder、bi-encoder、late-interaction）に適用できます。
- [Pooling API](../../models/pooling_models/README.md#pooling-api) (`/pooling`)
    - すべての[プーリングモデル](../../models/pooling_models/README.md)に適用できます。

## 音声認識 API { #speech-to-text-apis }

音声認識の詳細は[このページ](speech_to_text.md)を参照してください。

- [Transcriptions API](./speech_to_text.md#transcriptions-api) (`/v1/audio/transcriptions`)
    - [自動音声認識 (ASR) モデル](../../models/supported_models.md#transcription)にのみ適用できます。
- [Translation API](./speech_to_text.md#translations-api) (`/v1/audio/translations`)
    - [自動音声認識 (ASR) モデル](../../models/supported_models.md#transcription)にのみ適用できます。
- [Realtime API](./speech_to_text.md#realtime-api) (`/v1/realtime`)
    - [自動音声認識 (ASR) モデル](../../models/supported_models.md#realtime-transcription)にのみ適用できます。

## 独自 API { #custom-apis }

- [Classification API](../../models/pooling_models/classify.md#classification-api) (`/classify`)
    - [分類モデル](../../models/pooling_models/classify.md)にのみ適用できます。
- [Score API](../../models/pooling_models/scoring.md#score-api) (`/score`, `/v1/score`)
    - [スコアモデル](../../models/pooling_models/scoring.md)（cross-encoder、bi-encoder、late-interaction）に適用できます。
- [Pooling API](../../models/pooling_models/README.md#pooling-api) (`/pooling`)
    - すべての[プーリングモデル](../../models/pooling_models/README.md)に適用できます。
- [Generative Scoring API](generative_scoring.md#generative-scoring-api) (`/generative_scoring`)
    - [CausalLM モデル](../../models/generative_models.md)（タスク `"generate"`）に適用できます。
    - 指定された `label_token_ids` に対する次トークンの確率を計算します。

## 計測用 API { #instrumentator-apis }

### 基本 API { #basic-apis }

- `/version` - バージョン情報
- `/load` - サーバー負荷のメトリクス
- `/v1/models` - 利用可能なモデルの一覧
- `/health` - ヘルスチェック

### メトリクス API { #metrics-apis }

メトリクスの詳細は[このページ](../../design/metrics.md)を参照してください。

- `/metrics` - Prometheus 互換のメトリクス HTTP エンドポイント

### オフラインでの API ドキュメント { #offline-api-documentation }

FastAPI の `/docs` エンドポイントは既定でインターネット接続を必要とします。ネットワークから隔離された環境でオフラインでも利用できるようにするには、`--enable-offline-docs` フラグを指定します。

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --enable-offline-docs
```

### LoRA の動的ロード { #lora-dynamic-loading }

API サーバーでは LoRA の動的なロード・アンロードが有効になっています。ローカル開発でのみ使用してください。

- `/v1/load_lora_adapter` - LoRA の動的ロード
- `/v1/unload_lora_adapter` - LoRA の動的アンロード

### プロファイリング API { #profiling-apis }

vLLM のプロファイリングの詳細は[このページ](../../contributing/profiling.md)を参照してください。

- `/start_profile` - PyTorch プロファイラを開始
- `/stop_profile` - PyTorch プロファイラを停止

### SageMaker API { #sagemaker-apis }

- `/ping` - SageMaker のヘルスチェック
- `/invocations` - SageMaker 互換のエンドポイント（`/v1` エンドポイントと同じ推論処理へ振り分けられます）

## スケールアウト API { #scale-out-apis }

### Tokens IN <> Tokens OUT API { #tokens-in-tokens-out-apis }

- `/inference/v1/generate` - 補完を生成
- `/abort_requests` - 実行中のリクエストを中断（`--tokens-only` も指定した場合のみ）

### レンダラー API { #renderer-apis }

レンダラー API の詳細は[このページ](renderer.md)を参照してください。

- [Completions Render API](renderer.md) (`/v1/completions/render`)
    - completion リクエストをレンダリングします
- [Chat Completions Render API](renderer.md) (`/v1/chat/completions/render`)
    - chat completion をレンダリングします

### デレンダラー API { #derenderer-apis }

デレンダラー API の詳細は[このページ](derenderer.md)を参照してください。

- [Chat Completions Derender API](derenderer.md) (`/v1/chat/completions/derender`)
    - chat completion リクエストをデレンダリングします
- [Completions Derender API](derenderer.md) (`/v1/completions/derender`)
    - completion リクエストをデレンダリングします

## トークナイズ API { #tokenize-apis }

- `/tokenize` - テキストをトークナイズ
- `/detokenize` - トークンをデトークナイズ
- `/tokenizer_info` - チャットテンプレートや設定を含む、トークナイザーの詳細情報を取得

## Elastic Expert Parallelism (EEP) { #elastic-expert-parallelism-eep }

- `/scale_elastic_ep` - スケーリング操作を実行
- `/is_scaling_elastic_ep` - スケーリング中かどうかを確認

## 開発モードのサーバー { #server-in-development-mode }

VLLM_SERVER_DEV_MODE=1 を指定すると、開発用エンドポイントが有効になります。

**セキュリティ上の警告: これらのエンドポイントを本番環境で使用しないでください。**

### キャッシュ管理 API { #cache-management-apis }

- `/reset_prefix_cache` - プレフィックスキャッシュをリセット（サービスに影響する可能性があります）
- `/reset_mm_cache` - マルチモーダルキャッシュをリセット（サービスに影響する可能性があります）
- `/reset_encoder_cache` - エンコーダーキャッシュをリセット（サービスに影響する可能性があります）

### 重み転送 API（RL 学習） { #weight-transfer-apis-rl-training }

重み転送の詳細は[このページ](../../training/weight_transfer/README.md)を参照してください。

- `/pause` - 生成を一時停止（サービス停止状態になります）
- `/resume` - 生成を再開
- `/is_paused` - 生成が一時停止中かどうかを確認
- `/abort_requests` - スケジューラを停止せずに実行中のリクエスト（すべて、または指定した `request_ids`）を中断
- `/init_weight_transfer_engine` - RLHF 用の重み転送エンジンを初期化
- `/start_weight_update` - 重み更新に向けて推論エンジンを準備します。
- `/update_weights` - モデルの重みを更新（モデルの挙動が変わる可能性があります）
- `/finish_weight_update` - 重み更新を確定します
- `/get_world_size` - 分散実行の world size を取得

### Collective RPC { #collective-rpc }

- `/collective_rpc` - エンジン上で任意の RPC メソッドを実行（極めて危険です）

### サーバー情報 { #server-info }

- `/server_info` - サーバーの詳細な設定を取得

### スリープモード API { #sleep-mode-apis }

スリープモードの詳細は[このページ](../../features/sleep_mode.md)を参照してください。

- `/sleep` - エンジンをスリープ状態にする（サービス停止状態になります）
- `/wake_up` - スリープ状態のエンジンを復帰させる
- `/is_sleeping` - エンジンがスリープ中かどうかを確認

## チャットテンプレート { #chat-template }

言語モデルがチャットプロトコルをサポートするには、モデルのトークナイザー設定にチャットテンプレートが
含まれている必要があります。チャットテンプレートは Jinja2 テンプレートで、ロール・メッセージ・
その他チャット固有のトークンを入力にどうエンコードするかを定義します。

`NousResearch/Meta-Llama-3-8B-Instruct` のチャットテンプレートの例は[こちら](https://llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/#prompt-template-for-meta-llama-3)にあります。

instruction / chat 向けにファインチューニングされていても、チャットテンプレートが提供されていないモデルがあります。
そうしたモデルでは、`--chat-template` パラメータにチャットテンプレートのファイルパス、または文字列としての
テンプレートを指定できます。チャットテンプレートがないとサーバーはチャットを処理できず、
すべてのチャットリクエストがエラーになります。

```bash
vllm serve <model> --chat-template ./path-to-chat-template.jinja
```

vLLM コミュニティは主要なモデル向けのチャットテンプレートを提供しています。[examples](../../../examples) ディレクトリ以下にあります。

マルチモーダルのチャット API が加わったことで、OpenAI の仕様では `type` と `text` の両方を指定する新しい形式の
チャットメッセージも受け付けるようになりました。例を示します。

```python
completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Classify this sentiment: vLLM is wonderful!"},
            ],
        },
    ],
)
```

LLM 向けのチャットテンプレートの多くは `content` フィールドが文字列であることを想定していますが、
`meta-llama/Llama-Guard-3-1B` のような新しいモデルでは、リクエスト内の content が OpenAI のスキーマに従って
整形されていることを想定します。vLLM はこれをベストエフォートで自動判定し、
*"Detected the chat template content format to be..."* のようなログを出力したうえで、
受信したリクエストを判定した形式に合わせて内部的に変換します。形式は次のいずれかです。

- `"string"`: 文字列。
    - 例: `"Hello world"`
- `"openai"`: OpenAI のスキーマに似た辞書のリスト。
    - 例: `[{"type": "text", "text": "Hello world!"}]`

判定結果が期待どおりでない場合は、CLI 引数 `--chat-template-content-format` で使用する形式を上書きできます。

## Ray Serve LLM { #ray-serve-llm }

Ray Serve LLM を使うと、vLLM エンジンをスケーラブルかつ本番品質でサービングできます。vLLM と緊密に統合され、オートスケーリング・負荷分散・バックプレッシャーといった機能を追加します。

主な機能:

- OpenAI 互換の HTTP API と Python の API の両方を提供します。
- コードを変更せずに、単一 GPU からマルチノードのクラスタまでスケールします。
- Ray のダッシュボードとメトリクスを通じて、可観測性とオートスケーリングのポリシーを提供します。

次の例は、DeepSeek R1 のような大きなモデルを Ray Serve LLM でデプロイする方法を示しています: [examples/ray_serving/ray_serve_deepseek.py](../../../examples/ray_serving/ray_serve_deepseek.py)。

Ray Serve LLM の詳細は公式の [Ray Serve LLM ドキュメント](https://docs.ray.io/en/latest/serve/llm/index.html)（英語）を参照してください。
