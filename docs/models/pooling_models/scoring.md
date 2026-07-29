# スコアリングの使い方 { #scoring-usages }

スコアモデルは、2 つの入力プロンプト間の類似度スコアを計算するためのものです。`cross-encoder`、`late-interaction`、`bi-encoder` の 3 つのモデル種別（`score_type`）をサポートします。

!!! note
    vLLM が扱うのは RAG パイプラインのうちモデル推論の部分（埋め込みの生成やリランキングなど）のみです。より上位の RAG オーケストレーションには、[LangChain](https://github.com/langchain-ai/langchain) のような統合フレームワークを活用してください。

## 概要 { #summary }

- モデルの用途: スコアリング
- プーリングタスク:

| スコア種別        | プーリングタスク         | スコアリング関数         |
|--------------------|-----------------------|--------------------------|
| `cross-encoder`    | `classify`（注記参照） | 線形分類器        |
| `late-interaction` | `token_embed`         | late interaction（MaxSim） |
| `bi-encoder`       | `embed`               | コサイン類似度        |

- オフライン API:
    - `LLM.score`
- オンライン API:
    - [Score API](scoring.md#score-api)（`/score`、`/v1/score`）
    - [Cohere Rerank API](scoring.md#rerank-api)（`/rerank`、`/v1/rerank`、`/v2/rerank`）

!!! note
    分類モデルは、num_labels が 1 の出力を持つ場合にのみスコアリングモデルとして使え、スコアリング API を有効にできます。

### スコア種別 { #score-types }

サポートされる 3 つのスコアリング関数を下図に示します。

![スコア種別](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/models/pooling_models/score_types.svg)

## サポートされるモデル { #supported-models }

### Cross-encoder モデル { #cross-encoder-models }

[Cross-encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html)（リランカーとも呼ばれます）モデルは、2 つのプロンプトを入力として受け取り、num_labels が 1 の出力を返す分類モデルの一種です。

--8<-- [start:supported-cross-encoder-models]

#### テキストのみのモデル { #text-only-models }

| アーキテクチャ | モデル | HF モデルの例 | スコアテンプレート（注記参照） | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | ------------------------- | --------------------------- | --------------------------------------- |
| `BertForSequenceClassification` | BERT 系 | `cross-encoder/ms-marco-MiniLM-L-6-v2` など | N/A | | |
| `GemmaForSequenceClassification` | Gemma 系 | `BAAI/bge-reranker-v2-gemma`（注記参照）など | [bge-reranker-v2-gemma.jinja](../../../examples/pooling/score/template/bge-reranker-v2-gemma.jinja) | ✅︎ | ✅︎ |
| `GteNewForSequenceClassification` | mGTE-TRM（注記参照） | `Alibaba-NLP/gte-multilingual-reranker-base` など | N/A | | |
| `LlamaBidirectionalForSequenceClassification`<sup>C</sup> | 双方向 attention を持つ Llama 系 | `nvidia/llama-nemotron-rerank-1b-v2` など | [nemotron-rerank.jinja](../../../examples/pooling/score/template/nemotron-rerank.jinja) | ✅︎ | ✅︎ |
| `ModernBertForSequenceClassification` | ModernBERT 系 | `Alibaba-NLP/gte-reranker-modernbert-base` など | N/A | | |
| `Qwen2ForSequenceClassification`<sup>C</sup> | Qwen2 系 | `mixedbread-ai/mxbai-rerank-base-v2`（注記参照）など | [mxbai_rerank_v2.jinja](../../../examples/pooling/score/template/mxbai_rerank_v2.jinja) | ✅︎ | ✅︎ |
| `Qwen3ForSequenceClassification`<sup>C</sup> | Qwen3 系 | `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`、`Qwen/Qwen3-Reranker-0.6B`（注記参照）など | [qwen3_reranker.jinja](../../../examples/pooling/score/template/qwen3_reranker.jinja) | ✅︎ | ✅︎ |
| `RobertaForSequenceClassification` | RoBERTa 系 | `cross-encoder/quora-roberta-base` など | N/A | | |
| `XLMRobertaForSequenceClassification` | XLM-RoBERTa 系 | `BAAI/bge-reranker-v2-m3` など | N/A | | |
| `*Model`<sup>C</sup>、`*ForCausalLM`<sup>C</sup> など | 生成モデル | N/A | N/A | \* | \* |

<sup>C</sup> `--convert classify` により自動的に分類モデルへ変換されます。（[詳細](./README.md#model-conversion)）
\* 機能のサポート状況は元のモデルと同じです。

!!! note
    モデルによっては、正しく動作させるために特定のプロンプト形式が必要です。

    HF モデル例に対応するスコアテンプレートは [examples/pooling/score/template/](../../../examples/pooling/score/template) にあります。

    例: [examples/pooling/score/using_template_offline.py](../../../examples/pooling/score/using_template_offline.py) [examples/pooling/score/using_template_online.py](../../../examples/pooling/score/using_template_online.py)

!!! note
    公式のオリジナル `BAAI/bge-reranker-v2-gemma` は、次のコマンドで読み込みます。

    ```bash
    vllm serve BAAI/bge-reranker-v2-gemma --hf_overrides '{"architectures": ["GemmaForSequenceClassification"],"classifier_from_token": ["Yes"],"method": "no_post_processing"}'
    ```

!!! note
    第 2 世代の GTE モデル（mGTE-TRM）は `NewForSequenceClassification` という名前です。`NewForSequenceClassification` という名称は汎用的すぎるため、`GteNewForSequenceClassification` アーキテクチャを使うことを明示するには `--hf-overrides '{"architectures": ["GteNewForSequenceClassification"]}'` を指定してください。

!!! note
    公式のオリジナル `mxbai-rerank-v2` は、次のコマンドで読み込みます。

    ```bash
    vllm serve mixedbread-ai/mxbai-rerank-base-v2 --hf_overrides '{"architectures": ["Qwen2ForSequenceClassification"],"classifier_from_token": ["0", "1"], "method": "from_2_way_softmax"}'
    ```

!!! note
    公式のオリジナル `Qwen3 Reranker` は、次のコマンドで読み込みます。詳細は [examples/pooling/score/qwen3_reranker_offline.py](../../../examples/pooling/score/qwen3_reranker_offline.py) [examples/pooling/score/qwen3_reranker_online.py](../../../examples/pooling/score/qwen3_reranker_online.py) を参照してください。

    ```bash
    vllm serve Qwen/Qwen3-Reranker-0.6B --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
    ```

#### マルチモーダルモデル { #multimodal-models }

!!! note
    マルチモーダルモデルの入力について詳しくは、[このページ](../supported_models.md#list-of-multimodal-language-models)を参照してください。

| アーキテクチャ | モデル | 入力 | HF モデルの例 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `JinaVLForSequenceClassification` | JinaVL 系 | T + I<sup>E+</sup> | `jinaai/jina-reranker-m0` など | ✅︎ | ✅︎ |
| `LlamaNemotronVLForSequenceClassification` | Llama Nemotron Reranker + SigLIP | T + I<sup>E+</sup> | `nvidia/llama-nemotron-rerank-vl-1b-v2` | | |
| `Qwen3VLForSequenceClassification` | Qwen3-VL-Reranker | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3-VL-Reranker-2B`（注記参照）など | ✅︎ | ✅︎ |

<sup>C</sup> `--convert classify` により自動的に分類モデルへ変換されます。（[詳細](README.md#model-conversion)）
\* 機能のサポート状況は元のモデルと同じです。

!!! note
    Qwen3-Reranker と同様に、公式のオリジナル `Qwen3-VL-Reranker` を読み込むには次の `--hf_overrides` を指定する必要があります。`Qwen3-VL` は公式には画像の前処理に `qwen_vl_utils` を使いますが、vLLM は `transformers` の `video_processing_qwen3_vl` を使うため、公式の Hugging Face リポジトリの例とはわずかに結果が異なります。

    ```bash
    vllm serve Qwen/Qwen3-VL-Reranker-2B --hf_overrides '{"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
    ```

--8<-- [end:supported-cross-encoder-models]

### Late-interaction モデル { #late-interaction-models }

トークン埋め込みタスクをサポートするすべてのモデルは、2 つの入力プロンプトの late interaction を計算して類似度スコアを求める形で、score API も利用できます。トークン埋め込みモデルについて詳しくは、[このページ](token_embed.md)を参照してください。

--8<-- "docs/models/pooling_models/token_embed.md:supported-token-embed-models"

### Bi-encoder { #bi-encoder }

埋め込みタスクをサポートするすべてのモデルは、2 つの入力プロンプトの埋め込みのコサイン類似度を計算して類似度スコアを求める形で、score API も利用できます。埋め込みモデルについて詳しくは、[このページ](embed.md)を参照してください。

--8<-- "docs/models/pooling_models/embed.md:supported-embed-models"

## オフライン推論 { #offline-inference }

### プーリングのパラメータ { #pooling-parameters }

次の[プーリングパラメータ](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.PoolingParams)は cross-encoder モデルでのみサポートされ、late-interaction および bi-encoder モデルでは機能しません。

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
# このコードは上流のソースを参照してください: https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/pooling_params.py
```

### `LLM.score` { #llmscore }

[`score`](https://docs.vllm.ai/en/v0.26.0/api/vllm/entrypoints/pooling/offline/#vllm.entrypoints.pooling.offline.PoolingOfflineMixin.score) メソッドは、文のペア間の類似度スコアを出力します。

```python
from vllm import LLM

llm = LLM(model="BAAI/bge-reranker-v2-m3", runner="pooling")
(output,) = llm.score(
    "What is the capital of France?",
    "The capital of Brazil is Brasilia.",
)

score = output.outputs.score
print(f"Score: {score}")
```

コード例は [examples/basic/offline_inference/score.py](../../../examples/basic/offline_inference/score.py) にあります。

## オンラインサービング { #online-serving }

### Score API { #score-api }

vLLM の Score API（`/score`、`/v1/score`）は `LLM.score` と同様に、2 つの入力プロンプト間の類似度スコアを計算します。

#### パラメータ { #parameters }

サポートされる Score API のパラメータは次のとおりです。

```python
--8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
--8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
--8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-extra-params"
--8<-- "vllm/entrypoints/pooling/scoring/protocol.py:scoring-common-params"
--8<-- "vllm/entrypoints/pooling/scoring/protocol.py:score-request-params"
```

#### 例 { #examples }

##### 単一の推論 { #single-inference }

`queries` と `documents` の両方に文字列を渡すと、1 組の文ペアになります。

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/score' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "BAAI/bge-reranker-v2-m3",
  "encoding_format": "float",
  "queries": "What is the capital of France?",
  "documents": "The capital of France is Paris."
}'
```

??? console "レスポンス"

    ```json
    {
      "id": "score-request-id",
      "object": "list",
      "created": 693447,
      "model": "BAAI/bge-reranker-v2-m3",
      "data": [
        {
          "index": 0,
          "object": "score",
          "score": 1
        }
      ],
      "usage": {}
    }
    ```

##### バッチ推論 { #batch-inference }

`queries` に文字列、`documents` にリストを渡すと、`queries` と `documents` 内の各文字列から成る複数の文ペアが構成されます。
ペアの総数は `len(documents)` です。

??? console "リクエスト"

    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/score' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "model": "BAAI/bge-reranker-v2-m3",
      "queries": "What is the capital of France?",
      "documents": [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris."
      ]
    }'
    ```

??? console "レスポンス"

    ```json
    {
      "id": "score-request-id",
      "object": "list",
      "created": 693570,
      "model": "BAAI/bge-reranker-v2-m3",
      "data": [
        {
          "index": 0,
          "object": "score",
          "score": 0.001094818115234375
        },
        {
          "index": 1,
          "object": "score",
          "score": 1
        }
      ],
      "usage": {}
    }
    ```

`queries` と `documents` の両方にリストを渡すと、`queries` 内の各文字列と `documents` 内の対応する文字列から成る複数の文ペアが構成されます（`zip()` と同様）。
ペアの総数は `len(documents)` です。

??? console "リクエスト"

    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/score' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "model": "BAAI/bge-reranker-v2-m3",
      "encoding_format": "float",
      "queries": [
        "What is the capital of Brazil?",
        "What is the capital of France?"
      ],
      "documents": [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris."
      ]
    }'
    ```

??? console "レスポンス"

    ```json
    {
      "id": "score-request-id",
      "object": "list",
      "created": 693447,
      "model": "BAAI/bge-reranker-v2-m3",
      "data": [
        {
          "index": 0,
          "object": "score",
          "score": 1
        },
        {
          "index": 1,
          "object": "score",
          "score": 1
        }
      ],
      "usage": {}
    }
    ```

##### マルチモーダル入力 { #multi-modal-inputs }

リクエストにマルチモーダル入力（画像など）のリストを含む `content` を渡すことで、スコアリングモデルにマルチモーダル入力を与えられます。具体例は以下を参照してください。

=== "JinaVL-Reranker"

    モデルをサービングするには次のようにします。

    ```bash
    vllm serve jinaai/jina-reranker-m0
    ```

    リクエストのスキーマは OpenAI クライアントで定義されていないため、より低レベルの `requests` ライブラリを使ってサーバーにリクエストを送ります。

    ??? Code

        ```python
        import requests
        
        response = requests.post(
            "http://localhost:8000/v1/score",
            json={
                "model": "jinaai/jina-reranker-m0",
                "queries": "slm markdown",
                "documents": [
                    {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png"
                                },
                            }
                        ],
                    },
                    {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png"
                                },
                            }
                        ]
                    },
                ],
            },
        )
        response.raise_for_status()
        response_json = response.json()
        print("Scoring output:", response_json["data"][0]["score"])
        print("Scoring output:", response_json["data"][1]["score"])
        ```
完全な例:

- [examples/pooling/score/vision_score_api_online.py](../../../examples/pooling/score/vision_score_api_online.py)
- [examples/pooling/score/vision_rerank_api_online.py](../../../examples/pooling/score/vision_rerank_api_online.py)

### Cohere Rerank API { #cohere-rerank-api }

`/rerank`、`/v1/rerank`、`/v2/rerank` の各 API は、広く使われているオープンソースツールとの互換性を確保するため、[Jina AI の rerank API インターフェース](https://jina.ai/reranker/)と [Cohere の rerank API インターフェース](https://docs.cohere.com/v2/reference/rerank)の両方に対応しています。

コード例: [examples/pooling/score/rerank_api_online.py](../../../examples/pooling/score/rerank_api_online.py)

#### パラメータ { #parameters_1 }

サポートされる rerank API のパラメータは次のとおりです。

```python
--8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
--8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
--8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-extra-params"
--8<-- "vllm/entrypoints/pooling/scoring/protocol.py:scoring-common-params"
--8<-- "vllm/entrypoints/pooling/scoring/protocol.py:rerank-request-params"
```

#### 例 { #examples_1 }

リクエストパラメータ `top_n` は省略可能で、既定では `documents` フィールドの長さになります。
結果のドキュメントは関連度順に並び替えられ、`index` プロパティから元の順序を判別できます。

??? console "リクエスト"

    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/v1/rerank' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "model": "BAAI/bge-reranker-base",
      "query": "What is the capital of France?",
      "documents": [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
        "Horses and cows are both animals"
      ]
    }'
    ```

??? console "レスポンス"

    ```json
    {
      "id": "rerank-fae51b2b664d4ed38f5969b612edff77",
      "model": "BAAI/bge-reranker-base",
      "usage": {
        "total_tokens": 56
      },
      "results": [
        {
          "index": 1,
          "document": {
            "text": "The capital of France is Paris."
          },
          "relevance_score": 0.99853515625
        },
        {
          "index": 0,
          "document": {
            "text": "The capital of Brazil is Brasilia."
          },
          "relevance_score": 0.0005860328674316406
        }
      ]
    }
    ```

## その他の例 { #more-examples }

その他の例は [examples/pooling/score](../../../examples/pooling/score) にあります。

## サポートされる機能 { #supported-features }

cross-encoder モデルは、2 つのプロンプトを入力として受け取り num_labels が 1 の出力を返す分類モデルの一種であるため、その機能は（シーケンス）分類と同じになります。詳しくは[このページ](classify.md#supported-features)を参照してください。

### スコアテンプレート { #score-template }

スコアテンプレートがサポートされるのは **cross-encoder** モデルのみです。スコアリングに**埋め込み**モデルを使う場合、vLLM はスコアテンプレートを適用しません。

スコアリングモデルによっては、正しく動作させるために特定のプロンプト形式が必要です。`--chat-template` パラメータでカスタムのスコアテンプレートを指定できます（[チャットテンプレート](../../serving/online_serving/README.md#chat-template)を参照）。

チャットテンプレートと同様に、スコアテンプレートは `messages` のリストを受け取ります。スコアリングでは、各メッセージが `"query"` または `"document"` のいずれかの `role` 属性を持ちます。一般的な point-wise の cross-encoder では、query と document がちょうど 1 つずつ、計 2 つのメッセージが渡されます。query と document の内容にアクセスするには、Jinja の `selectattr` フィルタを使います。

- **Query**: `{{ (messages | selectattr("role", "eq", "query") | first).content }}`
- **Document**: `{{ (messages | selectattr("role", "eq", "document") | first).content }}`

この方法は、インデックスによるアクセス（`messages[0]`、`messages[1]`）よりも堅牢です。メッセージを意味的な役割で選択するためです。また、将来 `messages` に別のメッセージ種別が追加された場合でも、メッセージの並び順に依存せずに済みます。

テンプレートファイルの例: [examples/pooling/score/template/nemotron-rerank.jinja](../../../examples/pooling/score/template/nemotron-rerank.jinja)

### 活性化の有効化 / 無効化 { #enabledisable-activation }

`use_activation` により活性化の有効・無効を切り替えられます。これは cross-encoder モデルでのみ機能します。
