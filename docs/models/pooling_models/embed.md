# 埋め込みの使い方 { #embedding-usages }

埋め込みモデルは、テキスト・画像・音声などの非構造データを、埋め込み（embedding）と呼ばれる構造化された数値表現へ変換するために設計された機械学習モデルの一種です。

## 概要 { #summary }

- モデルの用途: （シーケンス）埋め込み
- プーリングタスク: `embed`
- オフライン API:
    - `LLM.embed(...)`
    - `LLM.encode(..., pooling_task="embed")`
    - `LLM.score(...)`
- オンライン API:
    - [Cohere Embed API](embed.md#cohere-embed-api)（`/v2/embed`）
    - [OpenAI 互換 Embeddings API](embed.md#openai-compatible-embeddings-api)（`/v1/embeddings`）
    - Pooling API（`/pooling`）

（シーケンス）埋め込みとトークン埋め込みの主な違いは出力の粒度にあります。（シーケンス）埋め込みは入力シーケンス全体に対して 1 つの埋め込みベクトルを生成し、トークン埋め込みはシーケンス内の各トークンごとに埋め込みを生成します。

多くの埋め込みモデルは（シーケンス）埋め込みとトークン埋め込みの両方をサポートします。トークン埋め込みの詳細は[このページ](token_embed.md)を参照してください。

## 代表的なユースケース { #typical-use-cases }

### 埋め込み { #embedding }

埋め込みモデルの最も基本的なユースケースは、入力を埋め込むことです（RAG などが典型です）。

### ペアごとの類似度 { #pairwise-similarity }

[Score API](scoring.md) を使って、ペアごとの類似度スコアを計算し、類似度行列を構築できます。

## サポートされるモデル { #supported-models }

--8<-- [start:supported-embed-models]

### テキストのみのモデル { #text-only-models }

| アーキテクチャ | モデル | HF モデルの例 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `BertModel` | BERT-based | `BAAI/bge-base-en-v1.5`, `Snowflake/snowflake-arctic-embed-xs`, etc. | | |
| `BertSpladeSparseEmbeddingModel` | SPLADE | `naver/splade-v3` | | |
| `Gemma2Model`<sup>C</sup> | Gemma 2-based | `BAAI/bge-multilingual-gemma2`, etc. | ✅︎ | ✅︎ |
| `Gemma3TextModel`<sup>C</sup> | Gemma 3-based | `google/embeddinggemma-300m`, etc. | ✅︎ | ✅︎ |
| `GritLM` | GritLM | `parasail-ai/GritLM-7B-vllm`. | ✅︎ | ✅︎ |
| `GteModel` | Arctic-Embed-2.0-M | `Snowflake/snowflake-arctic-embed-m-v2.0`. | | |
| `GteNewModel` | mGTE-TRM (see note) | `Alibaba-NLP/gte-multilingual-base`, etc. | | |
| `JinaEmbeddingsV5Model`<sup>C</sup> | Qwen3-based with task-specific LoRA adapters | `jinaai/jina-embeddings-v5-text-small` (see note) | ✅︎ | ✅︎ |
| `LlamaBidirectionalModel`<sup>C</sup> | Llama-based with bidirectional attention | `nvidia/llama-nemotron-embed-1b-v2`, etc. | ✅︎ | ✅︎ |
| `LlamaModel`<sup>C</sup>, `LlamaForCausalLM`<sup>C</sup>, `MistralModel`<sup>C</sup>, etc. | Llama-based | `intfloat/e5-mistral-7b-instruct`, etc. | ✅︎ | ✅︎ |
| `ModernBertModel` | ModernBERT-based | `Alibaba-NLP/gte-modernbert-base`, etc. | | |
| `NomicBertModel` | Nomic BERT | `nomic-ai/nomic-embed-text-v1`, `nomic-ai/nomic-embed-text-v2-moe`, `Snowflake/snowflake-arctic-embed-m-long`, etc. | | |
| `Qwen2Model`<sup>C</sup>, `Qwen2ForCausalLM`<sup>C</sup> | Qwen2-based | `ssmits/Qwen2-7B-Instruct-embed-base` (see note), `Alibaba-NLP/gte-Qwen2-7B-instruct` (see note), etc. | ✅︎ | ✅︎ |
| `Qwen3Model`<sup>C</sup>, `Qwen3ForCausalLM`<sup>C</sup> | Qwen3-based | `Qwen/Qwen3-Embedding-0.6B`, etc. | ✅︎ | ✅︎ |
| `RobertaModel`, `RobertaForMaskedLM` | RoBERTa-based | `sentence-transformers/all-roberta-large-v1`, etc. | | |
| `VoyageQwen3BidirectionalEmbedModel`<sup>C</sup> | Voyage Qwen3-based with bidirectional attention | `voyageai/voyage-4-nano`, etc. | ✅︎ | ✅︎ |
| `XLMRobertaModel` | XLMRobertaModel-based | `BAAI/bge-m3` (see note), `intfloat/multilingual-e5-base`, `jinaai/jina-embeddings-v3` (see note), etc. | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

!!! note
    第 2 世代の GTE モデル（mGTE-TRM）は `NewModel` という名前です。`NewModel` という名称は汎用的すぎるため、`GteNewModel` アーキテクチャを使うことを明示するには `--hf-overrides '{"architectures": ["GteNewModel"]}'` を指定してください。

!!! note
    `ssmits/Qwen2-7B-Instruct-embed-base` は Sentence Transformers の設定が正しく定義されていません。
    `--pooler-config '{"pooling_type": "MEAN"}'` を渡して平均プーリングを手動で設定する必要があります。

!!! note
    `Alibaba-NLP/gte-Qwen2-*` では、正しいトークナイザーを読み込むために `--trust-remote-code` を有効にする必要があります。
    [HF Transformers の関連 issue](https://github.com/huggingface/transformers/issues/34882) を参照してください。

!!! note
    `BAAI/bge-m3` モデルには、スパース埋め込みと colbert 埋め込みのための追加の重みが含まれています。詳細は[このページ](specific_models.md#baaibge-m3)を参照してください。

!!! note
    `jinaai/jina-embeddings-v3` は LoRA により複数のタスクをサポートしますが、vLLM では当面、LoRA の重みをマージすることで text-matching タスクのみをサポートします。

!!! note
    `jinaai/jina-embeddings-v5-text-small` には、タスク別の LoRA アダプタが 4 つ
    （`retrieval`、`text-matching`、`classification`、`clustering`）同梱されています。vLLM は
    ロード時に選択されたアダプタをベースの重みにマージします。タスクは
    `--hf-overrides '{"jina_task": "<task>"}'` で選択します。既定は `retrieval` です。

### マルチモーダルモデル { #multimodal-models }

!!! note
    マルチモーダルモデルの入力について詳しくは、[このページ](../supported_models.md#list-of-multimodal-language-models)を参照してください。

| アーキテクチャ | モデル | 入力 | HF モデルの例 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `CLIPModel` | CLIP | T / I | `openai/clip-vit-base-patch32`, `openai/clip-vit-large-patch14`, etc. | | |
| `LlamaNemotronVLModel` | Llama Nemotron Embedding + SigLIP | T + I | `nvidia/llama-nemotron-embed-vl-1b-v2` | | |
| `LlavaNextForConditionalGeneration`<sup>C</sup> | LLaVA-NeXT-based | T / I | `royokong/e5-v` | | ✅︎ |
| `Phi3VForCausalLM`<sup>C</sup> | Phi-3-Vision-based | T + I | `TIGER-Lab/VLM2Vec-Full` | | ✅︎ |
| `Qwen3VLForConditionalGeneration`<sup>C</sup> (see note) | Qwen3-VL | T + I + V | `Qwen/Qwen3-VL-Embedding-2B`, etc. | ✅︎ | ✅︎ |
| `SiglipModel` | SigLIP, SigLIP2 | T / I | `google/siglip-base-patch16-224`, `google/siglip2-base-patch16-224` | | |
| `*ForConditionalGeneration`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | \* | N/A | \* | \* |

<sup>C</sup> `--convert embed` により自動的に埋め込みモデルへ変換されます。（[詳細](./README.md#model-conversion)）
\* 機能のサポート状況は元のモデルと同じです。

お使いのモデルが上記の一覧にない場合、vLLM は [`as_embedding_model`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/adapters/#vllm.model_executor.models.adapters.as_embedding_model) を使ってモデルの自動変換を試みます。既定では、プロンプト全体の埋め込みは、最後のトークンに対応する正規化済みの hidden state から抽出されます。

!!! note
    `Qwen3-VL-Embedding` は公式には画像の前処理に `qwen_vl_utils` を使いますが、vLLM は `transformers` の `video_processing_qwen3_vl` を使うため、公式の Hugging Face リポジトリの例とはわずかに結果が異なります。`qwen_vl_utils` を使ったオフライン推論のサンプルコードは [vision_embedding_offline.py](../../../examples/pooling/embed/vision_embedding_offline.py) にあります。

!!! note
    vLLM は `--convert embed` により任意のアーキテクチャのモデルを埋め込みモデルへ自動変換できますが、最良の結果を得るには、そのために専用に学習されたプーリングモデルを使うべきです。

--8<-- [end:supported-embed-models]

## オフライン推論 { #offline-inference }

### プーリングのパラメータ { #pooling-parameters }

次の[プーリングパラメータ](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.PoolingParams)がサポートされています。

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
--8<-- "vllm/pooling_params.py:embed-pooling-params"
```

### `LLM.embed` { #llmembed }

[`embed`](https://docs.vllm.ai/en/v0.26.0/api/vllm/entrypoints/pooling/offline/#vllm.entrypoints.pooling.offline.PoolingOfflineMixin.embed) メソッドは、プロンプトごとに埋め込みベクトルを出力します。

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.embed("Hello, my name is")

embeds = output.outputs.embedding
print(f"Embeddings: {embeds!r} (size={len(embeds)})")
```

コード例は [examples/basic/offline_inference/embed.py](../../../examples/basic/offline_inference/embed.py) にあります。

### `LLM.encode` { #llmencode }

[`encode`](https://docs.vllm.ai/en/v0.26.0/api/vllm/entrypoints/pooling/offline/#vllm.entrypoints.pooling.offline.PoolingOfflineMixin.encode) メソッドは、vLLM のすべてのプーリングモデルで利用できます。

埋め込みモデルで `LLM.encode` を使う場合は `pooling_task="embed"` を指定します。

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="embed")

data = output.outputs.data
print(f"Data: {data!r}")
```

### `LLM.score` { #llmscore }

[`score`](https://docs.vllm.ai/en/v0.26.0/api/vllm/entrypoints/pooling/offline/#vllm.entrypoints.pooling.offline.PoolingOfflineMixin.score) メソッドは、文のペア間の類似度スコアを出力します。

埋め込みタスクをサポートするすべてのモデルは、2 つの入力プロンプトの埋め込みのコサイン類似度を計算して類似度スコアを求める形で、score API も利用できます。

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.score(
    "What is the capital of France?",
    "The capital of Brazil is Brasilia.",
)

score = output.outputs.score
print(f"Score: {score}")
```

## オンラインサービング { #online-serving }

### OpenAI 互換 Embeddings API { #openai-compatible-embeddings-api }

vLLM の Embeddings API は [OpenAI の Embeddings API](https://platform.openai.com/docs/api-reference/embeddings) と互換であり、[公式の OpenAI Python クライアント](https://github.com/openai/openai-python)から利用できます。

コード例: [examples/pooling/embed/openai_embedding_client.py](../../../examples/pooling/embed/openai_embedding_client.py)

#### Completion のパラメータ { #completion-parameters }

次の Classification API のパラメータがサポートされています。

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:completion-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:encoding-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:embed-params"
    ```

次の追加パラメータがサポートされています。

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:completion-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:encoding-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:embed-extra-params"
    ```

#### Chat のパラメータ { #chat-parameters }

チャット形式の入力（つまり `messages` を渡す場合）では、次のパラメータがサポートされています。

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:chat-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:encoding-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:embed-params"
    ```

代わりに、次の追加パラメータがサポートされます。

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:chat-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:encoding-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:embed-extra-params"
    ```

#### 例 { #examples }

モデルに[チャットテンプレート](../../serving/online_serving/README.md#chat-template)がある場合、`inputs` の代わりに `messages` のリスト（[Chat API](../../serving/online_serving/openai_compatible_server.md#chat-api) と同じスキーマ）を渡せます。これはモデルへの単一のプロンプトとして扱われます。OpenAI の型注釈を保ったまま API を呼び出す便利な関数を次に示します。

??? code

    ```python
    from openai import OpenAI
    from openai._types import NOT_GIVEN, NotGiven
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.create_embedding_response import CreateEmbeddingResponse

    def create_chat_embeddings(
        client: OpenAI,
        *,
        messages: list[ChatCompletionMessageParam],
        model: str,
        encoding_format: Union[Literal["base64", "float"], NotGiven] = NOT_GIVEN,
    ) -> CreateEmbeddingResponse:
        return client.post(
            "/embeddings",
            cast_to=CreateEmbeddingResponse,
            body={"messages": messages, "model": model, "encoding_format": encoding_format},
        )
    ```

##### マルチモーダル入力 { #multi-modal-inputs }

サーバー用のカスタムチャットテンプレートを定義し、リクエストで `messages` のリストを渡すことで、埋め込みモデルにマルチモーダル入力を与えられます。具体例は以下を参照してください。

=== "VLM2Vec"

    モデルをサービングするには次のようにします。

    ```bash
    vllm serve TIGER-Lab/VLM2Vec-Full --runner pooling \
      --trust-remote-code \
      --max-model-len 4096 \
      --chat-template examples/pooling/embed/template/vlm2vec_phi3v.jinja
    ```

    !!! important
        VLM2Vec は Phi-3.5-Vision と同じモデルアーキテクチャを持つため、テキスト生成モードではなく
        埋め込みモードでこのモデルを実行するには、`--runner pooling` を明示的に渡す必要があります。

        このモデル用のカスタムチャットテンプレートは元のものとまったく異なります。
        テンプレートは [examples/pooling/embed/template/vlm2vec_phi3v.jinja](../../../examples/pooling/embed/template/vlm2vec_phi3v.jinja) にあります。

    リクエストのスキーマは OpenAI クライアントで定義されていないため、より低レベルの `requests` ライブラリを使ってサーバーにリクエストを送ります。

    ??? code

        ```python
        from openai import OpenAI
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
        )
        image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

        response = create_chat_embeddings(
            client,
            model="TIGER-Lab/VLM2Vec-Full",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "Represent the given image."},
                    ],
                }
            ],
            encoding_format="float",
        )

        print("Image embedding output:", response.data[0].embedding)
        ```

=== "DSE-Qwen2-MRL"

    モデルをサービングするには次のようにします。

    ```bash
    vllm serve MrLight/dse-qwen2-2b-mrl-v1 --runner pooling \
      --trust-remote-code \
      --max-model-len 8192 \
      --chat-template examples/pooling/embed/template/dse_qwen2_vl.jinja
    ```

    !!! important
        VLM2Vec と同様に、`--runner pooling` を明示的に渡す必要があります。

        さらに `MrLight/dse-qwen2-2b-mrl-v1` は埋め込みに EOS トークンを必要とし、これはカスタムチャットテンプレート
        [examples/pooling/embed/template/dse_qwen2_vl.jinja](../../../examples/pooling/embed/template/dse_qwen2_vl.jinja) で処理されます。

    !!! important
        `MrLight/dse-qwen2-2b-mrl-v1` は、テキストクエリの埋め込みに最小サイズのプレースホルダー画像を必要とします。
        詳細は下記の完全なコード例を参照してください。

完全な例: [examples/pooling/embed/vision_embedding_online.py](../../../examples/pooling/embed/vision_embedding_online.py)

### Cohere Embed API { #cohere-embed-api }

vLLM の API は [Cohere の Embed v2 API](https://docs.cohere.com/reference/embed) とも互換です。この API は、切り詰め、出力次元数、埋め込みの型、入力の種類といった近年の埋め込み機能をサポートします。このエンドポイントは、（マルチモーダルモデルを含む）任意の埋め込みモデルで動作します。

#### Cohere Embed API のリクエストパラメータ { #cohere-embed-api-request-parameters }

| パラメータ | 型 | 必須 | 説明 |
| --------- | ---- | -------- | ----------- |
| `model` | string | はい | モデル名 |
| `input_type` | string | いいえ | プロンプト接頭辞のキー（モデル依存。下記参照） |
| `texts` | list[string] | いいえ | テキスト入力（`texts`、`images`、`inputs` のいずれか 1 つを使います） |
| `images` | list[string] | いいえ | Base64 のデータ URI 形式の画像 |
| `inputs` | list[object] | いいえ | テキストと画像が混在したコンテンツオブジェクト |
| `embedding_types` | list[string] | いいえ | 出力の型（既定: `["float"]`） |
| `output_dimension` | int | いいえ | 埋め込みをこの次元数に切り詰めます（Matryoshka） |
| `truncate` | string | いいえ | `END`、`START`、`NONE`（既定: `END`） |

#### テキストの埋め込み { #text-embedding }

```bash
curl -X POST "http://localhost:8000/v2/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Snowflake/snowflake-arctic-embed-m-v1.5",
    "input_type": "query",
    "texts": ["Hello world", "How are you?"],
    "embedding_types": ["float"]
  }'
```

??? console "レスポンス"

    ```json
    {
      "id": "embd-...",
      "embeddings": {
        "float": [
          [0.012, -0.034, ...],
          [0.056, 0.078, ...]
        ]
      },
      "texts": ["Hello world", "How are you?"],
      "meta": {
        "api_version": {"version": "2"},
        "billed_units": {"input_tokens": 12}
      }
    }
    ```

#### テキストと画像の混在入力 { #mixed-text-and-image-inputs }

マルチモーダルモデルでは、base64 のデータ URI を渡すことで画像を埋め込めます。`inputs` フィールドは、テキストと画像のコンテンツが混在したオブジェクトのリストを受け取ります。

```bash
curl -X POST "http://localhost:8000/v2/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/siglip-so400m-patch14-384",
    "inputs": [
      {
        "content": [
          {"type": "text", "text": "A photo of a cat"},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
        ]
      }
    ],
    "embedding_types": ["float"]
  }'
```

#### 埋め込みの型 { #embedding-types }

`embedding_types` パラメータは出力の形式を制御します。1 回の呼び出しで複数の型を要求できます。

| 型 | 説明 |
| ---- | ----------- |
| `float` | 生の float32 の埋め込み（既定） |
| `binary` | ビットパックされた符号付きバイナリ |
| `ubinary` | ビットパックされた符号なしバイナリ |
| `base64` | リトルエンディアンの float32 を base64 でエンコードしたもの |

```bash
curl -X POST "http://localhost:8000/v2/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Snowflake/snowflake-arctic-embed-m-v1.5",
    "input_type": "query",
    "texts": ["What is machine learning?"],
    "embedding_types": ["float", "binary"]
  }'
```

??? console "レスポンス"

    ```json
    {
      "id": "embd-...",
      "embeddings": {
        "float": [[0.012, -0.034, ...]],
        "binary": [[42, -117, ...]]
      },
      "texts": ["What is machine learning?"],
      "meta": {
        "api_version": {"version": "2"},
        "billed_units": {"input_tokens": 8}
      }
    }
    ```

#### 切り詰め { #truncation }

`truncate` パラメータは、モデルの最大シーケンス長を超える入力の扱いを制御します。

| 値 | 挙動 |
| ----- | --------- |
| `END`（既定） | 先頭のトークンを残し、末尾を切り捨てます |
| `START` | 末尾のトークンを残し、先頭を切り捨てます |
| `NONE` | 入力が長すぎる場合はエラーを返します |

#### input type とプロンプト接頭辞 { #input-type-and-prompt-prefixes }

`input_type` フィールドは、各テキスト入力の先頭に付けるプロンプト接頭辞を選択します。指定できる値はモデルによって異なります。

- **`config.json` に `task_instructions` を持つモデル**: `task_instructions` 辞書のキーが有効な `input_type` の値であり、対応する値が各テキストの先頭に付加されます。
- **`config_sentence_transformers.json` に prompts を持つモデル**: `prompts` 辞書のキーが有効な `input_type` の値です。たとえば `Snowflake/snowflake-arctic-embed-xs` は `"query"` を定義しているため、`input_type: "query"` を設定すると `"Represent this sentence for searching relevant passages: "` が先頭に付加されます。
- **その他のモデル**: `input_type` は受け付けられず、渡すとバリデーションエラーになります。

## その他の例 { #more-examples }

その他の例は [examples/pooling/embed](../../../examples/pooling/embed) にあります。

## サポートされる機能 { #supported-features }

### 正規化の有効化 / 無効化 { #enabledisable-normalize }

`use_activation` により正規化の有効・無効を切り替えられます。

### Matryoshka 埋め込み { #matryoshka-embeddings }

[Matryoshka Embeddings](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html#matryoshka-embeddings)（[Matryoshka Representation Learning（MRL）](https://arxiv.org/abs/2205.13147)）は、埋め込みモデルの学習に用いられる手法です。これにより、性能とコストのトレードオフを調整できます。

!!! warning
    すべての埋め込みモデルが Matryoshka Representation Learning で学習されているわけではありません。`dimensions` パラメータの誤用を避けるため、vLLM は Matryoshka 埋め込みをサポートしないモデルの出力次元数を変更しようとするリクエストに対してエラーを返します。

    たとえば `BAAI/bge-m3` モデルを使いながら `dimensions` パラメータを設定すると、次のエラーになります。

    ```json
    {"object":"error","message":"Model \"BAAI/bge-m3\" does not support matryoshka representation, changing output dimensions will lead to poor results.","type":"BadRequestError","param":null,"code":400}
    ```

#### Matryoshka 埋め込みを手動で有効にする { #manually-enable-matryoshka-embeddings }

現時点で、Matryoshka 埋め込みのサポートを示す公式のインターフェースはありません。vLLM では、`config.json` の `is_matryoshka` が `True` であれば、出力次元数を任意の値に変更できます。許容される出力次元数は `matryoshka_dimensions` で制御します。

Matryoshka 埋め込みをサポートしているものの vLLM がそれを認識できないモデルでは、設定を手動で上書きしてください。オフラインでは `hf_overrides={"is_matryoshka": True}` または `hf_overrides={"matryoshka_dimensions": [<許容する出力次元数>]}`、オンラインでは `--hf-overrides '{"is_matryoshka": true}'` または `--hf-overrides '{"matryoshka_dimensions": [<許容する出力次元数>]}'` を使います。

Matryoshka 埋め込みを有効にしてモデルをサービングする例を次に示します。

```bash
vllm serve Snowflake/snowflake-arctic-embed-m-v1.5 --hf-overrides '{"matryoshka_dimensions":[256]}'
```

#### オフライン推論 { #offline-inference_1 }

Matryoshka 埋め込みをサポートする埋め込みモデルでは、[`PoolingParams`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.PoolingParams) の dimensions パラメータで出力次元数を変更できます。

```python
from vllm import LLM, PoolingParams

llm = LLM(
    model="jinaai/jina-embeddings-v3",
    runner="pooling",
    trust_remote_code=True,
)
outputs = llm.embed(
    ["Follow the white rabbit."],
    pooling_params=PoolingParams(dimensions=32),
)
print(outputs[0].outputs)
```

コード例は [examples/pooling/embed/embed_matryoshka_fy_offline.py](../../../examples/pooling/embed/embed_matryoshka_fy_offline.py) にあります。

#### オンライン推論 { #online-inference }

次のコマンドで vLLM サーバーを起動します。

```bash
vllm serve jinaai/jina-embeddings-v3 --trust-remote-code
```

Matryoshka 埋め込みをサポートする埋め込みモデルでは、dimensions パラメータで出力次元数を変更できます。

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Follow the white rabbit.",
    "model": "jinaai/jina-embeddings-v3",
    "encoding_format": "float",
    "dimensions": 32
  }'
```

想定される出力:

```json
{"id":"embd-5c21fc9a5c9d4384a1b021daccaf9f64","object":"list","created":1745476417,"model":"jinaai/jina-embeddings-v3","data":[{"index":0,"object":"embedding","embedding":[-0.3828125,-0.1357421875,0.03759765625,0.125,0.21875,0.09521484375,-0.003662109375,0.1591796875,-0.130859375,-0.0869140625,-0.1982421875,0.1689453125,-0.220703125,0.1728515625,-0.2275390625,-0.0712890625,-0.162109375,-0.283203125,-0.055419921875,-0.0693359375,0.031982421875,-0.04052734375,-0.2734375,0.1826171875,-0.091796875,0.220703125,0.37890625,-0.0888671875,-0.12890625,-0.021484375,-0.0091552734375,0.23046875]}],"usage":{"prompt_tokens":8,"total_tokens":8,"completion_tokens":0,"prompt_tokens_details":null}}
```

OpenAI クライアントを使った例は [examples/pooling/embed/openai_embedding_matryoshka_fy_client.py](../../../examples/pooling/embed/openai_embedding_matryoshka_fy_client.py) にあります。

## 削除された機能 { #removed-features }

### PoolingParams からの `normalize` の削除 { #remove-normalize-from-poolingparams }

`normalize` は PoolingParams からすでに削除されています。代わりに `use_activation` を使ってください。
