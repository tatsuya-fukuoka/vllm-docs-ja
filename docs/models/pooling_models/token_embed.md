# トークン埋め込みの利用 { #token-embedding-usages }

## 概要 { #summary }

- モデルの用途: トークン分類モデル
- プーリングタスク: `token_embed`
- オフライン API:
    - `LLM.encode(..., pooling_task="token_embed")`
- オンライン API:
    - プーリング API（`/pooling`）

（シーケンス）埋め込みタスクとトークン埋め込みタスクの違いは、（シーケンス）埋め込みがシーケンスごとに 1 つの埋め込みを出力するのに対し、トークン埋め込みはトークンごとに埋め込みを出力する点です。

多くの埋め込みモデルは、（シーケンス）埋め込みとトークン埋め込みの両方をサポートしています。（シーケンス）埋め込みの詳細については、[このページ](embed.md)を参照してください。

!!! note

    Pooling multitask support has been removed since v0.21. When the default pooling task (embed) is not 
    what you want, you need to manually specify it via `PoolerConfig(task="token_embed")` offline or
    `--pooler-config.task token_embed` online.

## 代表的なユースケース { #typical-use-cases }

### マルチベクトル検索 { #multi-vector-retrieval }

実装例は次を参照してください。

オフライン: [examples/pooling/token_embed/multi_vector_retrieval_offline.py](../../../examples/pooling/token_embed/multi_vector_retrieval_offline.py)

オンライン: [examples/pooling/token_embed/multi_vector_retrieval_online.py](../../../examples/pooling/token_embed/multi_vector_retrieval_online.py)

### late interaction { #late-interaction }

score API を通じて、2 つの入力プロンプト間の late interaction による類似度スコアを計算できます。詳細は [Score API](scoring.md) を参照してください。

### 最終隠れ状態の抽出 { #extract-last-hidden-states }

任意のアーキテクチャのモデルは、`--convert embed` を使って埋め込みモデルに変換できます。その後、トークン埋め込みを使ってこれらのモデルから最終隠れ状態を抽出できます。

## 対応モデル { #supported-models }

--8<-- [start:supported-token-embed-models]

### テキストのみのモデル { #text-only-models }

| アーキテクチャ | モデル | HF モデルの例 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `ColBERTLfm2Model` | LFM2 | `LiquidAI/LFM2-ColBERT-350M` | | |
| `ColBERTModernBertModel` | ModernBERT | `lightonai/GTE-ModernColBERT-v1` | | |
| `ColBERTJinaRobertaModel` | Jina XLM-RoBERTa | `jinaai/jina-colbert-v2` | | |
| `HF_ColBERT` | BERT | `answerdotai/answerai-colbert-small-v1`, `colbert-ir/colbertv2.0` | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

### マルチモーダルモデル { #multimodal-models }

!!! note
    マルチモーダルモデルの入力については、[このページ](../supported_models.md#list-of-multimodal-language-models)を参照してください。

| アーキテクチャ | モデル | 入力 | HF モデルの例 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----- | ----------------- | ------------------------------ | ------------------------------------------ |
| `ColModernVBertForRetrieval` | ColModernVBERT | T / I | `ModernVBERT/colmodernvbert-merged` | | |
| `ColPaliForRetrieval` | ColPali | T / I | `vidore/colpali-v1.3-hf` | | |
| `ColQwen3` | Qwen3-VL | T / I | `TomoroAI/tomoro-colqwen3-embed-4b`, `TomoroAI/tomoro-colqwen3-embed-8b` | | |
| `ColQwen3_5` | ColQwen3.5 | T + I + V | `athrael-soju/colqwen3.5-4.5B-v3`, `vultr/VultronRetrieverPrime-Qwen3.5-8B` | | |
| `OpsColQwen3Model` | Qwen3-VL | T / I | `OpenSearch-AI/Ops-Colqwen3-4B`, `OpenSearch-AI/Ops-Colqwen3-8B` | | |
| `Qwen3VLNemotronEmbedModel` | Qwen3-VL | T / I | `nvidia/nemotron-colembed-vl-4b-v2`, `nvidia/nemotron-colembed-vl-8b-v2` | ✅︎ | ✅︎ |
| `*ForConditionalGeneration`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | \* | N/A | \* | \* |

<sup>C</sup> `--convert embed` によって自動的に埋め込みモデルへ変換されます。（[詳細](./README.md#model-conversion)）  
\* 機能のサポート状況は元のモデルと同じです。

上記の一覧にモデルがない場合は、[`as_embedding_model`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/adapters/#vllm.model_executor.models.adapters.as_embedding_model) を使ってモデルの自動変換を試みます。

### 特殊なモデル { #special-models }

| アーキテクチャ | モデル | HF モデルの例 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `JinaForRanking` | Qwen3-based | `jinaai/jina-reranker-v3` | | |

jina-reranker-v3 は、`last but not late interaction` という新しいアーキテクチャを採用したリストワイズの文書リランカーモデルです。詳細は [examples/pooling/token_embed/jina_reranker_v3_offline.py](../../../examples/pooling/token_embed/jina_reranker_v3_offline.py) を参照してください。

--8<-- [end:supported-token-embed-models]

## オフライン推論 { #offline-inference }

### プーリングパラメータ { #pooling-parameters }

次の[プーリングパラメータ](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.PoolingParams)がサポートされています。

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
--8<-- "vllm/pooling_params.py:embed-pooling-params"
```

### `LLM.encode` { #llmencode }

[`encode`](https://docs.vllm.ai/en/v0.26.0/api/vllm/entrypoints/pooling/offline/#vllm.entrypoints.pooling.offline.PoolingOfflineMixin.encode) メソッドは、vLLM のすべてのプーリングモデルで利用できます。

トークン埋め込みモデルで `LLM.encode` を使う場合は `pooling_task="token_embed"` を指定します。

```python
from vllm import LLM

llm = LLM(model="answerdotai/answerai-colbert-small-v1", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="token_embed")

data = output.outputs.data
print(f"Data: {data!r}")
```

### `LLM.score` { #llmscore }

[`score`](https://docs.vllm.ai/en/v0.26.0/api/vllm/entrypoints/pooling/offline/#vllm.entrypoints.pooling.offline.PoolingOfflineMixin.score) メソッドは、文のペア間の類似度スコアを出力します。

トークン埋め込みタスクをサポートするすべてのモデルは、2 つの入力プロンプトの late interaction を計算することで類似度スコアを求める score API の利用もサポートしています。

```python
from vllm import LLM

llm = LLM(model="answerdotai/answerai-colbert-small-v1", runner="pooling")
(output,) = llm.score(
    "What is the capital of France?",
    "The capital of Brazil is Brasilia.",
)

score = output.outputs.score
print(f"Score: {score}")
```

## オンラインサービング { #online-serving }

[プーリング API](README.md#pooling-api) を参照し、`"task":"token_embed"` を指定してください。

## その他の例 { #more-examples }

その他の例はこちらにあります: [examples/pooling/token_embed](../../../examples/pooling/token_embed)

## サポートされる機能 { #supported-features }

トークン埋め込みの機能は（シーケンス）埋め込みと一致しているはずです。詳細は[このページ](embed.md#supported-features)を参照してください。
