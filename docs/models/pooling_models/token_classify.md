# トークン分類の利用 { #token-classification-usages }

## 概要 { #summary }

- モデルの用途: トークン分類
- プーリングタスク: `token_classify`
- オフライン API:
    - `LLM.encode(..., pooling_task="token_classify")`
- オンライン API:
    - プーリング API（`/pooling`）

（シーケンス）分類とトークン分類の主な違いは、出力の粒度にあります。（シーケンス）分類は入力シーケンス全体に対して 1 つの結果を出力するのに対し、トークン分類はシーケンス内の各トークンごとに結果を出力します。

多くの分類モデルは、（シーケンス）分類とトークン分類の両方をサポートしています。（シーケンス）分類の詳細については、[このページ](classify.md)を参照してください。

!!! note

    プーリングのマルチタスク対応は v0.21 で削除されました。既定のプーリングタスク（classify）が
    目的に合わない場合は、オフラインでは `PoolerConfig(task="token_classify")`、オンラインでは
    `--pooler-config.task token_classify` で手動で指定する必要があります。

## 代表的なユースケース { #typical-use-cases }

### 固有表現抽出（NER） { #named-entity-recognition-ner }

実装例は次を参照してください。

オフライン: [examples/pooling/token_classify/ner_offline.py](../../../examples/pooling/token_classify/ner_offline.py)

オンライン: [examples/pooling/token_classify/ner_online.py](../../../examples/pooling/token_classify/ner_online.py)

### 強制アライメント { #forced-alignment }

強制アライメント（forced alignment）は、音声と参照テキストを入力として、単語レベルのタイムスタンプを出力します。

オフライン: [examples/pooling/token_classify/forced_alignment_offline.py](../../../examples/pooling/token_classify/forced_alignment_offline.py)

### スパース検索（語彙マッチング） { #sparse-retrieval-lexical-matching }

BAAI/bge-m3 モデルは、スパース検索のためにトークン分類を活用しています。詳細は[このページ](specific_models.md#baaibge-m3)を参照してください。

## 対応モデル { #supported-models }

| アーキテクチャ | モデル | HF モデルの例 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | --------------------------- | --------------------------------------- |
| `BertForTokenClassification` | bert-based | `boltuix/NeuroBERT-NER` (see note), etc. | | |
| `ModernBertForTokenClassification` | ModernBERT-based | `disham993/electrical-ner-ModernBERT-base` | | |
| `OpenAIPrivacyFilterForTokenClassification` | gpt-oss-based encoder | `openai/privacy-filter` | | |
| `Qwen3ForTokenClassification`<sup>C</sup> | Qwen3-based | `bd2lcco/Qwen3-0.6B-finetuned` | | |
| `RobertaForTokenClassification` | RoBERTa-based | `Jean-Baptiste/roberta-large-ner-english` | | |
| `XLMRobertaForTokenClassification` | XLM-RoBERTa-based | `Davlan/xlm-roberta-base-ner-hrl` | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

<sup>C</sup> `--convert classify` によって自動的に分類モデルへ変換されます。（[詳細](./README.md#model-conversion)）
\* 機能のサポート状況は元のモデルと同じです。

上記の一覧にモデルがない場合は、[`as_seq_cls_model`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/adapters/#vllm.model_executor.models.adapters.as_seq_cls_model) を使ってモデルの自動変換を試みます。既定では、最後のトークンに対応する隠れ状態を softmax したものからクラス確率が抽出されます。

### マルチモーダルモデル { #multimodal-models }

!!! note
    マルチモーダルモデルの入力については、[このページ](../supported_models.md#list-of-multimodal-language-models)を参照してください。

| アーキテクチャ                                  | モデル              | 入力            | HF モデルの例                          | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| --------------------------------------------- | ------------------- | ----------------- | ------------------------------------------ | ------------------------------ | ------------------------------------------ |
| `Qwen3ASRForcedAlignerForTokenClassification` | Qwen3-ForcedAligner | T + A<sup>+</sup> | `Qwen/Qwen3-ForcedAligner-0.6B` (see note) |                                | ✅︎                                         |

!!! note
    強制アライメントを利用するには `--hf-overrides '{"architectures": ["Qwen3ASRForcedAlignerForTokenClassification"]}'` が必要です。
    [examples/pooling/token_classify/forced_alignment_offline.py](../../../examples/pooling/token_classify/forced_alignment_offline.py) を参照してください。

### 報酬モデル { #reward-models }

トークン分類モデルを報酬モデルとして使う場合です。報酬モデルの詳細は[報酬モデル](reward.md)を参照してください。

--8<-- "docs/models/pooling_models/reward.md:supported-token-reward-models"

## オフライン推論 { #offline-inference }

### プーリングパラメータ { #pooling-parameters }

次の[プーリングパラメータ](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.PoolingParams)がサポートされています。

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
# このコードは上流のソースを参照してください: https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/pooling_params.py
```

### `LLM.encode` { #llmencode }

[`encode`](https://docs.vllm.ai/en/v0.26.0/api/vllm/entrypoints/pooling/offline/#vllm.entrypoints.pooling.offline.PoolingOfflineMixin.encode) メソッドは、vLLM のすべてのプーリングモデルで利用できます。

トークン分類モデルで `LLM.encode` を使う場合は `pooling_task="token_classify"` を指定します。

```python
from vllm import LLM

llm = LLM(model="boltuix/NeuroBERT-NER", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="token_classify")

data = output.outputs.data
print(f"Data: {data!r}")
```

## オンラインサービング { #online-serving }

[プーリング API](README.md#pooling-api) を参照し、`"task":"token_classify"` を指定してください。

## その他の例 { #more-examples }

その他の例はこちらにあります: [examples/pooling/token_classify](../../../examples/pooling/token_classify)

## サポートされる機能 { #supported-features }

トークン分類の機能は（シーケンス）分類と一致しているはずです。詳細は[このページ](classify.md#supported-features)を参照してください。
