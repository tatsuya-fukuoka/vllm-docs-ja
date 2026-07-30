# 報酬モデルの利用 { #reward-usages }

報酬モデル（RM: reward model）は、言語モデルが生成した出力の品質を評価・スコア付けし、人間の選好の代理として機能するよう設計されたモデルです。

## 概要 { #summary }

- モデルの用途: 報酬
- プーリングタスク:

| モデルの種類                        | プーリングタスク  |
|------------------------------------|----------------|
| （シーケンス）（結果）報酬モデル | classify       |
| トークン（結果）報酬モデル      | token_classify |
| プロセス報酬モデル              | token_classify |

- オフライン API:
    - `LLM.encode(..., pooling_task="...")`
- オンライン API:
    - プーリング API（`/pooling`）

## 対応モデル { #supported-models }

### 報酬モデル { #reward-models }

シーケンス分類モデルを（シーケンス）（結果）報酬モデルとして使う場合、使い方とサポートされる機能は通常の[分類モデル](classify.md)と同じです。

--8<-- [start:supported-sequence-reward-models]

| Architecture | Models | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `JambaForSequenceClassification` | Jamba | `ai21labs/Jamba-tiny-reward-dev`, etc. | ✅︎ | ✅︎ |
| `Qwen3ForSequenceClassification`<sup>C</sup> | Qwen3-based | `Skywork/Skywork-Reward-V2-Qwen3-0.6B`, etc. | ✅︎ | ✅︎ |
| `LlamaForSequenceClassification`<sup>C</sup> | Llama-based | `Skywork/Skywork-Reward-V2-Llama-3.2-1B`, etc. | ✅︎ | ✅︎ |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

<sup>C</sup> `--convert classify` によって自動的に分類モデルへ変換されます。（[詳細](./README.md#model-conversion)）  

上記の一覧にモデルがない場合は、[`as_seq_cls_model`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/adapters/#vllm.model_executor.models.adapters.as_seq_cls_model) を使ってモデルの自動変換を試みます。既定では、最後のトークンに対応する隠れ状態を softmax したものからクラス確率が抽出されます。

--8<-- [end:supported-sequence-reward-models]

### トークン報酬モデル { #token-reward-models }

（シーケンス）分類とトークン分類の主な違いは、出力の粒度にあります。（シーケンス）分類は入力シーケンス全体に対して 1 つの結果を出力するのに対し、トークン分類はシーケンス内の各トークンごとに結果を出力します。

トークン分類モデルをトークン（結果）報酬モデルとして使う場合、使い方とサポートされる機能は通常の[トークン分類モデル](token_classify.md)と同じです。

--8<-- [start:supported-token-reward-models]

| Architecture | Models | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `InternLM2ForRewardModel` | InternLM2-based | `internlm/internlm2-1_8b-reward`, `internlm/internlm2-7b-reward`, etc. | ✅︎ | ✅︎ |
| `Qwen2ForRewardModel` | Qwen2-based | `Qwen/Qwen2.5-Math-RM-72B`, etc. | ✅︎ | ✅︎ |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

<sup>C</sup> `--convert classify` によって自動的に分類モデルへ変換されます。（[詳細](./README.md#model-conversion)）  

上記の一覧にモデルがない場合は、[`as_seq_cls_model`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/adapters/#vllm.model_executor.models.adapters.as_seq_cls_model) を使ってモデルの自動変換を試みます。

--8<-- [end:supported-token-reward-models]

### プロセス報酬モデル { #process-reward-models }

中間ステップを評価するために使われるプロセス報酬モデルは、望ましい結果を得るうえで重要な役割を果たします。

| Architecture | Models | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `LlamaForCausalLM` | Llama-based | `peiyi9979/math-shepherd-mistral-7b-prm`, etc. | ✅︎ | ✅︎ |
| `Qwen2ForProcessRewardModel` | Qwen2-based | `Qwen/Qwen2.5-Math-PRM-7B`, etc. | ✅︎ | ✅︎ |

!!! important
    `peiyi9979/math-shepherd-mistral-7b-prm` のようなプロセス教師あり報酬モデルでは、プーリングの設定を
    明示的に指定する必要があります。例:
    `--pooler-config '{"pooling_type": "STEP", "step_tag_id": 123, "returned_token_ids": [456, 789]}'`

## オフライン推論 { #offline-inference }

### プーリングパラメータ { #pooling-parameters }

次の[プーリングパラメータ](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.PoolingParams)がサポートされています。

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
# このコードは上流のソースを参照してください: https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/pooling_params.py
```

### `LLM.encode` { #llmencode }

[`encode`](https://docs.vllm.ai/en/v0.26.0/api/vllm/entrypoints/pooling/offline/#vllm.entrypoints.pooling.offline.PoolingOfflineMixin.encode) メソッドは、vLLM のすべてのプーリングモデルで利用できます。

- 報酬モデル

（シーケンス）（結果）報酬モデルで `LLM.encode` を使う場合は `pooling_task="classify"` を指定します。

```python
from vllm import LLM

llm = LLM(model="Skywork/Skywork-Reward-V2-Qwen3-0.6B", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="classify")

data = output.outputs.data
print(f"Data: {data!r}")
```

- トークン報酬モデル

トークン（結果）報酬モデルで `LLM.encode` を使う場合は `pooling_task="token_classify"` を指定します。

```python
from vllm import LLM

llm = LLM(model="internlm/internlm2-1_8b-reward", runner="pooling", trust_remote_code=True)
(output,) = llm.encode("Hello, my name is", pooling_task="token_classify")

data = output.outputs.data
print(f"Data: {data!r}")
```

- プロセス報酬モデル

トークン（結果）報酬モデルで `LLM.encode` を使う場合は `pooling_task="token_classify"` を指定します。

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen2.5-Math-PRM-7B", runner="pooling")
(output,) = llm.encode("Hello, my name is<extra_0><extra_0><extra_0>", pooling_task="token_classify")

data = output.outputs.data
print(f"Data: {data!r}")
```

## オンラインサービング { #online-serving }

[プーリング API](README.md#pooling-api) を参照してください。報酬モデルの種類に対応するプーリングタスクは[上の表](#summary)を参照してください。

## その他の例 { #more-examples }

その他の例はこちらにあります: [examples/pooling/reward](../../../examples/pooling/reward)

## 非推奨の機能 { #deprecated-features }

### `LLM.reward` { #llmreward }

`llm.reward` API は非推奨となり、v0.24 で削除されました。代わりに `LLM.encode` を `pooling_task="classify"` または `pooling_task="token_classify"` と組み合わせて使ってください。
