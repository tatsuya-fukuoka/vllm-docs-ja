# 分類の利用 { #classification-usages }

分類（classification）とは、与えられた入力に最もよく対応する、あらかじめ定義されたカテゴリ・クラス・ラベルを予測することです。

## 概要 { #summary }

- モデルの用途: （シーケンス）分類
- プーリングタスク: `classify`
- オフライン API:
    - `LLM.classify(...)`
    - `LLM.encode(..., pooling_task="classify")`
- オンライン API:
    - [分類 API](classify.md#online-serving)（`/classify`）
    - プーリング API（`/pooling`）

（シーケンス）分類とトークン分類の主な違いは、出力の粒度にあります。（シーケンス）分類は入力シーケンス全体に対して 1 つの結果を出力するのに対し、トークン分類はシーケンス内の各トークンごとに結果を出力します。

多くの分類モデルは、（シーケンス）分類とトークン分類の両方をサポートしています。トークン分類の詳細については、[このページ](token_classify.md)を参照してください。

分類モデルの num_labels が 1 の場合に限り、そのモデルをスコアリングモデルとして使い、スコアリング API を有効にできます。詳細は[このページ](scoring.md)を参照してください。

## 代表的なユースケース { #typical-use-cases }

### 分類 { #classification }

分類モデルの最も基本的な用途は、入力データをあらかじめ定義されたクラスに分類することです。

## 対応モデル { #supported-models }

### テキストのみのモデル { #text-only-models }

| アーキテクチャ | モデル | HF モデルの例 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `GPT2ForSequenceClassification` | GPT2 | `nie3e/sentiment-polish-gpt2-small` | | |
| `Qwen2ForSequenceClassification`<sup>C</sup> | Qwen2-based | `jason9693/Qwen2.5-1.5B-apeach` | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

### マルチモーダルモデル { #multimodal-models }

!!! note
    マルチモーダルモデルの入力については、[このページ](../supported_models.md#list-of-multimodal-language-models)を参照してください。

| アーキテクチャ | モデル | 入力 | HF モデルの例 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `Qwen2_5_VLForSequenceClassification`<sup>C</sup> | Qwen2_5_VL-based | T + I<sup>E+</sup> + V<sup>E+</sup> | `muziyongshixin/Qwen2.5-VL-7B-for-VideoCls` | | |
| `*ForConditionalGeneration`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | \* | N/A | \* | \* |

<sup>C</sup> `--convert classify` によって自動的に分類モデルへ変換されます。（[詳細](./README.md#model-conversion)）  
\* 機能のサポート状況は元のモデルと同じです。

上記の一覧にモデルがない場合は、[`as_seq_cls_model`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/adapters/#vllm.model_executor.models.adapters.as_seq_cls_model) を使ってモデルの自動変換を試みます。既定では、最後のトークンに対応する隠れ状態を softmax したものからクラス確率が抽出されます。

### クロスエンコーダモデル { #cross-encoder-models }

クロスエンコーダ（リランカーとも呼ばれます）は、2 つのプロンプトを入力として受け取り、num_labels が 1 の出力を返す分類モデルの一種です。ほとんどの分類モデルは[クロスエンコーダモデル](scoring.md#cross-encoder-models)としても使えます。クロスエンコーダモデルの詳細は[このページ](scoring.md)を参照してください。

--8<-- "docs/models/pooling_models/scoring.md:supported-cross-encoder-models"

### 報酬モデル { #reward-models }

（シーケンス）分類モデルを報酬モデルとして使う場合です。詳細は[報酬モデル](reward.md)を参照してください。

--8<-- "docs/models/pooling_models/reward.md:supported-sequence-reward-models"

## オフライン推論 { #offline-inference }

### プーリングパラメータ { #pooling-parameters }

次の[プーリングパラメータ](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.PoolingParams)がサポートされています。

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
# このコードは上流のソースを参照してください: https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/pooling_params.py
```

### `LLM.classify` { #llmclassify }

[`classify`](https://docs.vllm.ai/en/v0.26.0/api/vllm/entrypoints/pooling/offline/#vllm.entrypoints.pooling.offline.PoolingOfflineMixin.classify) メソッドは、各プロンプトに対する確率ベクトルを出力します。

```python
from vllm import LLM

llm = LLM(model="jason9693/Qwen2.5-1.5B-apeach", runner="pooling")
(output,) = llm.classify("Hello, my name is")

probs = output.outputs.probs
print(f"Class Probabilities: {probs!r} (size={len(probs)})")
```

コード例はこちらにあります: [examples/basic/offline_inference/classify.py](../../../examples/basic/offline_inference/classify.py)

### `LLM.encode` { #llmencode }

[`encode`](https://docs.vllm.ai/en/v0.26.0/api/vllm/entrypoints/pooling/offline/#vllm.entrypoints.pooling.offline.PoolingOfflineMixin.encode) メソッドは、vLLM のすべてのプーリングモデルで利用できます。

分類モデルで `LLM.encode` を使う場合は `pooling_task="classify"` を指定します。

```python
from vllm import LLM

llm = LLM(model="jason9693/Qwen2.5-1.5B-apeach", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="classify")

data = output.outputs.data
print(f"Data: {data!r}")
```

## オンラインサービング { #online-serving }

### 分類 API { #classification-api }

オンラインの `/classify` API は `LLM.classify` と同様のものです。

#### completion のパラメータ { #completion-parameters }

分類 API では次のパラメータがサポートされています。

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:completion-params"
    # このコードは上流のソースを参照してください: https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/entrypoints/pooling/base/protocol.py
    ```

次の追加パラメータがサポートされています。

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:completion-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-extra-params"
    ```

#### chat のパラメータ { #chat-parameters }

チャット形式の入力（つまり `messages` を渡す場合）では、次のパラメータがサポートされています。

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:chat-params"
    # このコードは上流のソースを参照してください: https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/entrypoints/pooling/base/protocol.py
    ```

この場合、代わりに次の追加パラメータがサポートされます。

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:chat-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-extra-params"
    ```

#### リクエストの例 { #example-requests }

コード例: [examples/pooling/classify/classification_online.py](../../../examples/pooling/classify/classification_online.py)

文字列の配列を渡すことで、複数のテキストを分類できます。

```bash
curl -v "http://127.0.0.1:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jason9693/Qwen2.5-1.5B-apeach",
    "input": [
      "Loved the new café—coffee was great.",
      "This update broke everything. Frustrating."
    ]
  }'
```

??? console "Response"

    ```json
    {
      "id": "classify-7c87cac407b749a6935d8c7ce2a8fba2",
      "object": "list",
      "created": 1745383065,
      "model": "jason9693/Qwen2.5-1.5B-apeach",
      "data": [
        {
          "index": 0,
          "label": "Default",
          "probs": [
            0.565970778465271,
            0.4340292513370514
          ],
          "num_classes": 2
        },
        {
          "index": 1,
          "label": "Spoiled",
          "probs": [
            0.26448777318000793,
            0.7355121970176697
          ],
          "num_classes": 2
        }
      ],
      "usage": {
        "prompt_tokens": 20,
        "total_tokens": 20,
        "completion_tokens": 0,
        "prompt_tokens_details": null
      }
    }
    ```

`input` フィールドに文字列を直接渡すこともできます。

```bash
curl -v "http://127.0.0.1:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jason9693/Qwen2.5-1.5B-apeach",
    "input": "Loved the new café—coffee was great."
  }'
```

??? console "Response"

    ```json
    {
      "id": "classify-9bf17f2847b046c7b2d5495f4b4f9682",
      "object": "list",
      "created": 1745383213,
      "model": "jason9693/Qwen2.5-1.5B-apeach",
      "data": [
        {
          "index": 0,
          "label": "Default",
          "probs": [
            0.565970778465271,
            0.4340292513370514
          ],
          "num_classes": 2
        }
      ],
      "usage": {
        "prompt_tokens": 10,
        "total_tokens": 10,
        "completion_tokens": 0,
        "prompt_tokens_details": null
      }
    }
    ```

## その他の例 { #more-examples }

その他の例はこちらにあります: [examples/pooling/classify](../../../examples/pooling/classify)

## サポートされる機能 { #supported-features }

### 活性化関数の有効化 / 無効化 { #enabledisable-activation }

活性化関数は `use_activation` で有効・無効を切り替えられます。

### 問題の種類（`multi_label_classification` など） { #problem-type-eg-multi_label_classification }

Hugging Face の設定にある problem_type を通じて `problem_type` を変更できます。サポートされる問題の種類は `single_label_classification`、`multi_label_classification`、`regression` です。

transformers の [ForSequenceClassificationLoss](https://github.com/huggingface/transformers/blob/57bb6db6ee4cfaccc45b8d474dfad5a17811ca60/src/transformers/loss/loss_utils.py#L92) と整合するように実装されています。

### アフィンによるスコア較正 { #affine-score-calibration }

アフィンによるスコア較正は [Platt スケーリング](https://en.wikipedia.org/wiki/Platt_scaling)（Platt, 1999）としても知られ、分類器の出力を適切に較正された確率へ変換する方法として最も広く使われています。

較正は次の変換に従います。

`activation((logit - logit_mean) / logit_sigma)`

| パラメータ | 既定値 | 説明 |
| --------- | ------- | ----------- |
| `logit_mean` | `None` | logits から引く平均（スコアを中心化します） |
| `logit_sigma` | `None` | 平均を引いたあとの logits をスケーリングするための標準偏差 |

計算の順序は次のとおりです。

```python
logits -= logit_mean   # subtract mean (center scores)
logits /= logit_sigma  # divide by sigma (scale)
logits = activation(logits)  # e.g. sigmoid
```

設定例:

```bash
--pooler-config '{"use_activation": true, "logit_mean": 4.5, "logit_sigma": 1.0}'
```

## 削除された機能 { #removed-features }

### PoolingParams からの softmax の削除 { #remove-softmax-from-poolingparams }

`softmax` と `activation` はすでに PoolingParams から削除されています。`classify` と `token_classify` では任意の活性化関数を使えるようにしたため、代わりに `use_activation` を使ってください。
