# INT4 W4A16 { #int4-w4a16 }

vLLM は、メモリ削減と推論の高速化のために、重みを INT4 に量子化することをサポートしています。この量子化手法は、モデルサイズを削減しつつ、QPS（1 秒あたりのクエリ数）が低いワークロードで低レイテンシを保ちたい場合に特に有用です。

vLLM ですぐに使える[人気 LLM の INT4 量子化済みチェックポイント](https://huggingface.co/collections/neuralmagic/int4-llms-for-vllm-668ec34bf3c9fa45f857df2c)の HF コレクションもご覧ください。

!!! note
    INT4 の演算は compute capability 8.0 より上の NVIDIA GPU（Ampere、Ada Lovelace、Hopper、Blackwell）でサポートされます。

## 前提条件 { #prerequisites }

vLLM で INT4 量子化を使うには、[llm-compressor](https://github.com/vllm-project/llm-compressor/) ライブラリをインストールする必要があります。

```bash
(venv-llm-compressor) pip install llmcompressor
```

さらに、評価のために `vllm` と `lm-evaluation-harness` をインストールします。

```bash
(venv-vllm) pip install vllm "lm-eval[api]>=0.4.12"
```

vLLM と llm-compressor は同時に動作しない場合があるため、それぞれ別の環境を使ってください。

## 量子化の手順 { #quantization-process }

量子化の手順は主に 4 ステップです。

1. モデルの読み込み
2. キャリブレーションデータの準備
3. 量子化の適用
4. vLLM での精度評価

### 1. モデルの読み込み { #1-loading-the-model }

標準の `transformers` の AutoModel クラスを使って、モデルとトークナイザーを読み込みます。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2. キャリブレーションデータの準備 { #2-preparing-calibration-data }

重みを INT4 に量子化する場合、重みの更新量とキャリブレーション済みスケールを推定するためのサンプルデータが必要です。実際のデプロイで扱うデータに近いキャリブレーションデータを使うのが理想的です。汎用の instruction チューニング済みモデルであれば、`ultrachat` のようなデータセットを使えます。

```python
from datasets import load_dataset

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load and preprocess the dataset
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)
```

### 3. 量子化の適用 { #3-applying-quantization }

次に、量子化アルゴリズムを適用します。

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

# Configure the quantization algorithms
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

# Apply quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save the compressed model: Meta-Llama-3-8B-Instruct-W4A16-G128
SAVE_DIR = MODEL_ID.split("/")[1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

この処理により、重みが 4 ビット整数に量子化された W4A16 モデルが作成されます。

### 4. 精度の評価 { #4-evaluating-accuracy }

量子化後は、vLLM でモデルを読み込んで実行できます。

```python
from vllm import LLM

llm = LLM("./Meta-Llama-3-8B-Instruct-W4A16-G128")
```

精度を評価するには `lm_eval` を使います。

```bash
lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-W4A16-G128",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'
```

!!! note
    量子化されたモデルは `bos` トークンの有無に敏感な場合があります。評価を実行するときは必ず
    `add_bos_token=True` 引数を含めてください。

## ベストプラクティス { #best-practices }

- キャリブレーションデータはまず 512 サンプルから始め、精度が落ちる場合は増やす
- 特定のユースケースへの過剰適合を防ぐため、キャリブレーションデータには多様なサンプルを含める
- シーケンス長は 2048 を出発点にする
- モデルの学習に使われたチャットテンプレートまたは instruction テンプレートを使う
- モデルをファインチューニングしている場合は、学習データの一部をキャリブレーションに使うことを検討する
- 量子化アルゴリズムの主要なハイパーパラメータを調整する:
    - `dampening_frac` は GPTQ アルゴリズムの影響度を決めます。値を小さくすると精度が向上することがありますが、数値的な不安定さを招いてアルゴリズムが失敗する場合があります。
    - `actorder` は活性値の順序付けを設定します。層の重みを圧縮する際、チャネルを量子化する順序が結果に影響します。`actorder="weight"` を設定すると、レイテンシを増やさずに精度を改善できます。

次は、自分のユースケースに合わせて調整できる、より詳細な量子化レシピの例です。

```python
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
recipe = GPTQModifier(
    targets="Linear",
    config_groups={
        "config_group": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                group_size=128,
                symmetric=True,
                dynamic=False,
                actorder="weight",
            ),
        ),
    },
    ignore=["lm_head"],
    update_size=NUM_CALIBRATION_SAMPLES,
    dampening_frac=0.01,
)
```

## トラブルシューティングとサポート { #troubleshooting-and-support }

問題が発生した場合や機能のリクエストがある場合は、[vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor/issues) の GitHub リポジトリで issue を作成してください。 `llm-compressor` における INT4 量子化の完全な例は[こちら](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a16/llama3_example.py)にあります。
