# INT8 W4A8 { #int8-w4a8 }

vLLM は、メモリ削減と推論の高速化のために、重みを INT4 に、活性値を INT8 に量子化することをサポートしています。この量子化手法は、良好な性能を保ちながらモデルサイズを削減したい場合に特に有用です。

## 前提条件 { #prerequisites }

vLLM で INT8 W4A8 量子化を使うには、[llm-compressor](https://github.com/vllm-project/llm-compressor/) ライブラリをインストールする必要があります。

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
    dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2. キャリブレーションデータの準備 { #2-preparing-calibration-data }

活性値を INT8 に、重みを INT4 に量子化する場合、活性値のスケールを推定するためのサンプルデータが必要です。実際のデプロイで扱うデータに近いキャリブレーションデータを使うのが理想的です。汎用の instruction チューニング済みモデルであれば、`ultrachat` のようなデータセットを使えます。

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

以下のレシピは W4A8 モデル（int4 の重み、int8 の活性値）を作成します。Arm® の CPU では、[KleidiAI](https://github.com/ARM-software/kleidiai) によって高速化されます。

精度を重視する場合は groupwise を、推論性能を重視する場合は channelwise を使ってください。

=== "Groupwise"

    ```python
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    # Configure the quantization algorithms
    recipe = [
        GPTQModifier(
            targets="Linear",
            scheme="W4A8",
            ignore=["lm_head"],
            dampening_frac=0.01
        ),
    ]

    # Apply quantization
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # Save the compressed model: Meta-Llama-3-8B-Instruct-W4A8-G128-Dynamic-Per-Token
    SAVE_DIR = MODEL_ID.split("/")[1] + "-W4A8-G128-Dynamic-Per-Token"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    ```

=== "Channelwise"

    ```python
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from compressed_tensors.quantization import QuantizationStrategy, QuantizationType

    scheme = {
        "targets": ["Linear"],
        "weights": {
            "num_bits": 4,
            "type": QuantizationType.INT,
            "strategy": QuantizationStrategy.CHANNEL,
            "symmetric": True,
            "dynamic": False,
            "group_size": None,
        },
        "input_activations": {
            "num_bits": 8,
            "type": QuantizationType.INT,
            "strategy": QuantizationStrategy.TOKEN,
            "dynamic": True,
            "symmetric": False,
            "observer": None,
        },
        "output_activations": None,
    }

    recipe = [
        GPTQModifier(
            targets="Linear",
            config_groups={"group_0": scheme},
            ignore=["lm_head"],
            dampening_frac=0.01,
        ),
    ]

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # Save the compressed model: Meta-Llama-3-8B-Instruct-W4A8-Channelwise-Dynamic-Per-Token
    SAVE_DIR = MODEL_ID.split("/")[1] + "-W4A8-Channelwise-Dynamic-Per-Token"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    ```

### 4. 精度の評価 { #4-evaluating-accuracy }

=== "Groupwise"

    量子化後は、vLLM でモデルを読み込んで実行できます。

    ```python
    from vllm import LLM

    llm = LLM("./Meta-Llama-3-8B-Instruct-W4A8-G128-Dynamic-Per-Token")
    ```

    精度を評価するには `lm_eval` を使います。

    ```bash
    lm_eval --model vllm \
        --model_args pretrained="./Meta-Llama-3-8B-Instruct-W4A8-G128-Dynamic-Per-Token",add_bos_token=true \
        --tasks gsm8k \
        --num_fewshot 5 \
        --limit 250 \
        --batch_size 'auto'
    ```

=== "Channelwise"

    量子化後は、vLLM でモデルを読み込んで実行できます。

    ```python
    from vllm import LLM

    llm = LLM("./Meta-Llama-3-8B-Instruct-W4A8-Channelwise-Dynamic-Per-Token")
    ```

    精度を評価するには `lm_eval` を使います。

    ```bash
    lm_eval --model vllm \
        --model_args pretrained="./Meta-Llama-3-8B-Instruct-W4A8-Channelwise-Dynamic-Per-Token",add_bos_token=true \
        --tasks gsm8k \
        --num_fewshot 5 \
        --limit 250 \
        --batch_size 'auto'
    ```

!!! note
    量子化されたモデルは `bos` トークンの有無に敏感な場合があります。評価を実行するときは必ず
    `add_bos_token=True` 引数を含めてください。

## ベストプラクティス { #best-practices }

- キャリブレーションデータはまず 512 サンプルから始める（精度が落ちる場合は増やす）
- シーケンス長は 2048 を出発点にする
- モデルの学習に使われたチャットテンプレートまたは instruction テンプレートを使う
- モデルをファインチューニングしている場合は、学習データの一部をキャリブレーションに使うことを検討する

## トラブルシューティングとサポート { #troubleshooting-and-support }

問題が発生した場合や機能のリクエストがある場合は、[vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor/issues) の GitHub リポジトリで issue を作成してください。
