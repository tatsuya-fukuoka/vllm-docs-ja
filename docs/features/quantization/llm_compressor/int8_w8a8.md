# INT8 W8A8 { #int8-w8a8 }

vLLM は、メモリ削減と推論の高速化のために、重みと活性値を INT8 に量子化することをサポートしています。この量子化手法は、良好な性能を保ちながらモデルサイズを削減したい場合に特に有用です。

vLLM ですぐに使える[人気 LLM の INT8 量子化済みチェックポイント](https://huggingface.co/collections/neuralmagic/int8-llms-for-vllm-668ec32c049dca0369816415)の HF コレクションもご覧ください。

!!! note
    INT8 の演算は compute capability 7.5 より上の NVIDIA GPU（Turing、Ampere、Ada Lovelace、Hopper）でサポートされます。

!!! warning
    **Blackwell GPU の制限**: INT8 は compute capability 10.0 以上（RTX 6000 Blackwell など）ではサポートされません。
    代わりに [FP8 量子化](fp8.md)を使うか、Hopper / Ada / Ampere のアーキテクチャで実行してください。

## 前提条件 { #prerequisites }

vLLM で INT8 量子化を使うには、[llm-compressor](https://github.com/vllm-project/llm-compressor/) ライブラリをインストールする必要があります。

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

活性値を INT8 に量子化する場合、活性値のスケールを推定するためのサンプルデータが必要です。実際のデプロイで扱うデータに近いキャリブレーションデータを使うのが理想的です。汎用の instruction チューニング済みモデルであれば、`ultrachat` のようなデータセットを使えます。

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
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# Apply quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save the compressed model: Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token
SAVE_DIR = MODEL_ID.split("/")[1] + "-W8A8-Dynamic-Per-Token"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

この処理により、重みと活性値が 8 ビット整数に量子化された W8A8 モデルが作成されます。

### 4. 精度の評価 { #4-evaluating-accuracy }

量子化後は、vLLM でモデルを読み込んで実行できます。

```python
from vllm import LLM

llm = LLM("./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token")
```

精度を評価するには `lm_eval` を使います。

```bash
lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token",add_bos_token=true \
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
