# FP8 W8A8 { #fp8-w8a8 }

vLLM は、Nvidia H100 や AMD MI300x などの GPU におけるハードウェアアクセラレーションを利用した FP8（8 ビット浮動小数点）の重み・活性値量子化をサポートしています。
現時点で W8A8 が公式にサポートされるのは Hopper と Ada Lovelace の GPU のみです。
Turing / Ampere の GPU では、Marlin カーネルを利用した W8A16（重みのみ FP8）がサポートされます。
FP8 でモデルを量子化すると、精度への影響を最小限に抑えつつ、モデルのメモリ要求量を 1/2 に、スループットを最大 1.6 倍に改善できます。

vLLM ですぐに使える[人気 LLM の FP8 量子化済みチェックポイント](https://huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127)の HF コレクションもご覧ください。

ハードウェアで一般にサポートされる FP8 の型には 2 つの表現があり、それぞれ異なる場面で有用です。

- **E4M3**: 符号ビット 1、指数部 4 ビット、仮数部 3 ビットで構成されます。±448 までの値と `nan` を格納できます。
- **E5M2**: 符号ビット 1、指数部 5 ビット、仮数部 2 ビットで構成されます。±57344 までの値と ± `inf`、`nan` を格納できます。ダイナミックレンジが広がる代わりに、格納される値の精度は下がります。

!!! note
    FP8 の演算は compute capability 8.9 以上（Ada Lovelace、Hopper）の NVIDIA GPU でサポートされます。
    FP8 モデルは compute capability 7.5 以上（Turing）でも、FP8 Marlin を利用した重みのみの W8A16 として動作します。

## インストール { #installation }

vLLM で性能の出る FP8 量子化モデルを作るには、[llm-compressor](https://github.com/vllm-project/llm-compressor/) ライブラリをインストールする必要があります。

```bash
(venv-llm-compressor) pip install llmcompressor
```

さらに、評価のために `vllm` と `lm-evaluation-harness` をインストールします。

```bash
(venv-vllm) pip install vllm "lm-eval[api]>=0.4.12"
```

vLLM と llm-compressor は同時に動作しない場合があるため、それぞれ別の環境を使ってください。

## 量子化の手順 { #quantization-process }

量子化の手順は主に 3 ステップです。

1. モデルの読み込み
2. 量子化の適用
3. vLLM での精度評価

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

### 2. 量子化の適用 { #2-applying-quantization }

FP8 の量子化では、単純な RTN（round-to-nearest）量子化でも精度を回復できます。`FP8_DYNAMIC` スキームですべての `Linear` 層を対象にすることを推奨します。このスキームでは次のようにします。

- 重みには静的なチャネル単位の量子化
- 活性値には動的なトークン単位の量子化

単純な RTN は重みの量子化にデータを必要とせず、活性値は動的に量子化されるため、この量子化フローではキャリブレーションデータが不要です。

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Configure the simple PTQ quantization
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

# Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)

# Save the model: Meta-Llama-3-8B-Instruct-FP8-Dynamic
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

### 3. 精度の評価 { #3-evaluating-accuracy }

`vllm` でモデルを読み込んで実行します。

```python
from vllm import LLM

llm = LLM("./Meta-Llama-3-8B-Instruct-FP8-Dynamic")
result = llm.generate("Hello my name is")
print(result[0].outputs[0].text)
```

`lm_eval` で精度を評価します（例として `gsm8k` の 250 サンプル）。

!!! note
    量子化されたモデルは `bos` トークンの有無に敏感な場合があります。`lm_eval` は既定では `bos`
    トークンを付加しないため、評価を実行するときは必ず `add_bos_token=True` 引数を含めてください。

```bash
MODEL=$PWD/Meta-Llama-3-8B-Instruct-FP8-Dynamic
lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True \
  --tasks gsm8k  --num_fewshot 5 --batch_size auto --limit 250
```

得られるスコアの例は次のとおりです。

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
| --- |------:| -------------- |-----:| --------- | - |----:| - |-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.768|±  |0.0268|
|     |       |strict-match    |     5|exact_match|↑  |0.768|±  |0.0268|
```

## トラブルシューティングとサポート { #troubleshooting-and-support }

問題が発生した場合や機能のリクエストがある場合は、[vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor/issues) の GitHub リポジトリで issue を作成してください。

## オンライン動的量子化 { #online-dynamic-quantization }

元の精度が BF16 / FP16 のモデルを FP8 へ動的に量子化する処理は、キャリブレーションデータなしで vLLM 上で実行できます。コマンドラインで `--quantization="fp8"` を指定するか、LLM のコンストラクタで `quantization="fp8"` を設定すると有効になります。

このモードでは、最後の `lm_head` を除くすべての Linear モジュールの重みが、テンソル単位のスケールで FP8_E4M3 の精度に量子化されます。活性値については、高精度を保つために各 forward パスで最小値と最大値を計算し、動的なテンソル単位のスケールを求めます。そのため、このモードではレイテンシの改善は限定的です。

```python
from vllm import LLM

llm = LLM("facebook/opt-125m", quantization="fp8")
# INFO 06-10 17:55:42 model_runner.py:157] Loading model weights took 0.1550 GB
result = llm.generate("Hello, my name is")
print(result[0].outputs[0].text)
```
