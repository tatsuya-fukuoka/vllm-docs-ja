# BitsAndBytes { #bitsandbytes }

vLLM は、より効率的なモデル推論のために [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) をサポートしています。
BitsAndBytes はモデルを量子化してメモリ使用量を減らし、精度を大きく損なうことなく性能を高めます。
他の量子化手法と比べて、入力データを使った量子化モデルのキャリブレーションが不要な点が特徴です。

vLLM で BitsAndBytes を使う手順は次のとおりです。

```bash
pip install bitsandbytes>=0.49.2
```

vLLM はモデルの設定ファイルを読み取り、読み込み時の量子化と、量子化済みチェックポイントの両方に対応します。

bitsandbytes で量子化されたモデルは [Hugging Face](https://huggingface.co/models?search=bitsandbytes) で探せます。
これらのリポジトリには通常、quantization_config セクションを含む config.json があります。

## 量子化済みチェックポイントを読み込む { #read-quantized-checkpoint }

量子化済みのチェックポイントでは、vLLM が設定ファイルから量子化方式を推定しようとするため、quantization 引数を明示的に指定する必要はありません。

```python
from vllm import LLM
import torch
# unsloth/tinyllama-bnb-4bit is a pre-quantized checkpoint.
model_id = "unsloth/tinyllama-bnb-4bit"
llm = LLM(
    model=model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
```

## 読み込み時の量子化: 4bit で読み込む { #inflight-quantization-load-as-4bit-quantization }

BitsAndBytes で読み込み時に 4bit 量子化するには、quantization 引数を明示的に指定する必要があります。

```python
from vllm import LLM
import torch
model_id = "huggyllama/llama-7b"
llm = LLM(
    model=model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization="bitsandbytes",
)
```

## OpenAI 互換サーバー { #openai-compatible-server }

読み込み時の 4bit 量子化を行うには、モデルの引数に次を追加します。

```bash
--quantization bitsandbytes
```
