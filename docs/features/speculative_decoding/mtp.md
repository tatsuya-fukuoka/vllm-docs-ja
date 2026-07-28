# MTP（マルチトークン予測） { #mtp-multi-token-prediction }

MTP は、ターゲットモデル自身がマルチトークン予測の機能を備えている投機的デコーディングの手法です。
ドラフトモデルを使う手法と異なり、別途ドラフトモデルを用意する必要がありません。

MTP は次のような場合に有効です。

- 使用するモデルが MTP をネイティブにサポートしている。
- 追加の設定を最小限に抑えて、モデルベースの投機的デコーディングを使いたい。

## Gemma 4 のアシスタントモデル { #gemma-4-assistant-models }

Gemma 4 のアシスタントのチェックポイントは、vLLM の Gemma 4 MTP の経路を使います。
`--speculative-config` の `model` フィールドで渡しますが、汎用のドラフトモデルではありません。

アシスタントのチェックポイントを使って Gemma 4 をサービングする場合は `"method": "mtp"` を指定します。

```bash
vllm serve google/gemma-4-E2B-it \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --speculative-config '{"method":"mtp","model":"gg-hf-am/gemma-4-E2B-it-assistant","num_speculative_tokens":1}'
```

E2B、E4B、12B、26B-A4B、31B の Gemma 4 IT アシスタントのチェックポイントに対応しています。
タワー型のものは `model_type: gemma4_assistant`、エンコーダーを持たない
Gemma 4 Unified（12B）は `model_type: gemma4_unified_assistant` を使います。
vLLM は内部でどちらも `Gemma4MTPModel` に対応付け、アシスタントの層が
ターゲットモデルと KV キャッシュを共有するよう接続します。

古い vLLM で Gemma 4 のアシスタントのチェックポイントに対して
`SpeculativeConfig(method='draft_model', ...)` とログに出る場合、そのバージョンは
アシスタントを汎用のドラフトモデルとして扱っており、マルチモーダルの Gemma 4 を
ターゲットにすると初期化に失敗することがあります。Gemma 4 の MTP に対応した
バージョンへアップグレードしてください。

## オフラインの例 { #offline-example }

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="XiaomiMiMo/MiMo-7B-Base",
    tensor_parallel_size=1,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## オンラインの例 { #online-example }

```bash
vllm serve XiaomiMiMo/MiMo-7B-Base \
    --tensor-parallel-size 1 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## 注意点 { #notes }

- MTP は、vLLM で MTP に対応しているモデルファミリーでのみ動作します。
- `num_speculative_tokens` は投機の深さを制御します。まずは `1` のような
  小さい値から始めるとよいでしょう。
- モデルが MTP に対応していない場合は、EAGLE やドラフトモデルによる投機など
  別の手法を使ってください。
