# コンテキスト長の拡張 { #context-extension }

!!! note
    旧バージョンの vLLM で使われていた `--rope-scaling` パラメータはサポートされなくなりました。代わりに `--hf-overrides` で `rope_parameters` を指定してください。
ここでは、vLLM でモデルのコンテキスト長を拡張する例を紹介します。

## オフライン推論の例 { #offline-inference-example }

[`context_extension.py`](../../examples/features/context_extension/context_extension_offline.py) のスクリプトは、YARN の手法（rope_parameters）で Qwen モデルのコンテキスト長を拡張し、簡単なチャットを実行する例です。

### 使い方 { #usage }

```bash
python examples/features/context_extension/context_extension_offline.py
```

## OpenAI 互換のオンライン API を使う方法 { #openai-online-method }

vLLM の OpenAI 互換 API を使って、コンテキスト長を拡張したモデルをサービングすることもできます。

### 使い方 { #usage_1 }

YARN でコンテキスト長を拡張するには、次のコマンドで vLLM サーバーを起動します。

```bash
vllm serve Qwen/Qwen3-0.6B \
  --hf-overrides '{"rope_parameters": {"factor": 4.0, "original_max_position_embeddings": 32768, "rope_theta": 1000000, "rope_type": "yarn"}}' \
  --max-model-len 131072
```

### クライアントの例 { #client-example }

サーバーを起動したら、OpenAI の Python クライアントからやり取りできます。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"  # Dummy API key, required by the client
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"}
    ],
    max_tokens=128,
    temperature=0.8,
    top_p=0.95
)

print(response.choices[0].message.content)
```

### 主なパラメータ { #key-parameters }

指定できるパラメータは選んだ `rope_type` によって異なります。サポートされる RoPE の種類とそれぞれのパラメータの詳細は、[Hugging Face Transformers の RoPE のドキュメント](https://huggingface.co/docs/transformers/main/en/internal/rope_utils#transformers.RopeParameters)（英語）を参照してください。

よく使うパラメータ:

- `rope_type`: RoPE の実装の種類（`"yarn"`、`"linear"`、`"dynamic"` など）
- `factor`: コンテキスト長を何倍に拡張するか
- `original_max_position_embeddings`: モデル本来の最大位置埋め込み数

次のパラメータは vLLM 固有のものです。

- `max_model_len`: 拡張後の最大系列長（元の長さ × factor）。
  KV キャッシュの事前確保と、サービング時のリクエストの上限に使われます。
