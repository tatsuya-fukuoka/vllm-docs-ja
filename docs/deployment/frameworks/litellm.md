# LiteLLM { #litellm }

[LiteLLM](https://github.com/BerriAI/litellm) は、あらゆる LLM の API を OpenAI の形式で呼び出せるようにします（Bedrock、Huggingface、VertexAI、TogetherAI、Azure、OpenAI、Groq など）。

LiteLLM が担うこと:

- 入力を各プロバイダの `completion`・`embedding`・`image_generation` エンドポイント向けに変換する
- [一貫した出力](https://docs.litellm.ai/docs/completion/output) — テキストの応答は常に `['choices'][0]['message']['content']` から取得できる
- 複数のデプロイ（Azure / OpenAI など）にまたがるリトライ・フォールバックのロジック - [Router](https://docs.litellm.ai/docs/routing)
- プロジェクト・API キー・モデルごとの予算とレート制限の設定 - [LiteLLM Proxy Server (LLM Gateway)](https://docs.litellm.ai/docs/simple_proxy)

LiteLLM は vLLM 上のすべてのモデルに対応しています。

## 前提条件 { #prerequisites }

vLLM と litellm の環境を用意します。

```bash
pip install vllm litellm
```

## デプロイ { #deploy }

### チャット補完 { #chat-completion }

1. 対応するチャット補完モデルで vLLM サーバーを起動します。例:

    ```bash
    vllm serve qwen/Qwen1.5-0.5B-Chat
    ```

1. litellm から呼び出します。

??? code

    ```python
    import litellm 

    messages = [{"content": "Hello, how are you?", "role": "user"}]

    # hosted_vllm is prefix key word and necessary
    response = litellm.completion(
        model="hosted_vllm/qwen/Qwen1.5-0.5B-Chat", # pass the vllm model name
        messages=messages,
        api_base="http://{your-vllm-server-host}:{your-vllm-server-port}/v1",
        temperature=0.2,
        max_tokens=80,
    )

    print(response)
    ```

### 埋め込み { #embeddings }

1. 対応する埋め込みモデルで vLLM サーバーを起動します。例:

    ```bash
    vllm serve BAAI/bge-base-en-v1.5
    ```

1. litellm から呼び出します。

```python
from litellm import embedding   
import os

os.environ["HOSTED_VLLM_API_BASE"] = "http://{your-vllm-server-host}:{your-vllm-server-port}/v1"

# hosted_vllm is prefix key word and necessary
# pass the vllm model name
embedding = embedding(model="hosted_vllm/BAAI/bge-base-en-v1.5", input=["Hello world"])

print(embedding)
```

詳細は[チュートリアル「Using vLLM in LiteLLM」](https://docs.litellm.ai/docs/providers/vllm)（英語）を参照してください。
