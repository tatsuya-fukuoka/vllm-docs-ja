# ドラフトモデル { #draft-models }

次のコードは、ドラフトモデルを用いた投機的デコーディングを一度に 5 トークン投機する設定で、vLLM をオフラインモードで構成する例です。

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 5,
        "method": "draft_model",
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

オンラインモードで同等の起動を行うには、サーバー側で次のようにします。

```bash
vllm serve Qwen/Qwen3-4B-Thinking-2507 \
    --host 0.0.0.0 \
    --port 8000 \
    --seed 42 \
    -tp 1 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.8 \
    --speculative-config '{"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 5, "method": "draft_model"}'
```

クライアントとして completions をリクエストするコードは変わりません。

??? code

    ```python
    from openai import OpenAI

    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    # Completion API
    stream = False
    completion = client.completions.create(
        model=model,
        prompt="The future of AI is",
        echo=False,
        n=1,
        stream=stream,
    )

    print("Completion results:")
    if stream:
        for c in completion:
            print(c)
    else:
        print(completion)
    ```

## 語彙が異なる場合のドラフトモデル方式 { #draft-model-method-with-heterogeneous-vocabs }

  vLLM は既定で、ドラフトモデルとターゲットモデルが同じ語彙を共有していることを要求します。`use_heterogeneous_vocab: true` を設定すると **Token-Level Intersection (TLI)** アルゴリズムが有効になり、異なるトークナイザーを持つ別系統のモデルをドラフトモデルとして使えるようになります。
  
  現時点では、`use_heterogeneous_vocab` は `draft_sample_method='greedy'`（既定値）を必要とします。確率的なドラフトサンプリングはまだサポートされておらず、将来のリリースで追加される予定です。

  ```python
  from vllm import LLM, SamplingParams

  llm = LLM(
      model="Qwen/Qwen3-8B",
      speculative_config={                               
          "method": "draft_model",
          "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
          "num_speculative_tokens": 3,
          "use_heterogeneous_vocab": True,
      },
      gpu_memory_utilization=0.5,
  )
outputs = llm.generate(prompts，sampling_params)

for output in outputs:
      prompt = output.prompt
      generated_text = output.outputs[0].text
      print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

!!! warning
    注意: 投機的デコーディングに関する設定はすべて `--speculative-config` で指定してください。
    `--speculative-model` でモデルを指定し、`--num-speculative-tokens` などの関連パラメータを
    個別に追加する従来の方法は非推奨になりました。サポートされるキーと例については
    [`--speculative-config` のスキーマ](README.md#--speculative-config-schema) を参照してください。
