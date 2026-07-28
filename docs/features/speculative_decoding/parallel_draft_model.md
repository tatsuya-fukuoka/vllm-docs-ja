# 並列ドラフトモデル { #parallel-draft-models }

次のコードは、[PARD](https://arxiv.org/pdf/2504.18583)（並列ドラフトモデル）でドラフトを生成する投機的デコーディングを vLLM で使う設定です。

## PARD のオフラインモードの例 { #pard-offline-mode-example }

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "model": "amd/PARD-Qwen3-0.6B",
        "num_speculative_tokens": 12,
        "method": "draft_model",
        "parallel_drafting": True,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## PARD のオンラインモードの例 { #pard-online-mode-example }

```bash
vllm serve Qwen/Qwen3-4B \
    --host 0.0.0.0 \
    --port 8000 \
    --seed 42 \
    -tp 1 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.8 \
    --speculative-config '{"model": "amd/PARD-Qwen3-0.6B", "num_speculative_tokens": 12, "method": "draft_model", "parallel_drafting": true}'
```

## 学習済みの PARD の重み { #pre-trained-pard-weights }

- [amd/pard](https://huggingface.co/collections/amd/pard)
