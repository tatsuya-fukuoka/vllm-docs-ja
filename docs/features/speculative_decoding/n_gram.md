# N-gram による投機 { #n-gram-speculation }

次のコードは、プロンプト中の n-gram のマッチングでドラフトを生成する投機的デコーディングを
vLLM で使う設定です。詳細は[こちらのスレッド](https://x.com/joao_gante/status/1747322413006643259)（英語）を参照してください。

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
