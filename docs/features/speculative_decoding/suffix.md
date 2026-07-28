# Suffix デコーディング { #suffix-decoding }

次のコードは、Suffix デコーディング（[技術レポート](https://arxiv.org/abs/2411.04975)）でドラフトを生成する投機的デコーディングを vLLM で使う設定です。

n-gram と同様に、Suffix デコーディングは直近の `n` 個の生成トークンによるパターンマッチでドラフトトークンを生成できます。n-gram と異なる点は、(1) プロンプトと過去の生成の両方に対してパターンマッチできる、(2) 出現頻度をもとにもっとも可能性の高い続きを提案する、(3) 受理率を高めるため、リクエストごと・反復ごとに投機するトークン数を適応的に変える、という 3 点です。

Suffix デコーディングは、コード編集、エージェントのループ（自己反省や自己整合性など）、RL のロールアウトなど、繰り返しの多いタスクで高い性能を発揮します。

!!! tip "Arctic Inference のインストール"
    Suffix デコーディングには [Arctic Inference](https://github.com/snowflakedb/ArcticInference) が必要です。`pip install arctic-inference` でインストールできます。

!!! tip "Suffix デコーディングの投機トークン数"
    Suffix デコーディングは、デコードのステップごと・リクエストごとに投機するトークン数を動的に変えるため、`num_speculative_tokens` は投機トークン数の*上限*を指定します。`16` や `32`（既定）のような大きめの値を推奨します。

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 32,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
