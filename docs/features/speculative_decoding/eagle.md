# EAGLE ドラフトモデル { #eagle-draft-models }

次のコードは、[EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency)](https://arxiv.org/pdf/2401.15077) にもとづくドラフトモデルでドラフトを生成する投機的デコーディングを vLLM で使う設定です。リクエスト単位の受理率の取得方法を含む、オフラインモードのより詳しい例は [examples/features/speculative_decoding/spec_decode_offline.py](../../../examples/features/speculative_decoding/spec_decode_offline.py) にあります。

## Eagle のドラフタの例 { #eagle-drafter-example }

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 2,
        "method": "eagle",
    },
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Eagle3 のドラフタの例 { #eagle3-drafter-example }

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=2,
    speculative_config={
        "model": "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
        "draft_tensor_parallel_size": 2,
        "num_speculative_tokens": 2,
        "method": "eagle3",
    },
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## 学習済みの Eagle ドラフトモデル { #pre-trained-eagle-draft-models }

Hugging Face Hub には、さまざまな EAGLE のドラフトモデルが公開されています。

* [RedHatAI/speculator-models](https://huggingface.co/collections/RedHatAI/speculator-models)
* [yuhuili/models](https://huggingface.co/yuhuili/models?search=eagle)

!!! warning
    `vllm<0.7.0` を使っている場合は、[このスクリプト](https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d)で投機用モデルを変換し、`speculative_config` に `"model": "path/to/modified/eagle/model"` を指定してください。
