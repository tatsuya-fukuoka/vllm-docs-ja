# MLP ドラフトモデル { #mlp-draft-models }

次のコードは、文脈ベクトルとサンプリング済みトークンの両方を条件としてドラフトを予測するドラフトモデルによる投機的デコーディングを、vLLM で使う設定です。詳細は [The Hitchhiker's Guide to Speculative Decoding](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/) と [IBM Research の技術レポート](https://arxiv.org/abs/2404.19124)（いずれも英語）を参照してください。

## MLP のドラフタの例 { #mlp-drafter-example }

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    speculative_config={
        "model": "ibm-ai-platform/llama3-8b-accelerator",
        "draft_tensor_parallel_size": 1,
        "method": "mlp_speculator",
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

!!! warning "既知の問題"
    `ibm-ai-platform/llama3-70b-accelerator` は次のエラーで失敗することがあります。
    `AttributeError: 'MLPSpeculatorConfig' object has no attribute 'num_attention_heads'`.
    状況は [#34106](https://github.com/vllm-project/vllm/issues/34106) と
    [#34163](https://github.com/vllm-project/vllm/pull/34163) で追跡できます。

## 学習済みの MLP ドラフタモデル { #pre-trained-mlp-drafter-models }

この種の投機用モデルは、Hugging Face Hub にさまざまなものが公開されています。

- [llama-13b-accelerator](https://huggingface.co/ibm-ai-platform/llama-13b-accelerator)
- [llama3-8b-accelerator](https://huggingface.co/ibm-ai-platform/llama3-8b-accelerator)
- [codellama-34b-accelerator](https://huggingface.co/ibm-ai-platform/codellama-34b-accelerator)
- [llama2-70b-accelerator](https://huggingface.co/ibm-ai-platform/llama2-70b-accelerator)
- [llama3-70b-accelerator](https://huggingface.co/ibm-ai-platform/llama3-70b-accelerator)
- [granite-3b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-3b-code-instruct-accelerator)
- [granite-8b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-8b-code-instruct-accelerator)
- [granite-7b-instruct-accelerator](https://huggingface.co/ibm-granite/granite-7b-instruct-accelerator)
- [granite-20b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-20b-code-instruct-accelerator)
