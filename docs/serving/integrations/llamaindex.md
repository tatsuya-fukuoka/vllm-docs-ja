# LlamaIndex { #llamaindex }

vLLM は [LlamaIndex](https://github.com/run-llama/llama_index) からも利用できます。

LlamaIndex をインストールするには次を実行します。

```bash
pip install llama-index-llms-vllm -q
```

単一または複数の GPU で推論するには、`llamaindex` の `Vllm` クラスを使います。

```python
from llama_index.llms.vllm import Vllm

llm = Vllm(
    model="microsoft/Orca-2-7b",
    tensor_parallel_size=4,
    max_new_tokens=100,
    vllm_kwargs={"gpu_memory_utilization": 0.5},
)
```

詳細は[チュートリアル](https://docs.llamaindex.ai/en/latest/examples/llm/vllm/)（英語）を参照してください。
