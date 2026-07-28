# LangChain { #langchain }

vLLM は [LangChain](https://github.com/langchain-ai/langchain) からも利用できます。

LangChain をインストールするには次を実行します。

```bash
pip install langchain langchain_community -q
```

単一または複数の GPU で推論するには、`langchain` の `VLLM` クラスを使います。

??? code

    ```python
    from langchain_community.llms import VLLM

    llm = VLLM(
        model="Qwen/Qwen3-4B",
        trust_remote_code=True,  # mandatory for hf models
        max_new_tokens=128,
        top_k=10,
        top_p=0.95,
        temperature=0.8,
        # for distributed inference
        # tensor_parallel_size=...,
    )

    print(llm("What is the capital of France ?"))
    ```

詳細は[チュートリアル](https://python.langchain.com/docs/integrations/llms/vllm)（英語）を参照してください。
