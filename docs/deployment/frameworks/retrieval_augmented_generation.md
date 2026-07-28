# 検索拡張生成 (RAG) { #retrieval-augmented-generation }

[検索拡張生成 (RAG)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) は、生成 AI のモデルが新しい情報を検索して取り込めるようにする手法です。大規模言語モデル (LLM) とのやり取りを変更し、指定した文書群を参照しながらユーザーのクエリに応答させることで、学習済みの知識を補完します。これにより、LLM はドメイン固有の情報や最新の情報を利用できます。社内データにアクセスできるチャットボットの提供や、信頼できる情報源にもとづく回答の生成などが典型的なユースケースです。

利用できる組み合わせは次のとおりです。

- vLLM + [langchain](https://github.com/langchain-ai/langchain) + [milvus](https://github.com/milvus-io/milvus)
- vLLM + [llamaindex](https://github.com/run-llama/llama_index) + [milvus](https://github.com/milvus-io/milvus)

## vLLM + langchain { #vllm-langchain }

### 前提条件 { #prerequisites }

vLLM と langchain の環境を用意します。

```bash
pip install -U vllm \
            langchain_milvus langchain_openai \
            langchain_community beautifulsoup4 \
            langchain-text-splitters
```

### デプロイ { #deploy }

1. 対応する埋め込みモデルで vLLM サーバーを起動します。例:

    ```bash
    # Start embedding service (port 8000)
    vllm serve ssmits/Qwen2-7B-Instruct-embed-base
    ```

1. 対応するチャット補完モデルで vLLM サーバーを起動します。例:

    ```bash
    # Start chat service (port 8001)
    vllm serve qwen/Qwen1.5-0.5B-Chat --port 8001
    ```

1. 次のスクリプトを使います: [examples/applications/rag/retrieval_augmented_generation_with_langchain.py](../../../examples/applications/rag/retrieval_augmented_generation_with_langchain.py)

1. スクリプトを実行します。

    ```bash
    python retrieval_augmented_generation_with_langchain.py
    ```

## vLLM + llamaindex { #vllm-llamaindex }

### 前提条件 { #prerequisites_1 }

vLLM と llamaindex の環境を用意します。

```bash
pip install vllm \
            llama-index llama-index-readers-web \
            llama-index-llms-openai-like    \
            llama-index-embeddings-openai-like \
            llama-index-vector-stores-milvus \
```

### デプロイ { #deploy_1 }

1. 対応する埋め込みモデルで vLLM サーバーを起動します。例:

    ```bash
    # Start embedding service (port 8000)
    vllm serve ssmits/Qwen2-7B-Instruct-embed-base
    ```

1. 対応するチャット補完モデルで vLLM サーバーを起動します。例:

    ```bash
    # Start chat service (port 8001)
    vllm serve qwen/Qwen1.5-0.5B-Chat --port 8001
    ```

1. 次のスクリプトを使います: [examples/applications/rag/retrieval_augmented_generation_with_llamaindex.py](../../../examples/applications/rag/retrieval_augmented_generation_with_llamaindex.py)

1. スクリプトを実行します。

    ```bash
    python retrieval_augmented_generation_with_llamaindex.py
    ```
