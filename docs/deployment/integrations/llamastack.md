# Llama Stack { #llama-stack }

vLLM は [Llama Stack](https://github.com/llamastack/llama-stack) からも利用できます。

Llama Stack をインストールするには次を実行します。

```bash
pip install llama-stack -q
```

## OpenAI 互換 API を使った推論 { #inference-using-openai-compatible-api }

Llama Stack のサーバーを起動し、次の設定で vLLM サーバーを指すようにします。

```yaml
inference:
  - provider_id: vllm0
    provider_type: remote::vllm
    config:
      url: http://127.0.0.1:8000
```

このリモート vLLM プロバイダの詳細は[こちらのガイド](https://llama-stack.readthedocs.io/en/latest/providers/inference/remote_vllm.html)（英語）を参照してください。

## 組み込み vLLM を使った推論 { #inference-using-embedded-vllm }

[インラインプロバイダ](https://github.com/llamastack/llama-stack/tree/main/llama_stack/providers/inline/inference)も利用できます。この方法を使う設定例は次のとおりです。

```yaml
inference:
  - provider_type: vllm
    config:
      model: Llama3.1-8B-Instruct
      tensor_parallel_size: 4
```
