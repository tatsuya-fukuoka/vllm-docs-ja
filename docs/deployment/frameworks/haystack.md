# Haystack { #haystack }

[Haystack](https://github.com/deepset-ai/haystack) は、LLM・Transformer モデル・ベクトル検索などを活用したアプリケーションを構築できるエンドツーエンドの LLM フレームワークです。RAG（検索拡張生成）、ドキュメント検索、質問応答、回答生成のいずれについても、最新の埋め込みモデルと LLM をパイプラインとして組み合わせ、エンドツーエンドの NLP アプリケーションを構築できます。

vLLM をバックエンドとする大規模言語モデル (LLM) のサーバーをデプロイし、OpenAI 互換のエンドポイントを利用できます。

## 前提条件 { #prerequisites }

vLLM と Haystack の環境を用意します。

```bash
pip install vllm haystack-ai
```

## デプロイ { #deploy }

1. 対応するチャット補完モデルで vLLM サーバーを起動します。例:

    ```bash
    vllm serve mistralai/Mistral-7B-Instruct-v0.1
    ```

1. Haystack の `OpenAIGenerator` および `OpenAIChatGenerator` コンポーネントから vLLM サーバーに問い合わせます。

??? code

    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.utils import Secret

    generator = OpenAIChatGenerator(
        # for compatibility with the OpenAI API, a placeholder api_key is needed
        api_key=Secret.from_token("VLLM-PLACEHOLDER-API-KEY"),
        model="mistralai/Mistral-7B-Instruct-v0.1",
        api_base_url="http://{your-vLLM-host-ip}:{your-vLLM-host-port}/v1",
        generation_kwargs={"max_tokens": 512},
    )

    response = generator.run(
      messages=[ChatMessage.from_user("Hi. Can you help me plan my next trip to Italy?")]
    )

    print("-"*30)
    print(response)
    print("-"*30)
    ```

```console
------------------------------
{'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text=' Of course! Where in Italy would you like to go and what type of trip are you looking to plan?')], _name=None, _meta={'model': 'mistralai/Mistral-7B-Instruct-v0.1', 'index': 0, 'finish_reason': 'stop', 'usage': {'completion_tokens': 23, 'prompt_tokens': 21, 'total_tokens': 44, 'completion_tokens_details': None, 'prompt_tokens_details': None}})]}
------------------------------
```

詳細は[チュートリアル「Using vLLM in Haystack」](https://github.com/deepset-ai/haystack-integrations/blob/main/integrations/vllm.md)（英語）を参照してください。
