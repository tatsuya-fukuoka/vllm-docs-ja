# AutoGen { #autogen }

[AutoGen](https://github.com/microsoft/autogen) は、自律的に動作したり人と協働したりするマルチエージェントの AI アプリケーションを作るためのフレームワークです。

## 前提条件 { #prerequisites }

vLLM と [AutoGen](https://microsoft.github.io/autogen/0.2/docs/installation/) の環境を用意します。

```bash
pip install vllm

# Install AgentChat and OpenAI client from Extensions
# AutoGen requires Python 3.10 or later.
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

## デプロイ { #deploy }

1. 対応するチャット補完モデルで vLLM サーバーを起動します。例:

    ```bash
    vllm serve mistralai/Mistral-7B-Instruct-v0.2
    ```

1. AutoGen から呼び出します。

??? code

    ```python
    import asyncio
    from autogen_core.models import UserMessage
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_core.models import ModelFamily


    async def main() -> None:
        # Create a model client
        model_client = OpenAIChatCompletionClient(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            base_url="http://{your-vllm-host-ip}:{your-vllm-host-port}/v1",
            api_key="EMPTY",
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "family": ModelFamily.MISTRAL,
                "structured_output": True,
            },
        )

        messages = [UserMessage(content="Write a very short story about a dragon.", source="user")]

        # Create a stream.
        stream = model_client.create_stream(messages=messages)

        # Iterate over the stream and print the responses.
        print("Streamed responses:")
        async for response in stream:
            if isinstance(response, str):
                # A partial response is a string.
                print(response, flush=True, end="")
            else:
                # The last response is a CreateResult object with the complete message.
                print("\n\n------------\n")
                print("The complete response:", flush=True)
                print(response.content, flush=True)

        # Close the client when done.
        await model_client.close()


    asyncio.run(main())
    ```

詳細は次のチュートリアルを参照してください。

- [Using vLLM in AutoGen](https://microsoft.github.io/autogen/0.2/docs/topics/non-openai-models/local-vllm/)（英語）

- [OpenAI 互換 API の例](https://microsoft.github.io/autogen/stable/reference/python/autogen_ext.models.openai.html#autogen_ext.models.openai.OpenAIChatCompletionClient)（英語）
