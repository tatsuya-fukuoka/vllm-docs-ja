# インターリーブド思考 { #interleaved-thinking }

## はじめに { #introduction }

インターリーブド思考（interleaved thinking）を使うと、モデルはツール呼び出しの合間に推論できるようになり、ツールの実行結果を受け取ったあとでより高度な判断を下せます。この機能により、モデルは推論ステップを挟みながら複数のツール呼び出しを連鎖させ、途中の結果にもとづいて細やかな判断を行えます。

重要: インターリーブド思考はトークン使用量と応答レイテンシを増加させます。有効にする際は、予算と性能の要件を考慮してください。

## インターリーブド思考の仕組み { #how-interleaved-thinking-works }

インターリーブド思考では、モデルは次のことができます。

- 次に何をするかを決める前に、ツール呼び出しの結果について推論する
- 推論ステップを挟みながら複数のツール呼び出しを連鎖させる
- 途中の結果にもとづいて、より細やかな判断を下す
- ツールを選んだ理由を透明性のある形で示す

## 対応モデル { #supported-models }

vLLM は現在、次のインターリーブド思考モデルに対応しています。

| モデルシリーズ | 推論パーサー名 |
| ------------ | --------------------- |
| moonshotai/Kimi-K2-Thinking | kimi_k2 |
| MiniMaxAI/MiniMax-M2 | minimax_m2 |

## 使用例 { #example-usage }

ツール呼び出しでインターリーブド思考を使うには、この機能に対応したモデルを指定し、chat completion のリクエストでツール呼び出しを有効にします。次に例を示します。

??? code

    ```python
    """
    vllm serve MiniMaxAI/MiniMax-M2 \
      --tensor-parallel-size 4 \
      --tool-call-parser minimax_m2 \
      --reasoning-parser minimax_m2 \
      --enable-auto-tool-choice
    """
    import json
    
    from openai import OpenAI
    
    client = OpenAI(base_url="http://localhost:8000/v1",     api_key="dummy")
    
    
    def get_current_weather(location: str, unit: "str"):
        """Get the current weather in a given location"""
        if unit == "celsius":
            return f"The current temperature in {location} is 22°C."
        else:
            return f"The current temperature in {location} is 72°F."
    
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given     location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state, e.g.,     'San Francisco, CA'",
                        },
                        "unit": {"type": "string", "enum":     ["celsius", "fahrenheit"]},
                    },
                    "required": ["location", "unit"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What's the weather in Fahrenheit like in San Francisco?"}]
    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    tool_call = response.choices[0].message.tool_calls[0].function
    
    messages.append(
        {
            "role": "assistant",
            "tool_calls": response.choices[0].message.tool_calls,
            "reasoning": response.choices[0].message.reasoning, # append reasoning
        }
    )
    
    # Simulate tool execution
    available_tools = {"get_weather": get_current_weather}
    
    completion_tool_calls = response.choices[0].message.tool_calls
    for call in completion_tool_calls:
        tool_to_call = available_tools[call.function.name]
        args = json.loads(call.function.arguments)
        result = tool_to_call(**args)
        messages.append(
            {
                "role": "tool",
                "content": result,
                "tool_call_id": call.id,
                "name": call.function.name,
            }
        )
    response_2 = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    print(response_2.choices[0].message.content)
    ```
この例では、天気を取得する関数を使って、ツール呼び出しを伴うインターリーブド思考をどう構成するかを示しています。モデルは最終的な応答を生成する前に、ツールの結果について推論します。
