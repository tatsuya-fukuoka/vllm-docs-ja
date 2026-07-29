# 推論（reasoning）の出力 { #reasoning-outputs }

vLLM は [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) のような推論モデルをサポートしています。これらのモデルは、推論の過程と最終的な結論の両方を含む出力を生成するよう設計されています。

推論モデルは、出力に追加の `reasoning` フィールドを返します。ここには最終的な結論に至るまでの推論の過程が含まれます。このフィールドは、それ以外のモデルの出力には含まれません。

!!! warning
    `reasoning` は以前 `reasoning_content` と呼ばれていました。移行するには、`reasoning_content` を `reasoning` にそのまま置き換えてください。

## サポートされるモデル { #supported-models }

vLLM は現在、次の推論モデルをサポートしています。

| モデルシリーズ | パーサー名 | 構造化出力のサポート | ツール呼び出し |
| ------------ | ----------- | ---------------- | ----------- |
| [Cohere Command A Reasoning](https://huggingface.co/CohereLabs/command-a-reasoning-08-2025) | `cohere_command3` | `json`, `regex` | ✅ |
| [DeepSeek R1 series](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) | `deepseek_r1` | `json`, `regex` | ❌ |
| [Gemma 4 series](https://huggingface.co/google/gemma-4-26B-A4B-it) | `gemma4` | `json`, `regex` | ✅ |
| [DeepSeek-V3.1](https://huggingface.co/collections/deepseek-ai/deepseek-v31-68a491bed32bd77e7fca048f) | `deepseek_v3` | `json`, `regex` | ❌ |
| [ERNIE-4.5-VL series](https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-PT) | `ernie45` | `json`, `regex` | ❌ |
| [ERNIE-4.5-21B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking) | `ernie45` | `json`, `regex` | ✅ |
| [GLM-4.5 series](https://huggingface.co/collections/zai-org/glm-45-687c621d34bda8c9e4bf503b) | `glm45` | `json`, `regex` | ✅ |
| [Holo2 series](https://huggingface.co/collections/Hcompany/holo2) | `holo2` | `json`, `regex` | ✅ |
| [Hunyuan A13B series](https://huggingface.co/collections/tencent/hunyuan-a13b-685ec38e5b46321e3ea7c4be) | `hunyuan_a13b` | `json`, `regex` | ✅ |
| [IBM Granite 3.2 language models](https://huggingface.co/collections/ibm-granite/granite-32-language-models-67b3bc8c13508f6d064cff9a) | `granite` | ❌ | ❌ |
| [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) | `minimax_m2_append_think` | `json`, `regex` | ✅ |
| [Qwen3 series](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) | `qwen3` | `json`, `regex` | ✅ |
| [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) | `deepseek_r1` | `json`, `regex` | ✅ |

!!! note
    IBM Granite 3.2 と DeepSeek-V3.1 の推論は既定で無効です。有効にするには `chat_template_kwargs` に `thinking=True` を渡す必要があります。
    Qwen3 シリーズの推論機能は既定で有効です。無効にするには `chat_template_kwargs` に `enable_thinking=False` を渡す必要があります。
    Gemma 4 の推論は既定で無効です。有効にするには `chat_template_kwargs` に `enable_thinking=True` を渡すか、`reasoning_effort` を設定してください（自動的に有効になります）。
    DeepSeek-V3.1 のツール呼び出しは非 thinking モードでサポートされます。
    Holo2 の推論は既定で有効です。無効にするには `chat_template_kwargs` に `thinking=False` を渡す必要があります。

## クイックスタート { #quickstart }

推論モデルを使うには、chat completion エンドポイントへのリクエスト時に `--reasoning-parser` フラグを指定する必要があります。`--reasoning-parser` フラグは、モデル出力から推論内容を抽出するために使う推論パーサーを指定します。

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --reasoning-parser deepseek_r1
```

次に、レスポンスに推論内容が含まれるはずのリクエストをモデルに送ります。

??? code

    ```python
    from openai import OpenAI

    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    # Round 1
    messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
    # For granite, add: `extra_body={"chat_template_kwargs": {"thinking": True}}`
    # For Qwen3 series, if you want to disable thinking in reasoning mode, add:
    # extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    response = client.chat.completions.create(model=model, messages=messages)

    reasoning = response.choices[0].message.reasoning
    content = response.choices[0].message.content

    print("reasoning:", reasoning)
    print("content:", content)
    ```

`reasoning` フィールドには最終的な結論に至るまでの推論の過程が、`content` フィールドには最終的な結論が含まれます。

## ストリーミングの chat completions { #streaming-chat-completions }

推論モデルではストリーミングの chat completions もサポートされています。`reasoning` フィールドは、[chat completion のレスポンスチャンク](https://platform.openai.com/docs/api-reference/chat/streaming)の `delta` フィールドで利用できます。

??? console "Json"

    ```json
    {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1694268190,
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "reasoning": "is",
                },
                "logprobs": null,
                "finish_reason": null
            }
        ]
    }
    ```

OpenAI の Python クライアントライブラリは、ストリーミング出力の `reasoning` 属性を公式にはサポートしていません。ただし、レスポンスの追加属性はサポートされています。`hasattr` を使って、レスポンスに `reasoning` 属性があるかどうかを確認できます。例:

??? code

    ```python
    from openai import OpenAI

    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
    # For granite, add: `extra_body={"chat_template_kwargs": {"thinking": True}}`
    # For Qwen3 series, if you want to disable thinking in reasoning mode, add:
    # extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    print("client: Start streaming chat completions...")
    printed_reasoning = False
    printed_content = False

    for chunk in stream:
        # Safely extract reasoning and content from delta,
        # defaulting to None if attributes don't exist or are empty strings
        reasoning = (
            getattr(chunk.choices[0].delta, "reasoning", None) or None
        )
        content = getattr(chunk.choices[0].delta, "content", None) or None

        if reasoning is not None:
            if not printed_reasoning:
                printed_reasoning = True
                print("reasoning:", end="", flush=True)
            print(reasoning, end="", flush=True)
        elif content is not None:
            if not printed_content:
                printed_content = True
                print("\ncontent:", end="", flush=True)
            # Extract and print the content
            print(content, end="", flush=True)
    ```

アクセスする前に、レスポンスに `reasoning` が存在するかを確認することを忘れないでください。[例](https://github.com/vllm-project/vllm/blob/main/examples/reasoning/openai_chat_completion_with_reasoning_streaming.py)も参照してください。

## ツール呼び出し { #tool-calling }

ツール呼び出しと推論パーサーの両方が有効な場合も、推論内容を利用できます。なお、ツール呼び出しが関数を解析するのは `content` フィールドからのみで、`reasoning` からは解析しません。

??? code

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location", "unit"],
                }
            },
        }
    ]

    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
        tools=tools,
        tool_choice="auto",
    )

    print(response)
    tool_call = response.choices[0].message.tool_calls[0].function

    print(f"reasoning: {response.choices[0].message.reasoning}")
    print(f"Function called: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")
    ```

その他の例は [examples/reasoning/openai_chat_completion_tool_calls_with_reasoning.py](../../examples/reasoning/openai_chat_completion_tool_calls_with_reasoning.py) を参照してください。

## サーバーレベルの既定 chat template kwargs { #server-level-default-chat-template-kwargs }

`--default-chat-template-kwargs` の CLI 引数で、サーバーレベルの既定の `chat_template_kwargs` を設定できます。クライアントがリクエストごとに指定しなくても、すべてのリクエストに対して推論の挙動を設定できるため便利です。

### 既定で thinking モードを無効にする { #disabling-thinking-mode-by-default }

Qwen3 のように thinking が既定で有効なモデルでは、サーバー全体で無効にできます。

```bash
vllm serve Qwen/Qwen3-8B \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

### 既定で thinking モードを有効にする { #enabling-thinking-mode-by-default }

IBM Granite 3.2 や DeepSeek-V3.1 のように thinking が既定で無効なモデルでは、サーバー全体で有効にできます。

```bash
vllm serve ibm-granite/granite-3.2-2b-instruct \
    --reasoning-parser granite \
    --default-chat-template-kwargs '{"thinking": true}'
```

### リクエスト単位の上書き { #request-level-override }

リクエスト単位の `chat_template_kwargs` は、常にサーバーの既定値より優先されます。たとえばサーバーを `enable_thinking=false` で起動していても、クライアントは特定のリクエストで有効にできます。

```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}}  # Overrides server default
)
```

## thinking budget の制御 { #thinking-budget-control }

[Qwen3](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html#thinking-budget)、[DeepSeek](https://www.alibabacloud.com/help/en/model-studio/deep-thinking)、[Nemotron3](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) など一部のモデルは、推論に使うトークン数の上限を制限する thinking budget をサポートしています。

トークンの計数は `reasoning_start_str` から始まります。推論のトークン数が設定した `thinking_token_budget` に達すると、vLLM はモデルに `reasoning_end_str` を出力させ、推論ブロックを終了させます。

この機能を使うには次のようにします。

- `--reasoning-parser` で推論の抽出を有効にします。
- `--reasoning-config` で推論の境界トークン（`reasoning_start_str`、`reasoning_end_str` など）を定義します。設定しない場合、vLLM は推論パーサーからこれらのトークンを自動的に初期化しようとします。
- `thinking_token_budget`（サンプリングパラメータ）でリクエストごとの推論トークン数の上限を設定します。

`thinking_token_budget` を指定しない場合、`max_tokens` などの通常の生成制約を超える明示的な推論の上限は適用されません。

`--reasoning-config` は、次のフィールドを持つ [`ReasoningConfig`](https://docs.vllm.ai/en/v0.26.0/api/vllm/config/#vllm.config.ReasoningConfig) に対応する JSON オブジェクトを受け取ります。

| フィールド                 | 型           | 説明                                      |
|-----------------------|----------------|--------------------------------------------------|
| `reasoning_start_str` | `str \| null`  | 推論内容の開始を示す文字列 |
| `reasoning_end_str`   | `str \| null`  | 推論内容の終了を示す文字列   |

!!! note
    `reasoning_end_str` には、推論終了トークンの前に置く移行フレーズを含められます。たとえば `reasoning_end_str` を `"I have to give the solution based on the reasoning directly now.</think>"` に設定すると、budget を使い切ったときにモデルがそのフレーズを出力するようになり、推論の終了がより自然になります。

### オンラインサービング { #online-serving }

```bash
vllm serve Qwen/Qwen3-0.6B \
    --reasoning-parser qwen3 \
    --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "I have to give the solution based on the reasoning directly now.</think>"}'
```

次に、推論トークンを制限するために `thinking_token_budget` を指定してリクエストを送ります。

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      { "role": "user", "content": "9.11 and 9.8, which is greater?" }
    ],
    "thinking_token_budget": 10
  }'
```

### オフライン推論 { #offline-inference }

```python
from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    reasoning_config=ReasoningConfig(
        reasoning_start_str="<think>",
        reasoning_end_str="I have to give the solution based on the thinking directly now.</think>",
    ),
)

sampling_params = SamplingParams(thinking_token_budget=10)

messages = [
    {"role": "user", "content": "9.11 and 9.8, which is greater?"},
]

outputs = llm.chat(messages, sampling_params=sampling_params)

for output in outputs:
    print("text:", output.outputs[0].text)
```

## `enable_thinking` の自動有効化 { #automatic-enable_thinking-activation }

一部のモデル（Gemma 4、DeepSeek-V4-Pro、IBM Granite 3.2 など）は、thinking モードを有効にするために chat template kwargs に `enable_thinking: true` を必要とします。これがないと、他の設定にかかわらず推論トークンは生成されません。

Chat Completions のリクエストで `reasoning_effort`（Responses API のリクエストでは `reasoning.effort`）を設定すると、vLLM は自動的に `enable_thinking` を chat template kwargs に挿入します。

- `reasoning_effort` が `"low"`、`"medium"`、`"high"` → `enable_thinking = true`
- `reasoning_effort` が `"none"` → `enable_thinking = false`
- `reasoning_effort` を未設定 → `enable_thinking` は挿入されません（従来の挙動を維持）

つまり、`reasoning_effort` を使う場合に `chat_template_kwargs: {"enable_thinking": true}` を手動で渡す必要はなくなり、自動的に処理されます。

!!! note
    `chat_template_kwargs` で `enable_thinking` を明示的に設定した場合、その値が自動挿入より優先されます。必要に応じて挙動を上書きできます。

    テンプレートが `enable_thinking` を宣言していないモデル（DeepSeek R1 など）では、挿入された kwarg は `resolve_chat_template_kwargs` によって無害に取り除かれます。

### 例 { #example }

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# reasoning_effort automatically enables thinking for models that need it
response = client.chat.completions.create(
    model="google/gemma-4-26B-A4B-it",
    messages=[{"role": "user", "content": "What is 15 * 37?"}],
    reasoning_effort="high",  # Automatically sets enable_thinking=true
)

print(response.choices[0].message.reasoning)
print(response.choices[0].message.content)
```

## 推論出力の抑制 { #suppressing-reasoning-output }

`include_reasoning` パラメータを使うと、API レスポンスから推論内容を抑制できます。`false` に設定すると、推論トークンは引き続き生成されるものの（モデルの品質には影響しません）、レスポンスからは除外されます。これにより、推論の挙動を変えずにネットワークのトラフィックを削減できます。

このパラメータは Chat Completions API と Responses API の両方で、ストリーミング・非ストリーミングいずれのリクエストでもサポートされます。

`include_reasoning=false` の場合、vLLM はトークン単位のメタデータ（logprobs とトークン ID）も抑制します。logprob のエントリ内のデコード済みトークンテキストや生のトークン ID を通じて推論内容が漏れるのを防ぐためです。

### Chat Completions API { #chat-completions-api }

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
model = client.models.list().data[0].id

# Reasoning is included by default (include_reasoning=True)
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What is 15 * 37?"}],
    extra_body={"include_reasoning": False},
)

msg = response.choices[0].message
assert msg.content  # Content is still present
assert not getattr(msg, "reasoning", None)  # Reasoning is suppressed
```

ストリーミングでも同様に動作し、推論の delta はチャンクから省かれます。

```python
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What is 15 * 37?"}],
    stream=True,
    extra_body={"include_reasoning": False},
)

for chunk in stream:
    delta = chunk.choices[0].delta
    # delta.reasoning will always be None
    if delta.content:
        print(delta.content, end="", flush=True)
```

### Responses API { #responses-api }

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.responses.create(
    model=client.models.list().data[0].id,
    input="What is 15 * 37?",
    include_reasoning=False,
)

# No "reasoning" items in output
types = [item.type for item in response.output]
assert "reasoning" not in types
```

## 制限事項 { #limitations }

- 推論内容が利用できるのは、オンラインサービングの chat completion エンドポイント（`/v1/chat/completions`）、Anthropic Messages API（`/v1/messages`）、Responses API（`/v1/responses`）のみです。

## 新しい推論モデルをサポートする方法 { #how-to-support-a-new-reasoning-model }

[vllm/reasoning/deepseek_r1_reasoning_parser.py](../../vllm/reasoning/deepseek_r1_reasoning_parser.py) と同様に、新しい `ReasoningParser` を追加できます。

??? code

    ```python
    # import the required packages

    from vllm.reasoning import ReasoningParser, ReasoningParserManager
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage

    # define a reasoning parser and register it to vllm
    # the name list in register_module can be used
    # in --reasoning-parser.
    class ExampleParser(ReasoningParser):
        def __init__(self, tokenizer: TokenizerLike):
            super().__init__(tokenizer)

        def extract_reasoning_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
        ) -> DeltaMessage | None:
            """
            Instance method that should be implemented for extracting reasoning
            from an incomplete response; for use when handling reasoning calls and
            streaming. Has to be an instance method because  it requires state -
            the current tokens/diffs, but also the information about what has
            previously been parsed and extracted (see constructor)
            """

        def extract_reasoning(
            self,
            model_output: str,
            request: ChatCompletionRequest | ResponsesRequest,
        ) -> tuple[str | None, str | None]:
            """
            Extract reasoning content from a complete model-generated string.

            Used for non-streaming responses where we have the entire model response
            available before sending to the client.

            Parameters:
            model_output: str
                The model-generated string to extract reasoning content from.

            request: ChatCompletionRequest
                The request object that was used to generate the model_output.

            Returns:
            tuple[Optional[str], Optional[str]]
                A tuple containing the reasoning content and the content.
            """
    # Register the reasoning parser
    ReasoningParserManager.register_lazy_module(
        name="example",
        module_path="vllm.reasoning.example_reasoning_parser",
        class_name="ExampleParser",
    )
    ```

さらに構造化出力を有効にするには、[vllm/reasoning/deepseek_r1_reasoning_parser.py](../../vllm/reasoning/deepseek_r1_reasoning_parser.py) にあるものと同様の新しい `Reasoner` を作成する必要があります。

??? code

    ```python
    @dataclass
    class DeepSeekReasoner(Reasoner):
        """
        Reasoner for DeepSeek R series models.
        """
        start_token_id: int
        end_token_id: int

        start_token: str = "<think>"
        end_token: str = "</think>"

        @classmethod
        def from_tokenizer(cls, tokenizer: PythonBackend) -> Reasoner:
            return cls(
                start_token_id=tokenizer.encode("<think>", add_special_tokens=False)[0],
                end_token_id=tokenizer.encode("</think>", add_special_tokens=False)[0],
            )

        def is_reasoning_end(self, input_ids: list[int]) -> bool:
            return self.end_token_id in input_ids

        def is_reasoning_end_streaming(self, input_ids: list[int], delta_ids: list[int]) -> bool:
            return self.end_token_id in delta_token_ids
        ...
    ```

[xgrammar](https://github.com/mlc-ai/xgrammar) のような構造化出力エンジンは、`end_token_id` を使ってモデル出力に推論内容が含まれているかを確認し、含まれている場合は構造化出力をスキップします。

最後に、`--reasoning-parser` フラグを使ってそのモデルの推論を有効にできます。

```bash
vllm serve <model_tag> --reasoning-parser example
```
