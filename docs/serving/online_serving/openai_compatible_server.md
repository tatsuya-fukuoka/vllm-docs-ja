# OpenAI 互換サーバー { #openai-compatible-server }

vLLM は、OpenAI の [Completions API](https://platform.openai.com/docs/api-reference/completions) や [Chat API](https://platform.openai.com/docs/api-reference/chat) などを実装した HTTP サーバーを提供します。これにより、モデルをサービングして HTTP クライアントからやり取りできます。

## サポートしている API { #supported-apis }

現在サポートしている OpenAI API は次のとおりです。

- [Completions API](#completions-api) (`/v1/completions`)
    - [テキスト生成モデル](../../models/generative_models.md)にのみ適用できます。
    - *注意: `suffix` パラメータはサポートされていません。*
- [Chat Completions API](#chat-api) (`/v1/chat/completions`)
    - [チャットテンプレート](../online_serving/README.md#chat-template)を持つ[テキスト生成モデル](../../models/generative_models.md)にのみ適用できます。
    - *注意: `user` パラメータは無視されます。*
    - *注意:* `parallel_tool_calls` パラメータを `false` にすると、vLLM は 1 リクエストにつきツール呼び出しを 0 個または 1 個しか返しません。`true`（既定値）にすると 1 リクエストで複数のツール呼び出しを返せます。ただし `true` にしても複数返ることは保証されません。この挙動はモデルに依存し、すべてのモデルが並列のツール呼び出しに対応しているわけではないためです。
- [Chat Completions batch API](#chat-api) (`/v1/chat/completions/batch`)
- [Responses API](#responses-api) (`/v1/responses`, `/v1/responses/{response_id}`, `/v1/responses/{response_id}/cancel`)
    - [テキスト生成モデル](../../models/generative_models.md)にのみ適用できます。
- [Embeddings API](../../models/pooling_models/embed.md#openai-compatible-embeddings-api) (`/v1/embeddings`)
    - [埋め込みモデル](../../models/pooling_models/embed.md)にのみ適用できます。
- [Transcriptions API](./speech_to_text.md#transcriptions-api) (`/v1/audio/transcriptions`)
    - [自動音声認識 (ASR) モデル](../../models/supported_models.md#transcription)にのみ適用できます。
- [Translation API](./speech_to_text.md#translations-api) (`/v1/audio/translations`)
    - [自動音声認識 (ASR) モデル](../../models/supported_models.md#transcription)にのみ適用できます。

## Completions API { #completions-api }

ターミナルで vLLM を[インストール](../../getting_started/installation/README.md)し、[`vllm serve`](../../configuration/serve_args.md) コマンドでサーバーを起動します（[Docker](../../deployment/docker.md) イメージを使うこともできます）。

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
  --dtype auto \
  --api-key token-abc123
```

サーバーを呼び出すには、好みのテキストエディタで HTTP クライアントを使うスクリプトを作成し、モデルに送りたいメッセージを記述して実行します。以下は[公式の OpenAI Python クライアント](https://github.com/openai/openai-python)を使った例です。

??? code

    ```python
    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )

    completion = client.chat.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "user", "content": "Hello!"},
        ],
    )

    print(completion.choices[0].message)
    ```

!!! tip
    vLLM は、OpenAI がサポートしていないパラメータ（たとえば `top_k`）にも対応しています。
    OpenAI クライアントからこれらを渡すには、リクエストの `extra_body` パラメータを使います（`top_k` の場合は `extra_body={"top_k": 50}`）。

!!! important
    既定では、Hugging Face のモデルリポジトリに `generation_config.json` があればサーバーがそれを適用します。つまり、一部のサンプリングパラメータの既定値がモデル作成者の推奨値で上書きされます。

    この動作を無効にするには、サーバー起動時に `--generation-config vllm` を指定してください。

## 追加パラメータ { #extra-parameters }

vLLM は、OpenAI API には含まれない一連のパラメータをサポートしています。
これらを使うには、OpenAI クライアントの追加パラメータとして渡すか、
HTTP を直接呼び出している場合は JSON ペイロードにそのまま含めてください。

```python
completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"},
    ],
    extra_body={
        "structured_outputs": {"choice": ["positive", "negative"]},
    },
)
```

## 追加の HTTP ヘッダー { #extra-http-headers }

現時点でサポートしている HTTP リクエストヘッダーは `X-Request-Id` のみです。
`--enable-request-id-headers` を指定すると有効になります。

??? code

    ```python
    completion = client.chat.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"},
        ],
        extra_headers={
            "x-request-id": "sentiment-classification-00001",
        },
    )
    print(completion._request_id)

    completion = client.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        prompt="A robot may not injure a human being",
        extra_headers={
            "x-request-id": "completion-test",
        },
    )
    print(completion._request_id)
    ```

## API リファレンス { #api-reference }

### Completions API { #completions-api_1 }

vLLM の Completions API は [OpenAI の Completions API](https://platform.openai.com/docs/api-reference/completions) と互換性があり、
[公式の OpenAI Python クライアント](https://github.com/openai/openai-python)からやり取りできます。

コード例: [examples/basic/online_serving/openai_completion_client.py](../../../examples/basic/online_serving/openai_completion_client.py)

#### 追加パラメータ { #extra-parameters_1 }

次の[サンプリングパラメータ](https://docs.vllm.ai/en/v0.26.0/api/#inference-parameters)がサポートされています。

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/completion/protocol.py:completion-sampling-params"
    ```

次の追加パラメータがサポートされています。

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/completion/protocol.py:completion-extra-params"
    ```

### Chat API { #chat-api }

vLLM の Chat API は [OpenAI の Chat Completions API](https://platform.openai.com/docs/api-reference/chat) と互換性があり、
[公式の OpenAI Python クライアント](https://github.com/openai/openai-python)からやり取りできます。

[Vision](https://platform.openai.com/docs/guides/vision) と
[Audio](https://platform.openai.com/docs/guides/audio?audio-generation-quickstart-example=audio-in) に関するパラメータの両方をサポートしています。
詳細は[マルチモーダル入力](../../features/multimodal_inputs.md)のガイドを参照してください。

- *注意: `image_url.detail` パラメータはサポートされていません。*

コード例: [examples/basic/online_serving/openai_chat_completion_client.py](../../../examples/basic/online_serving/openai_chat_completion_client.py)

#### 追加パラメータ { #extra-parameters_2 }

次の[サンプリングパラメータ](https://docs.vllm.ai/en/v0.26.0/api/#inference-parameters)がサポートされています。

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/chat_completion/protocol.py:chat-completion-sampling-params"
    ```

次の追加パラメータがサポートされています。

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/chat_completion/protocol.py:chat-completion-extra-params"
    ```

### Responses API { #responses-api }

vLLM の Responses API は [OpenAI の Responses API](https://platform.openai.com/docs/api-reference/responses) と互換性があり、
[公式の OpenAI Python クライアント](https://github.com/openai/openai-python)からやり取りできます。

コード例: [examples/tool_calling/openai_responses_client_with_tools.py](../../../examples/tool_calling/openai_responses_client_with_tools.py)

#### 追加パラメータ { #extra-parameters_3 }

リクエストオブジェクトでは次の追加パラメータがサポートされています。

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/responses/protocol.py:responses-extra-params"
    ```

レスポンスオブジェクトでは次の追加パラメータがサポートされています。

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/responses/protocol.py:responses-response-extra-params"
    ```
