# デレンダラー API { #derenderer-apis }

デレンダラー API は、[レンダラー API](renderer.md) と対になる後処理の API です。`/render` がリクエストをトークン ID に変換する（前処理）のに対し、`/derender` は生成されたトークン ID を、完全な OpenAI 互換のレスポンスへ戻します（デトークナイズ、reasoning の解析、ツール呼び出しの解析）。いずれも GPU を必要としません。

これにより、分離型サービングにおけるトークン入力・トークン出力のエンジンの一巡が完成します。

- **GPU レスの後処理**: デトークナイズ、reasoning の解析、ツール呼び出しの解析を、`/render` をホストするのと同じ GPU レスのフロントエンドで実行します
- **パーサーの同等性**: デレンダラーは vLLM のツールパーサーと reasoning パーサーを再利用するため、分離型のデプロイでも通常の `vllm serve` と同じ `content` / `reasoning` / `tool_calls` の分割結果が得られます
- **非ストリーミング**: これらのエンドポイントは、すべてのトークン ID が揃った完全な `GenerateResponse` を受け取り、一度に解析します。ストリーミングのデレンダリングには別のエンドポイント設計が必要で、現時点では未対応ですが計画中です

どちらのエンドポイントも、[`vllm launch render`](../../cli/launch/render.md) で起動する GPU レスのレンダリングサーバーが `/render` 系のエンドポイントとあわせてホストします。

## パイプライン { #pipeline }

```text
                render                 generate                derender
  request  ───────────────▶  token_ids  ─────────▶  token_ids  ──────────▶  response
 (chat /            (GPU less)          (token-in /            (GPU less)   (OpenAI
 completion)            │               token-out engine)          ▲        compatible)
                        └─────────────── request + prompt_tokens ──┘
```

デレンダリングの工程には、エンジンが返す `token_ids` だけでは足りません。レンダリングの工程から引き継いだ元の `chat_request` / `completion_request` と `prompt_tokens` も使います（[リクエストの形式](#request-format)を参照）。これにより、ツールパーサーと reasoning パーサーが必要な文脈を得られます。

## API リファレンス { #api-reference }

- Chat Completions Derender API (`/v1/chat/completions/derender`)
    - 単一の `GenerateResponse` を後処理して `ChatCompletionResponse` にします
- Completions Derender API (`/v1/completions/derender`)
    - `GenerateResponse` のリスト（プロンプトごとに 1 つ）を後処理して `CompletionResponse` にします

## リクエストの形式 { #request-format }

各リクエストは、エンジンの `GenerateResponse` と、GPU なしで最終的なレスポンスを再構成するために必要な呼び出し側のメタデータをまとめたものです。

`/v1/chat/completions/derender`:

??? code

    ```python
    --8<-- "vllm/entrypoints/scale_out/token_in_token_out/protocol.py:derender-chat-request"
    ```

`/v1/completions/derender`:

??? code

    ```python
    --8<-- "vllm/entrypoints/scale_out/token_in_token_out/protocol.py:derender-completion-request"
    ```

サイズが大きすぎるペイロードは、`tokenizer.decode()` やパーサーが動く前に `400` で拒否されます。

## 例 { #example }

以下の例は、GPU レスのレンダリングサーバー（`/render`、`/derender`）とトークン入力・トークン出力のエンジン（`/inference/v1/generate`）に対して、チャットリクエストの `render → generate → derender` の一巡を実行します。

```python
import httpx

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
RENDER = "http://localhost:8100"  # vllm launch render ...
ENGINE = "http://localhost:8200"  # token-in / token-out engine

chat_request = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 32,
}

with httpx.Client(timeout=60.0) as client:
    # 1. Render: request -> token IDs (GPU less)
    generate_request = client.post(
        f"{RENDER}/v1/chat/completions/render", json=chat_request
    ).json()
    prompt_tokens = len(generate_request["token_ids"])

    # 2. Generate: token IDs -> token IDs (token-in / token-out engine)
    generate_response = client.post(
        f"{ENGINE}/inference/v1/generate", json=generate_request
    ).json()

    # 3. Derender: token IDs -> ChatCompletionResponse (GPU less)
    response = client.post(
        f"{RENDER}/v1/chat/completions/derender",
        json={
            "model": MODEL,
            "generate_response": generate_response,
            "prompt_tokens": prompt_tokens,
            "chat_request": chat_request,
        },
    ).json()

print(response["choices"][0]["message"]["content"])
```

`chat_request` を渡すと、デレンダラーは設定されたツールパーサーと reasoning パーサーを実行できます。そのため `response["choices"][0]["message"]` には、`vllm serve` のサーバーと同じ `content` / `reasoning` / `tool_calls` の分割結果が入ります。単純なデトークナイズだけでよい場合は `chat_request` を省略してください。
