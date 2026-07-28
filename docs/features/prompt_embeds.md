# プロンプト埋め込み入力 { #prompt-embedding-inputs }

このページでは、vLLM にプロンプト埋め込み（prompt embedding）を入力として渡す方法を説明します。

## プロンプト埋め込みとは { #what-are-prompt-embeddings }

大規模言語モデルにおけるテキストデータの伝統的な流れは、テキストから（トークナイザーを通じて）トークン ID へ、そしてトークン ID からプロンプト埋め込みへ、というものです。従来のデコーダのみのモデル（meta-llama/Llama-3.1-8B-Instruct など）では、トークン ID をプロンプト埋め込みに変換するこのステップは、学習済みの埋め込み行列からの参照によって行われます。しかし、モデルが処理できるのは自身のトークン語彙に対応する埋め込みだけに限られるわけではありません。

## オフライン推論 { #offline-inference }

マルチモーダルデータを入力するには、[vllm.inputs.EmbedsPrompt][] のスキーマに従います。

- `prompt_embeds`: プロンプト / トークンの埋め込み列を表す torch テンソル。形状は (sequence_length, hidden_size) で、sequence_length はトークン埋め込みの数、hidden_size はモデルの隠れ層のサイズ（埋め込みサイズ）です。

### Hugging Face Transformers からの入力 { #hugging-face-transformers-inputs }

次の例のように、Hugging Face Transformers のモデルから得たプロンプト埋め込みを、プロンプト埋め込み辞書の `'prompt_embeds'` フィールドに渡せます。

[examples/features/prompt_embed/prompt_embed_offline.py](../../examples/features/prompt_embed/prompt_embed_offline.py)

## オンラインサービング { #online-serving }

vLLM の OpenAI 互換サーバーは、[Completions API](https://platform.openai.com/docs/api-reference/completions) と [Chat Completions API](https://platform.openai.com/docs/api-reference/chat) の両方でプロンプト埋め込み入力を受け付けます。どちらも `vllm serve` の `--enable-prompt-embeds` フラグで有効になります。

### Completions API { #completions-api }

プロンプト埋め込みの入力は、JSON リクエストボディの `'prompt_embeds'` キーで追加します。

1 つのリクエストで `'prompt_embeds'` と `'prompt'` の入力が混在している場合、プロンプト埋め込みの結果が常に先に返されます。

プロンプト埋め込みは base64 エンコードされた torch テンソルとして渡します。

Completions エンドポイントは `prompt_embeds` にチャットテンプレートを適用**しません**。モデルが何らかのチャットテンプレートを前提としている場合、テンプレート適用済みのプロンプト全体に対する埋め込みを生成するのは呼び出し側の責任です。つまり、チャットテンプレートを適用してから、その結果のトークン ID を埋め込みに変換してください。モデルが通常必要とするもの（システムプロンプト、ロールのマーカー、生成プロンプトなど）は、あらかじめ埋め込み対象のトークンに含めておく必要があります。

### Chat Completions API { #chat-completions-api }

プロンプト埋め込みは、チャットメッセージのコンテンツパートとしてテキストと交互に含められます。

```json
{
  "messages": [
    {
      "role": "system",
      "content": [
        {"type": "text", "text": "You are a helpful assistant."},
        {"type": "prompt_embeds", "data": "<base64_encoded_tensor>"}
      ]
    },
    {
      "role": "user",
      "content": [
        {"type": "prompt_embeds", "data": "<base64_encoded_tensor>"},
        {"type": "text", "text": "Summarize the above."}
      ]
    }
  ]
}
```

各 `prompt_embeds` コンテンツパートには `data` フィールドがあり、形状 `(num_tokens, hidden_size)` の `torch.Tensor` を base64 エンコードしたものを格納します。`prompt_embeds` パートは、どのメッセージにも、テキストパートとの相対位置を問わず複数含められます。サーバーはチャットテンプレートのレンダリング時に各パートを適切な数のプレースホルダートークンへ展開し、計算済みの埋め込みを対応する位置でモデルの入力に差し込みます。

Completions API とは異なり、`prompt_embeds` のコンテンツパートにはコンテンツ**のみ**をエンコードすべきで、テンプレート適用済みの会話を入れてはいけません。サーバーはリクエスト時に、埋め込まれたコンテンツの周りにチャットテンプレートを適用します。これは、プレーンテキストの `content` 文字列に対して行うのと同じ処理です。ここにテンプレート適用済みの会話全体を埋め込むと、テンプレートが二重に適用され、モデルへの入力が不正になります。

!!! warning
    埋め込みの形状が誤っていると、vLLM エンジンがクラッシュする可能性があります。
    このフラグは信頼できるユーザーに対してのみ有効にしてください。

### OpenAI クライアント経由で Transformers の入力を渡す { #transformers-inputs-via-openai-client }

まず、OpenAI 互換サーバーを起動します。

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct --runner generate \
  --max-model-len 4096 --enable-prompt-embeds
```

次のように OpenAI クライアントを利用できます。

[examples/features/prompt_embed/prompt_embed_inference_with_openai_client.py](../../examples/features/prompt_embed/prompt_embed_inference_with_openai_client.py)
