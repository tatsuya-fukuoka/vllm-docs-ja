# レンダラー API { #renderer-apis }

vLLM のレンダラー API は、レンダリング（前処理）の工程を切り離し、トークン入力・トークン出力の API サーバーを実現するために設計されています。

- フロントエンドの GPU レス化: 前処理（トークナイズ、マルチモーダル入力の処理）と後処理（デトークナイズ、ツール呼び出しの解析、reasoning の解析）を GPU なしで実行できます。
- トークナイズの分離: llm-d、Dynamo、独自フロントエンドなど、推論エンジン全体を動かさずに vLLM の前処理ロジックだけを利用したいユースケースに対応します。
- トークン入力・トークン出力のエンジン: エンジンをリクエストの前処理から切り離し、純粋なトークン入出力のサービスにします。

## API リファレンス { #api-reference }

- [Completions Render API](renderer.md) (`/v1/completions/render`)
    - completion リクエストをレンダリングします
- [Chat Completions Render API](renderer.md) (`/v1/chat/completions/render`)
    - chat completion をレンダリングします

生成されたトークン ID を OpenAI 互換のレスポンスに戻す後処理側については、[デレンダラー API](derenderer.md) を参照してください。
