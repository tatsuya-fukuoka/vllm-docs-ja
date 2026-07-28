# RunPod { #runpod }

vLLM は [RunPod](https://www.runpod.io/) 上にデプロイできます。RunPod は、AI 推論のワークロード向けにオンデマンドおよびサーバーレスの GPU インスタンスを提供するクラウド GPU プラットフォームです。

## 前提条件 { #prerequisites }

- GPU Pod を利用できる RunPod のアカウント
- CUDA 対応テンプレート（`runpod/pytorch` など）で動作する GPU Pod

## サーバーの起動 { #starting-the-server }

RunPod の Pod に SSH で接続し、vLLM の OpenAI 互換サーバーを起動します。

```bash
vllm serve <model-name> \
    --host 0.0.0.0 \
    --port 8000
```

!!! note

    コンテナの外からアクセスできるよう、`--host 0.0.0.0` ですべてのインターフェイスにバインドしてください。

## ポート 8000 の公開 { #exposing-port-8000 }

RunPod はプロキシ経由で HTTP サービスを公開します。ポート 8000 にアクセスできるようにするには次の手順を行います。

1. RunPod のダッシュボードで Pod の設定を開きます。
2. 公開する HTTP ポートの一覧に `8000` を追加します。
3. Pod の再起動後、RunPod が次の形式の公開 URL を提供します。

    ```text
    https://<pod-id>-8000.proxy.runpod.net
    ```

## 502 Bad Gateway のトラブルシューティング { #troubleshooting-502-bad-gateway }

RunPod のプロキシが返す `502 Bad Gateway` は、通常サーバーがまだ待ち受けていないことを意味します。よくある原因は次のとおりです。

- **モデルの読み込み中** — 大きなモデルはダウンロードと GPU メモリへの読み込みに時間がかかります。Pod のログで進捗を確認してください。
- **バインド先のホストが誤っている** — `--host 0.0.0.0` を指定したか確認してください。既定の `127.0.0.1` にバインドすると、プロキシからサーバーに到達できません。
- **ポートの不一致** — `--port` の値が RunPod のダッシュボードで公開したポートと一致しているか確認してください。
- **GPU メモリ不足** — 割り当てた GPU に対してモデルが大きすぎる可能性があります。ログに CUDA の OOM エラーがないか確認し、より大きなインスタンスを使うか、複数 GPU の Pod では `--tensor-parallel-size` の追加を検討してください。

## デプロイの確認 { #verifying-the-deployment }

サーバーが起動したら、curl でリクエストを送って確認します。

!!! console "コマンド"

    ```bash
    curl https://<pod-id>-8000.proxy.runpod.net/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "<model-name>",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "max_tokens": 50
        }'
    ```

!!! console "レスポンス"

    ```json
    {
        "id": "chat-abc123",
        "object": "chat.completion",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I'm doing well, thank you for asking! How can I help you today?"
                },
                "index": 0,
                "finish_reason": "stop"
            }
        ]
    }
    ```

サーバーのヘルスチェックのエンドポイントも確認できます。

```bash
curl https://<pod-id>-8000.proxy.runpod.net/health
```
