# AnythingLLM { #anythingllm }

[AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) は、あらゆるドキュメント・リソース・コンテンツを、チャット中に LLM が参照できるコンテキストへ変換できるフルスタックのアプリケーションです。

vLLM をバックエンドとする大規模言語モデル (LLM) のサーバーをデプロイし、OpenAI 互換のエンドポイントを利用できます。

## 前提条件 { #prerequisites }

vLLM の環境を用意します。

```bash
pip install vllm
```

## デプロイ { #deploy }

1. 対応するチャット補完モデルで vLLM サーバーを起動します。例:

    ```bash
    vllm serve Qwen/Qwen1.5-32B-Chat-AWQ --max-model-len 4096
    ```

1. [AnythingLLM デスクトップ版](https://anythingllm.com/desktop)をダウンロードしてインストールします。

1. AI プロバイダを設定します。

    - 画面下部の 🔧 レンチアイコン -> **Open settings** -> **AI Providers** -> **LLM** をクリックします。
    - 次の値を入力します。
        - LLM Provider: Generic OpenAI
        - Base URL: `http://{vllm server host}:{vllm server port}/v1`
        - Chat Model Name: `Qwen/Qwen1.5-32B-Chat-AWQ`

    ![set AI providers](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/anything-llm-provider.png)

1. ワークスペースを作成します。

    1. 画面下部の ↺ 戻るアイコンをクリックしてワークスペース一覧に戻ります。
    1. ワークスペース（`vllm` など）を作成し、チャットを始めます。

    ![create a workspace](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/anything-llm-chat-without-doc.png)

1. ドキュメントを追加します。

    1. 📎 添付アイコンをクリックします。
    1. ドキュメントをアップロードします。
    1. そのドキュメントを選択してワークスペースに移動します。
    1. 保存して埋め込みを作成します。

    ![add a document](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/anything-llm-upload-doc.png)

1. ドキュメントをコンテキストとしてチャットします。

    ![chat with your context](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/anything-llm-chat-with-doc.png)
