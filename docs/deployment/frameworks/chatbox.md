# Chatbox { #chatbox }

[Chatbox](https://github.com/chatboxai/chatbox) は、Windows・Mac・Linux で使える LLM 向けのデスクトップクライアントです。

vLLM をバックエンドとする大規模言語モデル (LLM) のサーバーをデプロイし、OpenAI 互換のエンドポイントを利用できます。

## 前提条件 { #prerequisites }

vLLM の環境を用意します。

```bash
pip install vllm
```

## デプロイ { #deploy }

1. 対応するチャット補完モデルで vLLM サーバーを起動します。例:

    ```bash
    vllm serve qwen/Qwen1.5-0.5B-Chat
    ```

1. [Chatbox デスクトップ版](https://chatboxai.app/en#download)をダウンロードしてインストールします。

1. 設定画面の左下から「Add Custom Provider」を選び、次のように設定します。
    - API Mode: `OpenAI API Compatible`
    - Name: vllm
    - API Host: `http://{vllm server host}:{vllm server port}/v1`
    - API Path: `/chat/completions`
    - Model: `qwen/Qwen1.5-0.5B-Chat`

    ![Chatbox settings screen](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/chatbox-settings.png)

1. `Just chat` を開いてチャットを始めます。

    ![Chatbot chat screen](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/chatbox-chat.png)
