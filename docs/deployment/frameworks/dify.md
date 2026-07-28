# Dify { #dify }

[Dify](https://github.com/langgenius/dify) はオープンソースの LLM アプリ開発プラットフォームです。直感的なインターフェイスに、エージェント型 AI のワークフロー、RAG のパイプライン、エージェント機能、モデル管理、可観測性などを備え、プロトタイプから本番まで素早く進められます。

モデルプロバイダとして vLLM に対応しており、大規模言語モデルを効率的にサービングできます。

このガイドでは、vLLM をバックエンドとして Dify をデプロイする手順を説明します。

## 前提条件 { #prerequisites }

vLLM の環境を用意します。

```bash
pip install vllm
```

さらに [Docker](https://docs.docker.com/engine/install/) と [Docker Compose](https://docs.docker.com/compose/install/) をインストールします。

## デプロイ { #deploy }

1. 対応するチャット補完モデルで vLLM サーバーを起動します。例:

    ```bash
    vllm serve Qwen/Qwen1.5-7B-Chat
    ```

1. docker compose で Dify のサーバーを起動します（[詳細](https://github.com/langgenius/dify?tab=readme-ov-file#quick-start)）。

    ```bash
    git clone https://github.com/langgenius/dify.git
    cd dify
    cd docker
    cp .env.example .env
    docker compose up -d
    ```

1. ブラウザで `http://localhost/install` を開き、基本のログイン情報を設定してログインします。

1. 右上のユーザーメニュー（プロフィールアイコン）から Settings を開き、`Model Provider` をクリックして `vLLM` プロバイダを見つけ、インストールします。

1. モデルプロバイダの情報を次のように入力します。

    - **Model Type**: `LLM`
    - **Model Name**: `Qwen/Qwen1.5-7B-Chat`
    - **API Endpoint URL**: `http://{vllm_server_host}:{vllm_server_port}/v1`
    - **Model Name for API Endpoint**: `Qwen/Qwen1.5-7B-Chat`
    - **Completion Mode**: `Completion`

    ![Dify settings screen](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/dify-settings.png)

1. テスト用のチャットボットを作るには、`Studio → Chatbot → Create from Blank` を選び、種類として Chatbot を選択します。

    ![Dify create chatbot screen](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/dify-create-chatbot.png)

1. 作成したチャットボットをクリックしてチャット画面を開き、モデルとやり取りを始めます。

    ![Dify chat screen](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/dify-chat.png)
