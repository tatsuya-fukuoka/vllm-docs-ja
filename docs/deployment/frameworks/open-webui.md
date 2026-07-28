# Open WebUI { #open-webui }

[Open WebUI](https://github.com/open-webui/open-webui) は、完全にオフラインで動作するように設計された、拡張性が高く機能豊富で使いやすいセルフホスト型の AI プラットフォームです。
Ollama や OpenAI 互換 API など各種の LLM ランナーに対応し、RAG の機能も内蔵しているため、強力な AI のデプロイ手段になります。

vLLM で Open WebUI を使い始めるには、次の手順に従ってください。

1. [Docker](https://docs.docker.com/engine/install/) をインストールします。

2. 対応するチャット補完モデルで vLLM サーバーを起動します。

    ```console
    vllm serve Qwen/Qwen3-0.6B-Chat
    ```

    !!! note
        vLLM サーバーを起動するときは、`--host` と `--port` フラグでホストとポートを必ず指定してください。
        例:

        ```console
        vllm serve <model> --host 0.0.0.0 --port 8000
        ```

3. Open WebUI の Docker コンテナを起動します。

    ```console
    docker run -d \
        --name open-webui \
        -p 3000:8080 \
        -v open-webui:/app/backend/data \
        -e OPENAI_API_BASE_URL=http://0.0.0.0:8000/v1 \
        --restart always \
        ghcr.io/open-webui/open-webui:main
    ```

4. ブラウザで <http://open-webui-host:3000/> を開きます。

    ページ上部に `Qwen/Qwen3-0.6B-Chat` というモデル名が表示されるはずです。

    ![Web portal of model Qwen/Qwen3-0.6B-Chat](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/open_webui.png)
