# Streamlit { #streamlit }

[Streamlit](https://github.com/streamlit/streamlit) を使うと、Python のスクリプトを数週間ではなく数分でインタラクティブな Web アプリに変えられます。ダッシュボードの構築、レポートの生成、チャットアプリの作成などが可能です。

vLLM をバックエンドの API サーバーとしてすぐに統合でき、API 呼び出しによる強力な LLM 推論を利用できます。

## 前提条件 { #prerequisites }

必要なパッケージをインストールして vLLM の環境を用意します。

```bash
pip install vllm streamlit openai
```

## デプロイ { #deploy }

1. 対応するチャット補完モデルで vLLM サーバーを起動します。例:

    ```bash
    vllm serve Qwen/Qwen1.5-0.5B-Chat
    ```

1. 次のスクリプトを使います: [examples/applications/chatbot/streamlit_openai_chatbot_webserver.py](../../../examples/applications/chatbot/streamlit_openai_chatbot_webserver.py)

1. Streamlit の Web UI を起動してチャットを始めます。

    ```bash
    streamlit run streamlit_openai_chatbot_webserver.py

    # or specify the VLLM_API_BASE or VLLM_API_KEY
    VLLM_API_BASE="http://vllm-server-host:vllm-server-port/v1" \
        streamlit run streamlit_openai_chatbot_webserver.py

    # start with debug mode to view more details
    streamlit run streamlit_openai_chatbot_webserver.py --logger.level=debug
    ```

    ![Chat with vLLM assistant in Streamlit](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/streamlit-chat.png)
