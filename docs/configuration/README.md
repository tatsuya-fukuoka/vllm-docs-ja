# 設定オプション { #configuration-options }

このセクションでは、vLLM を実行するときによく使うオプションを説明します。

設定には大きく 3 つの階層があり、優先度は次の順です（高い順）。

- [リクエストパラメータ](../serving/online_serving/openai_compatible_server.md#completions-api)と[入力引数](https://docs.vllm.ai/en/v0.26.0/api/#inference-parameters)
- [エンジン引数](./engine_args.md)
- [環境変数](./env_vars.md)
