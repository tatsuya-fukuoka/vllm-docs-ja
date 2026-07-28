# サーバー引数 { #server-arguments }

`vllm serve` コマンドは OpenAI 互換サーバーを起動するために使います。

## CLI 引数 { #cli-arguments }

`vllm serve` コマンドは OpenAI 互換サーバーを起動するために使います。
指定できるオプションは [CLI リファレンス](../cli/README.md)を参照してください。

## 設定ファイル { #configuration-file }

CLI 引数は [YAML](https://yaml.org/) の設定ファイルから読み込むこともできます。
引数名は[上記](serve_args.md)の長い形式（ロングオプション）で記述する必要があります。

例:

```yaml
# config.yaml

model: meta-llama/Llama-3.1-8B-Instruct
host: "127.0.0.1"
port: 6379
uvicorn-log-level: "info"
```

上記の設定ファイルを使うには次のようにします。

```bash
vllm serve --config config.yaml
```

!!! note
    同じ引数がコマンドラインと設定ファイルの両方で指定された場合は、コマンドラインの値が優先されます。
    優先順位は `コマンドライン > 設定ファイルの値 > 既定値` です。
    たとえば `vllm serve SOME_MODEL --config config.yaml` では、設定ファイルの `model` より SOME_MODEL が優先されます。
