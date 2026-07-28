# Codex { #codex }

[Codex](https://github.com/openai/codex) は、ターミナル上で動作する OpenAI 公式のエージェント型コーディングツールです。コードベースを理解し、ファイルを編集し、コマンドを実行して、効率的なコーディングを支援します。

Codex の向き先を vLLM サーバーにすると、OpenAI API の代わりに自分のモデルをバックエンドとして使えます。次のような用途に便利です。

- 完全にローカル・プライベートなコーディング支援を動かす
- ツール呼び出しに対応したオープンウェイトのモデルを使う
- 独自モデルのテストと開発

## 仕組み { #how-it-works }

vLLM は OpenAI Responses API を実装しています。これは Codex が OpenAI のサーバーと通信するのと同じ API です。Codex の向き先を vLLM サーバーに設定すると、Codex は OpenAI ではなく vLLM にリクエストを送ります。vLLM はそのリクエストをローカルのモデルで動くように変換し、Codex が期待する形式でレスポンスを返します。

つまり、ツール呼び出しに適切に対応したモデルであれば、vLLM でサービングするだけで Codex の OpenAI モデルの代替として利用できます。

## 要件 { #requirements }

Codex は、ツール呼び出しの能力が高いモデルを必要とします。モデルは OpenAI Responses のツール呼び出し API をサポートしている必要があります。モデルでツール呼び出しを有効にする方法は[ツール呼び出し](../../features/tool_calling.md)を参照してください。

## インストール { #installation }

まず、[公式のインストールガイド](https://github.com/openai/codex)に従って Codex をインストールします。

## vLLM サーバーの起動 { #starting-the-vllm-server }

ツール呼び出しに対応したモデルで vLLM を起動します。以下は `Qwen/Qwen3-27B` を使う例です。

```bash
vllm serve Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder

```

他のモデルでは、`--enable-auto-tool-choice` と適切な `--tool-call-parser` でツール呼び出しを明示的に有効にする必要があります。モデルごとの正しいフラグは[ツール呼び出しのドキュメント](../../features/tool_calling.md)を参照してください。

## Codex の設定 { #configuring-codex }

Codex は `~/.codex/config.toml` にある TOML ファイルで設定します。このファイルを作成または編集して、Codex の向き先を vLLM サーバーにします。

```toml
model = "my-model"
model_provider = "vllm"

[model_providers.vllm]
name = "vLLM"
env_key = "VLLM_API_KEY"
base_url = "http://localhost:8000/v1"
wire_api = "responses"
```

各設定項目の意味:

| 項目 | 説明 |
| ----- | ----------- |
| `model` | 使用するモデル名。vLLM に渡した `--served-model-name` と一致させる必要があります。 |
| `model_provider` | ローカルの vLLM サーバーを使うには `"vllm"` を指定します。 |
| `[model_providers.vllm]` | vLLM プロバイダの設定セクション。 |
| `name` | vLLM プロバイダの表示名。 |
| `env_key` | Codex が API キーとして読み取る環境変数の名前。vLLM は既定で認証を必要としないため、任意の値で構いません。 |
| `base_url` | vLLM サーバーの OpenAI 互換 API エンドポイントの URL（既定は `http://localhost:8000/v1`）。 |
| `wire_api` | 使用する API の形式。OpenAI Responses API を使うには `"responses"` を指定します。 |

!!! tip
    vLLM は既定で認証を必要としないため、`env_key` には任意のダミーの環境変数を指定できます。
    ```bash
    export VLLM_API_KEY=dummy
    ```

!!! warning
    `responses` API を使う場合は、お使いの vLLM のバージョンが OpenAI Responses API に対応していることを確認してください。

## 動作確認 { #testing-the-setup }

Codex を設定したら、プロジェクトのディレクトリで起動します。

```bash
codex
```

プロジェクト内のファイルの説明を求めるなど、簡単なプロンプトで接続を確認します。モデルが正しく応答すれば設定は成功です。vLLM でサービングしたモデルを使って Codex でコーディングできます。

## トラブルシューティング { #troubleshooting }

**接続が拒否される**: vLLM が起動していて、指定した URL でアクセスできることを確認してください。ポートが一致していること、`base_url` に `/v1` のパスが含まれていることも確認します。

**ツール呼び出しが動かない**: モデルがツール呼び出しに対応していること、正しい `--tool-call-parser` フラグで有効化されていることを確認してください。[ツール呼び出し](../../features/tool_calling.md)を参照してください。

**モデルが見つからない**: `~/.codex/config.toml` の `model` が、vLLM に渡した `--served-model-name` と一致していることを確認してください。
