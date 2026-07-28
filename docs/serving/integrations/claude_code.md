# Claude Code { #claude-code }

[Claude Code](https://code.claude.com/docs/en/quickstart) は、ターミナル上で動作する Anthropic 公式のエージェント型コーディングツールです。コードベースを理解し、ファイルを編集し、コマンドを実行して、効率的なコーディングを支援します。

Claude Code の向き先を vLLM サーバーにすると、Anthropic API の代わりに自分のモデルをバックエンドとして使えます。次のような用途に便利です。

- 完全にローカル・プライベートなコーディング支援を動かす
- ツール呼び出しに対応したオープンウェイトのモデルを使う
- 独自モデルのテストと開発

## 仕組み { #how-it-works }

vLLM は Anthropic Messages API を実装しています。これは Claude Code が Anthropic のサーバーと通信するのと同じ API です。`ANTHROPIC_BASE_URL` を vLLM サーバーに向けると、Claude Code は Anthropic ではなく vLLM にリクエストを送ります。vLLM はそのリクエストをローカルのモデルで動くように変換し、Claude Code が期待する形式でレスポンスを返します。

つまり、ツール呼び出しに適切に対応したモデルであれば、vLLM でサービングするだけで Claude Code の Claude モデルの代替として利用できます。

## 要件 { #requirements }

Claude Code は、ツール呼び出しの能力が高いモデルを必要とします。モデルは OpenAI 互換のツール呼び出し API をサポートしている必要があります。モデルでツール呼び出しを有効にする方法は[ツール呼び出し](../../features/tool_calling.md)を参照してください。

## インストール { #installation }

まず、[公式のインストールガイド](https://docs.anthropic.com/en/docs/claude-code/getting-started)に従って Claude Code をインストールします。

## vLLM サーバーの起動 { #starting-the-vllm-server }

ツール呼び出しに対応したモデルで vLLM を起動します。以下は `openai/gpt-oss-120b` を使う例です。

```bash
vllm serve openai/gpt-oss-120b --served-model-name my-model --enable-auto-tool-choice --tool-call-parser openai
```

他のモデルでは、`--enable-auto-tool-choice` と適切な `--tool-call-parser` でツール呼び出しを明示的に有効にする必要があります。モデルごとの正しいフラグは[ツール呼び出しのドキュメント](../../features/tool_calling.md)を参照してください。

## Claude Code の設定 { #configuring-claude-code }

vLLM サーバーを指す環境変数を付けて Claude Code を起動します。

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_OPUS_MODEL=my-model \
ANTHROPIC_DEFAULT_SONNET_MODEL=my-model \
ANTHROPIC_DEFAULT_HAIKU_MODEL=my-model \
claude
```

各環境変数の意味:

| 変数                             | 説明                                                                   |
| -------------------------------- | --------------------------------------------------------------------- |
| `ANTHROPIC_BASE_URL`             | vLLM サーバーを指します（既定のポートは 8000）                        |
| `ANTHROPIC_API_KEY`              | vLLM は既定で認証を必要としないため、任意の値で構いません              |
| `ANTHROPIC_AUTH_TOKEN`           | 必須です。値は任意で構いません。                                       |
| `ANTHROPIC_DEFAULT_OPUS_MODEL`   | Opus 相当のリクエストで使うモデル名                                    |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | Sonnet 相当のリクエストで使うモデル名                                  |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL`  | Haiku 相当のリクエストで使うモデル名                                   |

!!! tip
    これらの環境変数は、シェルのプロファイル（`.bashrc`、`.zshrc` など）や Claude Code の設定ファイル（`~/.claude/settings.json`）に書いておくか、ラッパースクリプトを作っておくと便利です。

!!! warning
    Claude Code は最近、システムプロンプトにリクエストごとのハッシュを埋め込むようになりました。これによりプロンプトが毎回変わるため[プレフィックスキャッシュ](../../design/prefix_caching.md)が効かなくなり、性能が大きく低下することがあります。vLLM 0.17.1 より後のバージョンでは自動的に対処されますが、それより古いバージョンでは `~/.claude/settings.json` の `"env"` セクションに `"CLAUDE_CODE_ATTRIBUTION_HEADER": "0"` を追加してください（Unsloth の[ブログ記事](https://unsloth.ai/docs/basics/claude-code#fixing-90-slower-inference-in-claude-code)を参照）。

## 動作確認 { #testing-the-setup }

Claude Code が起動したら、簡単なプロンプトで接続を確認します。

![Claude Code example chat](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/claude-code-example.png)

モデルが正しく応答すれば設定は成功です。vLLM でサービングしたモデルを使って Claude Code でコーディングできます。

## トラブルシューティング { #troubleshooting }

**接続が拒否される**: vLLM が起動していて、指定した URL でアクセスできることを確認してください。ポートが一致しているかも確認します。

**ツール呼び出しが動かない**: モデルがツール呼び出しに対応していること、正しい `--tool-call-parser` フラグで有効化されていることを確認してください。[ツール呼び出し](../../features/tool_calling.md)を参照してください。

**モデルが見つからない**: `--served-model-name` が環境変数のモデル名と一致していることを確認してください。Hugging Face の `openai/gpt-oss-120b` のように `/` を含むモデル名はそのままでは使えないため、Claude Code ではこの制約に注意してください。
