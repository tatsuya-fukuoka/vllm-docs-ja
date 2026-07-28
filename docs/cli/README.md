# vLLM CLI ガイド { #vllm-cli-guide }

vllm コマンドラインツールは、vLLM のモデルを実行・管理するために使います。まずはヘルプメッセージを表示してみましょう。

```bash
vllm --help
```

利用できるコマンド:

```bash
vllm {chat,complete,serve,launch,bench,collect-env,run-batch}
```

## serve { #serve }

vLLM の OpenAI 互換 API サーバーを起動します。

モデルを指定して起動する:

```bash
vllm serve meta-llama/Llama-2-7b-hf
```

ポートを指定する:

```bash
vllm serve meta-llama/Llama-2-7b-hf --port 8100
```

Unix ドメインソケットでサービングする:

```bash
vllm serve meta-llama/Llama-2-7b-hf --uds /tmp/vllm.sock
```

その他のオプションは --help で確認できます。

```bash
# To list all flags
vllm serve --help=all

# To view an argument group
vllm serve --help=ModelConfig

# To view a single argument
vllm serve --help=max-num-seqs

# To search by keyword or flag name
vllm serve --help=max
```

!!! tip "人が読みやすい整数の指定"
    多くの整数の引数では、利便性のために人が読みやすい接尾辞を指定できます。例:

    - `1k` = 1,000（10 進のキロ）
    - `1K` = 1,024（2 進のキビ）
    - `1m` = 1,000,000（10 進のメガ）
    - `1M` = 1,048,576（2 進のメビ）
    - `1g` / `1G` = 10 億 / 1 ギビ
    - `1t` / `1T` = 1 兆 / 1 テビ
    
    10 進の接尾辞（`k`、`m`、`g`、`t`）は小数も受け付けます: `25.6k` = 25,600。
    2 進の接尾辞（`K`、`M`、`G`、`T`）は整数のみです: `32K` = 32,768。
    
    対応する引数には `--max-model-len`、`--max-num-batched-tokens`、`--max-num-scheduled-tokens`、`--kv-cache-memory-bytes`、`--safetensors-prefetch-block-size` などがあります。

指定できる引数の完全な一覧は [vllm serve](./serve.md) を参照してください。

## launch { #launch }

vLLM の個々のコンポーネントを起動します。

```bash
# Launch the rendering server component
vllm launch render meta-llama/Llama-3.2-1B-Instruct

# Inspect all available flags for the render component
vllm launch render --help=all
```

現在の launch コンポーネントのリファレンスは [vllm launch render](./launch/render.md) を参照してください。

## chat { #chat }

起動中の API サーバー経由でチャットの応答を生成します。

```bash
# Directly connect to localhost API without arguments
vllm chat

# Specify API url
vllm chat --url http://{vllm-serve-host}:{vllm-serve-port}/v1

# Quick chat with a single prompt
vllm chat --quick "hi"

# Print TTFT and throughput statistics after each response
vllm chat --stats
```

指定できる引数の完全な一覧は [vllm chat](./chat.md) を参照してください。

## complete { #complete }

起動中の API サーバー経由で、与えられたプロンプトに対するテキスト補完を生成します。

```bash
# Directly connect to localhost API without arguments
vllm complete

# Specify API url
vllm complete --url http://{vllm-serve-host}:{vllm-serve-port}/v1

# Quick complete with a single prompt
vllm complete --quick "The future of AI is"

# Print TTFT and throughput statistics after each response
vllm complete --stats
```

指定できる引数の完全な一覧は [vllm complete](./complete.md) を参照してください。

## bench { #bench }

レイテンシ、オンラインサービングのスループット、オフライン推論のスループットのベンチマークを実行します。

ベンチマークのコマンドを使うには、`pip install vllm[bench]` で追加の依存関係をインストールしてください。

利用できるコマンド:

```bash
vllm bench {latency, serve, throughput}
```

### latency { #latency }

1 バッチ分のリクエストのレイテンシを計測します。

```bash
vllm bench latency \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy
```

指定できる引数の完全な一覧は [vllm bench latency](./bench/latency.md) を参照してください。

### serve { #serve }

オンラインサービングのスループットを計測します。

```bash
vllm bench serve \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --host server-host \
    --port server-port \
    --random-input-len 32 \
    --random-output-len 4  \
    --num-prompts  5
```

指定できる引数の完全な一覧は [vllm bench serve](./bench/serve.md) を参照してください。

### throughput { #throughput }

オフライン推論のスループットを計測します。

```bash
vllm bench throughput \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy
```

指定できる引数の完全な一覧は [vllm bench throughput](./bench/throughput.md) を参照してください。

## collect-env { #collect-env }

環境情報の収集を開始します。

```bash
vllm collect-env
```

## run-batch { #run-batch }

プロンプトをバッチで実行し、結果をファイルに書き出します。

ローカルのファイルを使う場合:

```bash
vllm run-batch \
    -i features/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

リモートのファイルを使う場合:

```bash
vllm run-batch \
    -i https://raw.githubusercontent.com/vllm-project/vllm/main/examples/features/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

指定できる引数の完全な一覧は [vllm run-batch](./run-batch.md) を参照してください。

## さらに詳しく { #more-help }

各サブコマンドの詳細なオプションは次で確認できます。

```bash
vllm <subcommand> --help
```
