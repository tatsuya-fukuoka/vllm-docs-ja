# Run:ai Model Streamer によるモデルの読み込み { #loading-models-with-runai-model-streamer }

Run:ai Model Streamer は、テンソルを並行して読み込みながら GPU メモリへストリーミングするライブラリです。詳しくは [Run:ai Model Streamer のドキュメント](https://github.com/run-ai/runai-model-streamer/blob/master/docs/README.md)を参照してください。

vLLM は、Run:ai Model Streamer を使って Safetensors 形式の重みを読み込むことをサポートしています。まず、vLLM の RunAI オプション依存パッケージをインストールする必要があります。

```bash
pip3 install vllm[runai]
```

OpenAI 互換サーバーとして実行するには、`--load-format runai_streamer` フラグを追加します。

```bash
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct \
    --load-format runai_streamer
```

AWS S3 のオブジェクトストアからモデルを実行するには次のようにします。

```bash
vllm serve s3://core-llm/Llama-3-8b \
    --load-format runai_streamer
```

Google Cloud Storage からモデルを実行するには次のようにします。

```bash
vllm serve gs://core-llm/Llama-3-8b \
    --load-format runai_streamer
```

Azure Blob Storage からモデルを実行するには次のようにします。

```bash
AZURE_STORAGE_ACCOUNT_NAME=<account> \
vllm serve az://<container>/<model-path> \
    --load-format runai_streamer
```

認証には `DefaultAzureCredential` が使われます。これは `az login`、マネージド ID、環境変数（`AZURE_CLIENT_ID`、`AZURE_TENANT_ID`、`AZURE_CLIENT_SECRET`）などの方式に対応しています。

S3 互換のオブジェクトストアからモデルを実行するには次のようにします。

```bash
RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 \
AWS_EC2_METADATA_DISABLED=true \
AWS_ENDPOINT_URL=https://storage.googleapis.com \
vllm serve s3://core-llm/Llama-3-8b \
    --load-format runai_streamer
```

## 調整可能なパラメータ { #tunable-parameters }

パラメータは `--model-loader-extra-config` で調整できます。

`distributed` は、分散ストリーミングを使うかどうかを制御します。現時点では CUDA と ROCm のデバイスでのみ利用できます。オブジェクトストレージや高スループットのネットワークファイル共有からの読み込み時間を大幅に短縮できます。分散ストリーミングの詳細は[こちら](https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/usage.md#distributed-streaming)を参照してください。

```bash
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct \
    --load-format runai_streamer \
    --model-loader-extra-config '{"distributed":true}'
```

`concurrency` は、ファイルから CPU バッファへテンソルを読み込む際の並行度と OS スレッド数を制御します。S3 から読み込む場合は、ホストが S3 サーバーに対して開くクライアントインスタンスの数になります。

```bash
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct \
    --load-format runai_streamer \
    --model-loader-extra-config '{"concurrency":16}'
```

ファイルからテンソルを読み込む先の CPU メモリバッファのサイズを制御し、その上限を設けることもできます。CPU バッファのメモリ制限の詳細は[こちら](https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/env-vars.md#runai_streamer_memory_limit)を参照してください。

```bash
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct \
    --load-format runai_streamer \
    --model-loader-extra-config '{"memory_limit":5368709120}'
```

!!! note
    調整可能なパラメータや、環境変数で設定できる追加のパラメータについては、
    [環境変数のドキュメント](https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/env-vars.md)を参照してください。

## シャード化されたモデルの読み込み { #sharded-model-loading }

vLLM は、Run:ai Model Streamer を使ってシャード化されたモデルを読み込むこともサポートしています。これは、複数ファイルに分割された大きなモデルで特に役立ちます。この機能を使うには `--load-format runai_streamer_sharded` フラグを指定します。

```bash
vllm serve /path/to/sharded/model --load-format runai_streamer_sharded
```

シャード対応のローダーは、モデルファイルが通常の sharded state ローダーと同じ命名パターン `model-rank-{rank}-part-{part}.safetensors` に従っていることを前提とします。このパターンは `--model-loader-extra-config` の `pattern` パラメータでカスタマイズできます。

```bash
vllm serve /path/to/sharded/model \
    --load-format runai_streamer_sharded \
    --model-loader-extra-config '{"pattern":"custom-model-rank-{rank}-part-{part}.safetensors"}'
```

シャード化されたモデルファイルを作成するには、[examples/features/sharded_state/save_sharded_state_offline.py](../../../examples/features/sharded_state/save_sharded_state_offline.py) のスクリプトを利用できます。このスクリプトは、Run:ai Model Streamer のシャード対応ローダーと互換性のあるシャード形式でモデルを保存する方法を示しています。

シャード対応のローダーは、通常の Run:ai Model Streamer と同じ調整可能なパラメータ（`concurrency` や `memory_limit` を含む）をすべてサポートします。設定方法も同じです。

```bash
vllm serve /path/to/sharded/model \
    --load-format runai_streamer_sharded \
    --model-loader-extra-config '{"concurrency":16, "memory_limit":5368709120}'
```

!!! note
    シャード対応のローダーは、各ワーカーがチェックポイント全体ではなく自分のシャードだけを
    読めばよいテンソル並列やパイプライン並列のモデルで特に効率的です。
