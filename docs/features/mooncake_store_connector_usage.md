# MooncakeStoreConnector 利用ガイド { #mooncakestoreconnector-usage-guide }

MooncakeStoreConnector は、[MooncakeDistributedStore](https://github.com/kvcache-ai/Mooncake) を共有の KV キャッシュプールとして使う KV キャッシュコネクタです。プレフィル側とデコード側の間で直接 P2P の KV 転送を行う `MooncakeConnector` とは異なり、MooncakeStoreConnector は外部の分散ストアへの KV キャッシュのオフロードを可能にし、次をサポートします。

- **CPU / ディスクへのオフロード**: Mooncake の transfer engine を通じて CPU メモリやディスクへオフロードし、実効的な KV キャッシュ容量を拡張します。
- **インスタンスをまたいだプレフィックスキャッシュ**: ハッシュベースの重複排除により、複数の vLLM インスタンスがストアを通じてキャッシュ済みの KV ブロックを共有できます。
- **単一ノードと複数ノードのデプロイ**: 単独の KV キャッシュ拡張としても、プレフィル / デコード分離構成でも動作します。

## 前提条件 { #prerequisites }

### Mooncake のインストール { #install-mooncake }

pip で mooncake をインストールします。

```bash
uv pip install mooncake-transfer-engine
```

インストール方法の詳細やソースからのビルドについては、[Mooncake 公式リポジトリ](https://github.com/kvcache-ai/Mooncake)を参照してください。

### Mooncake マスターサーバーの起動 { #start-the-mooncake-master-server }

Mooncake のマスターはメタデータを管理し、分散ストアを調整します。vLLM を起動する前にこれを起動してください。

```bash
mooncake_master --port 50051
```

既定のポート:

- RPC: 50051

複数の vLLM インスタンスが同じマスターサーバーを共有できます。

### Mooncake の設定 { #configure-mooncake }

JSON の設定ファイル（例: `mooncake_config.json`）を作成します。

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": "80GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false
}
```

- `mode`: トポロジの選択です。`"embedded"`（既定、PR-40900 のベースライン）では、各 vLLM ランクがプロセス内で `global_segment_size` 分をプールに提供します。`"standalone-store"` ではランクは純粋なリクエスタとなり、外部の `mooncake_client` プロセスが CPU プールと（必要に応じて）SSD 階層を保有します。
- `protocol`: 最良の性能を得るには `"rdma"` を使ってください。`"tcp"` はフォールバックとして機能します。
- `global_segment_size`: 分散プールに提供する CPU メモリ量（GPU あたり）。`embedded` モードでは `> 0`、`standalone-store` モードでは `0` である必要があります。
- `local_buffer_size`: このノード自身の操作用のプライベートバッファ（GPU あたり）。
- `enable_offload`: `true` の場合、vLLM は DirectIO のステージングバッファを確保し、大きなプレフィルが所有側の SSD 書き込みの上限を超えないようにします。`mooncake_master` および（存在する場合は）外部の `mooncake_client` の `--enable_offload=true` フラグと合わせて設定してください。

設定ファイルのパスは環境変数で指定します。

```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json
```

## 使い方 { #usage }

### 単一ノードでの KV キャッシュのオフロード { #single-node-kv-cache-offloading }

MooncakeStoreConnector を使って KV キャッシュを CPU メモリにオフロードし、実効的なキャッシュサイズを拡張します。

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
```

### プレフィル / デコード分離（XpYd） { #disaggregated-prefill-decode-xpyd }

プレフィル / デコード分離モードでは、`MultiConnector` を使って `MooncakeConnector`（P2P の KV 転送）と `MooncakeStoreConnector`（共有 KV キャッシュプール）を組み合わせます。これにより、プレフィル側とデコード側の直接的な P2P 転送と、分散ストアを介したインスタンス横断のプレフィックスキャッシュ共有の両方が可能になります。
**プレフィルノード:**

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
VLLM_MOONCAKE_BOOTSTRAP_PORT=50052 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8100 \
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": "kv_producer"
                },
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_both"
                }
            ]
        }
    }'
```

**デコードノード:**

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
VLLM_MOONCAKE_BOOTSTRAP_PORT=50053 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8200 \
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": "kv_consumer"
                },
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_consumer"
                }
            ]
        }
    }'
```

**プロキシ:**

プレフィルノードとデコードノードの間でリクエストをルーティングするには、分離用のプロキシが必要です。プロキシは `do_remote_prefill=True` / `do_remote_decode=True` を割り当て、`MooncakeConnector` による P2P 転送を調整します。プロキシのセットアップ方法は [MooncakeConnector 利用ガイド](mooncake_connector_usage.md)を参照してください。

### ディスクへのオフロード { #disk-offloading }

ディスクへのオフロードは、通常 `standalone-store` モードで運用します。外部の `mooncake_client` プロセスが CPU プールと SSD 階層を保有し、各 vLLM ランクは純粋なリクエスタとなります。これにより、ランクごとに SSD プールが重複するのを避けられ、DirectIO の予算管理を 1 つのプロセスに集約できます。

エンドツーエンドのディスクオフロードには、次の 3 点を揃える必要があります。

1. **`mooncake_master`** を `--enable_offload=true` で起動する。
2. **`mooncake_client`**（所有側）を `--enable_offload=true` と、`MOONCAKE_OFFLOAD_FILE_STORAGE_PATH` による SSD のパス指定とともに起動する。
3. **vLLM 側**では JSON 設定ファイルに `"enable_offload": true` を設定する（これはコネクタが読み取る設定であり、環境変数では**ありません**）。

vLLM 側の `mooncake_config.json` の例:

```json
{
  "mode": "standalone-store",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": 0,
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "mlx5_0",
  "enable_offload": true
}
```

このランクをローカルの所有セグメントに向けるには次のように設定します。

```bash
export MOONCAKE_PREFERRED_SEGMENT=127.0.0.1:50053
```

所有側の SSD ディレクトリ、ディスク上の追い出しポリシー、DirectIO のステージングバッファサイズは、`mooncake_client` 側で標準の Mooncake の環境変数（`MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`、`MOONCAKE_BUCKET_EVICTION_POLICY`、`MOONCAKE_USE_URING`、`MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`、`MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` など）で制御します。これらは vLLM の JSON 設定とは独立しています。

## 環境変数 { #environment-variables }

| 変数 | 説明 | 既定値 |
| --- | --- | --- |
| `MOONCAKE_CONFIG_PATH` | Mooncake の JSON 設定ファイルへのパス | （必須） |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | MooncakeConnector の P2P 転送のブートストラップポート（分離モードのみ） | 8998 |
| `MOONCAKE_PREFERRED_SEGMENT` | このランクのレプリカを特定の所有セグメント（`host:port`）に固定します。`standalone-store` モードで使用 | — |
| `MOONCAKE_REQUESTER_LOCAL_HOSTNAME` | vLLM のランクがリクエスタとして Mooncake に登録するホスト名を上書きします。既定はランクの解決済み IP です。 | — |
| `VLLM_MOONCAKE_STORE_TIER_LOG` | `1` の場合、可観測性のためにバッチごとの階層サマリー（メモリ / ディスクのヒット）をログに出力します | 無効 |
| `VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO` | 1 回の `batch_get_into_multi_buffers` 呼び出しでリクエスタが埋める、所有側の DirectIO ステージングバッファの割合。小さくするほど事前分割が保守的になり、往復回数が増えます。 | 0.9 |

## KV 転送の設定 { #kv-transfer-config }

### kv_role の選択肢 { #kv-role-options }

- **kv_producer**: KV キャッシュをプールに保存するインスタンス向け。
- **kv_consumer**: KV キャッシュをプールから読み込むインスタンス向け。
- **kv_both**: KV キャッシュの保存と読み込みの両方を行うインスタンス。単一ノードでの CPU オフロードや、プレフィルインスタンスではこれを使います。

### kv_connector_extra_config { #kv_connector_extra_config }

- `load_async`（bool）: 計算と I/O のオーバーラップを高めるため、非同期の読み込みを有効にします。既定値: `true`。
- `lookup_async`（bool）: 外部のプレフィックスキャッシュの検索をバックグラウンドスレッドで実行し、スケジューラのステップをブロックしないようにします。リクエストは実行中の検索が完了するまで保留され、以降のステップで再開されます。既定値: `false`。
- `enable_cross_layers_blocks`（bool）: 保存操作を減らすため、層をまたぐブロックのパッキングを有効にします。既定値: `false`。
- `lookup_rpc_port`（int）: ZMQ の検索 RPC ソケット用のポートを指定します。既定値: `0`。
- `cache_prefix`（str）: すべてのストアキーの先頭に付加される名前空間です。これにより、別々のデプロイが 1 つの Mooncake マスターを、互いを汚染せずに共有できます。異なる接頭辞を設定したインスタンス同士は、同じプロンプトであっても相手のキャッシュブロックを見ることはありません。プレフィックスキャッシュを共有すべきすべてのインスタンスは、同じ値を使う必要があります。既定値: `""`（接頭辞なし。キーは接頭辞なしの形式とバイト単位で同一です）。

## 注意事項 { #notes }

### プロセス間で再現可能なブロックハッシュ { #reproducible-block-hashes-across-processes }

`MooncakeStoreConnector` は、分散ストアを共有するすべての vLLM プロセスでブロックハッシュが一致することを前提としています。Python は既定でプロセスごとにハッシュのシードをランダム化するため、同一のプロンプトでもプロセスによって異なるブロックハッシュが生成され、プロセス間でのプレフィックスキャッシュのヒットが妨げられることがあります。

ストアを共有するすべてのインスタンス（DP ランク、別々のプレフィル / デコードノード、同じ Mooncake ストアを指すその他の vLLM プロセス）で、固定の `PYTHONHASHSEED` を設定してください。

```bash
PYTHONHASHSEED=0 vllm serve ...
```
