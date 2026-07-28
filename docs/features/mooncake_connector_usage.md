# MooncakeConnector 利用ガイド { #mooncakeconnector-usage-guide }

## Mooncake について { #about-mooncake }

Mooncake は、高速に相互接続された DRAM / SSD リソース上に多階層のキャッシュプールを構築することで、特にオブジェクトストレージが遅い環境における大規模言語モデル（LLM）の推論効率を高めることを目指しています。従来のキャッシュシステムと比べ、Mooncake は（GPUDirect）RDMA 技術を利用してゼロコピーでデータを直接転送しつつ、1 台のマシン上の複数 NIC のリソースを最大限に活用します。

Mooncake の詳細は [Mooncake プロジェクト](https://github.com/kvcache-ai/Mooncake)と [Mooncake のドキュメント](https://kvcache-ai.github.io/Mooncake/)を参照してください。

## 前提条件 { #prerequisites }

### インストール { #installation }

pip で mooncake をインストールします: `uv pip install mooncake-transfer-engine`

インストール方法の詳細は [Mooncake 公式リポジトリ](https://github.com/kvcache-ai/Mooncake)を参照してください。

## 使い方 { #usage }

### プレフィルノード (192.168.0.2) { #prefiller-node-19216802 }

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8010 --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
```

### デコードノード (192.168.0.3) { #decoder-node-19216803 }

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8020 --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

### プロキシ { #proxy }

```bash
python examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py --prefill http://192.168.0.2:8010 --decode http://192.168.0.3:8020
```

これで、ポート 8000 経由でプロキシサーバーにリクエストを送れるようになります。

## 環境変数 { #environment-variables }

- `VLLM_MOONCAKE_BOOTSTRAP_PORT`: Mooncake ブートストラップサーバーのポート
    - 既定値: 8998
    - プレフィルインスタンスでのみ必要
    - headless インスタンスでは、マスターインスタンスと同じ値にする必要があります
    - 各インスタンスは同一ホスト上で一意のポートを使う必要があります。異なるホスト間で同じポート番号を使うのは問題ありません

- `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`: 特定のリクエストについて、プレフィル側の KV キャッシュを自動的に解放するまでのタイムアウト（秒）。（任意）
    - 既定値: 480
    - リクエストが中断され、デコード側がまだプレフィル側に通知していない場合、プレフィルインスタンスはこのタイムアウト後に KV キャッシュのブロックを解放し、無期限に保持し続けるのを防ぎます。

## KV 転送の設定 { #kv-transfer-config }

### kv_role の選択肢 { #kv-role-options }

- **kv_producer**: KV キャッシュを生成するプレフィルインスタンス向け
- **kv_consumer**: プレフィル側から KV キャッシュを受け取るデコードインスタンス向け
- **kv_both**: コネクタがプロデューサーとコンシューマーの両方として動作できる対称的な機能を有効にします。実験的な構成や、役割の区別があらかじめ決まっていないシナリオで柔軟に対応できます。

### kv_connector_extra_config { #kv_connector_extra_config }

- **num_workers**: 1 つのプレフィルワーカーが mooncake で KV キャッシュを転送する際のスレッドプールのサイズ。（既定値 10）
- **mooncake_protocol**: Mooncake コネクタのプロトコル。（既定値 "rdma"）

## サンプルスクリプト / コード { #example-scriptscode }

vLLM リポジトリにある次のサンプルスクリプトを参照してください。

- [run_mooncake_connector.sh](../../examples/disaggregated/mooncake_connector/run_mooncake_connector.sh)
- [mooncake_connector_proxy.py](../../examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py)
