# MoRIIOConnector 利用ガイド { #moriioconnector-usage-guide }

`MoRIIOConnector` は、PD 分離のデプロイにおける KV キャッシュ転送に使う高性能な KV コネクタです。きわめて低いオーバーヘッドで P2P 通信を行う ROCm の [MoRI-IO](https://github.com/rocm/mori) 通信ライブラリをもとに構築されています。

## 前提条件 { #prerequisites }

### インストール { #installation }

**Docker:** MoRI は ROCm 版の公式 vLLM イメージ `vllm/vllm-openai-rocm:nightly` に同梱されています。

**手動インストール:** MoRI の wheel は次のコマンドでインストールできます。

```bash
pip install amd_mori
```

詳細は [Dockerfile.rocm_base](../../docker/Dockerfile.rocm_base) を、ソースから MoRI をビルドする手順は [MoRI 公式リポジトリ](https://github.com/rocm/mori)を参照してください。

適切な NIC のユーザー空間ライブラリのインストール手順については、[NIC のユーザー空間ライブラリのインストール](#appendix-installing-nic-userspace-libraries)を参照してください。

## 基本的な使い方（単一ホスト） { #basic-usage-single-host }

まずプロキシを起動してください。プロデューサーとコンシューマーのインスタンスは、プロキシに到達できるまで登録をリトライします。

### プロデューサー（プレフィル側）の設定 { #producer-prefiller-configuration }

KV キャッシュを生成するプレフィルインスタンスを起動します。

```bash
# Prefill instance (GPU 0-3) 
export VLLM_ROCM_USE_AITER=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3
 
vllm serve Qwen/Qwen3-235B-A22B-FP8 \
  -tp 4 \
  --port 20005 \
  --gpu-memory-utilization 0.9 \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "proxy_ip": "127.0.0.1",
      "proxy_ping_port": "36367",
      "http_port": "20005",
      "handshake_port": "6301",
      "notify_port": "6105"
    }
  }'
```

### コンシューマー（デコード側）の設定 { #consumer-decoder-configuration }

KV キャッシュを消費するデコードインスタンスを起動します。

```bash
# Decode instance (GPU 4-7)
export VLLM_ROCM_USE_AITER=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HIP_VISIBLE_DEVICES=4,5,6,7

vllm serve Qwen/Qwen3-235B-A22B-FP8 \
  -tp 4 \
  --port 40005 \
  --gpu-memory-utilization 0.9 \
  --kv-transfer-config '{
    "kv_connector": "MoRIIOConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
      "proxy_ip": "127.0.0.1",
      "http_port": "40005",
      "proxy_ping_port": "36367",
      "handshake_port": "7301",
      "notify_port": "7501"
    }
  }'
```

### プロキシサーバー { #proxy-server }

プロキシはプロデューサーとコンシューマーのインスタンスの前段に立ち、受け取ったリクエストをそれらへルーティングします。推奨されるプロキシは `vllm-router` で、手動インストールするか Docker コンテナとして実行できます。以下のポート `36367` は、各 vLLM インスタンスで設定した `proxy_ping_port` である点に注意してください。

**Docker:**

```bash
docker run \
  --network host \
  vllm/vllm-router:nightly \
  vllm-router \
  --vllm-pd-disaggregation \
  --kv-connector moriio \
  --vllm-discovery-address "0.0.0.0:36367"
```

**手動インストール:**

```bash
pip install vllm-router
vllm-router \
  --vllm-pd-disaggregation \
  --kv-connector moriio \
  --vllm-discovery-address "0.0.0.0:36367"
```

あるいは、vLLM に同梱されているリファレンス実装のプロキシを使うこともできます。

```bash
cd <path_to>/vllm
pip install quart aiohttp msgpack
python examples/disaggregated/disaggregated_serving/moriio_toy_proxy_server.py
```

## 設定 { #configuration }

このコネクタは、アプリケーションレベルとトランスポートレベルの 2 段階で設定します。

### アプリケーションレベルの設定 { #application-level-configuration }

**モード:** MoRI には WRITE モードと READ モードの 2 つの動作モードがあります。

- WRITE モードでは、プロデューサーが層ごとに、計算した KV ブロックをコンシューマーのメモリへ能動的にプッシュします。
- READ モードでは、ブロックの準備完了が通知され次第、コンシューマーがプロデューサーからまとめて KV ブロックをプルします。

既定では WRITE モードが使われます。READ モードにするには `--kv-transfer-config.kv_connector_extra_config.read_mode true` を設定します。

**コントロールプレーンの設定:** MoRI は KV のバイト列を RDMA / xGMI で転送しますが、プロデューサーとコンシューマーはハンドシェイク、ブロック ID の交換、死活監視、完了通知のために帯域外の TCP チャンネルも必要とします。これらのキーは `kv_connector_extra_config` の下に置きます。

- `proxy_ip`: プレフィル側とデコード側の前段に立つ分離用プロキシ / ルーターの IP アドレス。各 vLLM インスタンスは、自身を登録しハートビートを送るためにこれを使い、プロキシは受け取ったリクエストのルーティング先を把握します。
- `proxy_ping_port`: `proxy_ip` 上でプロキシがインスタンスのハートビートと登録メッセージを待ち受ける TCP ポート。停止した vLLM インスタンスの検出と、ルーティングテーブルの鮮度維持に使われます。
- `http_port`: この vLLM インスタンスが OpenAI 互換 API を公開する HTTP ポート。プロキシはこのポートを登録し、インスタンスを選んだあとユーザーのリクエストをこのポートへ転送します。
- `handshake_port`: プレフィル側とデコード側の間で 1 回だけ行う MoRI エンジンのハンドシェイクに使う TCP ポート。KV 転送を行う前に、両者はここで RDMA のエンジンディスクリプタを交換します。
- `notify_port`: プレフィル側とデコード側の間の制御・同期メッセージに使う TCP ポート。2 つのモードで用途が異なります。
    - WRITE モード: **ブロックの割り当て:** デコード側が自分のブロック ID をプレフィル側に通知し、プレフィル側が計算済みの KV ブロックをデコードインスタンスの正しい場所へプッシュできるようにします。**完了通知:** すべてのブロックの転送が終わると、プレフィル側はデコード側に、そのブロックを安全に使えることを通知します。
    - READ モード: **完了通知:** デコード側がプレフィル側からすべてのブロックを読み終えると、プレフィル側に通知し、KV キャッシュのブロックを解放できるようにします。

!!! note
    `notify_port` は *ベース* ポートとして使われます。インスタンス内の各 (DP ランク, TP ランク) の組は、
    ランクにもとづくオフセットを足した `notify_port + offset` を使います。`notify_port` から始まる
    範囲がホスト上で空いていることを確認してください。

### トランスポートの設定 { #transport-configuration }

MoRI には RDMA と xGMI の 2 つのトランスポートバックエンドがあります。バックエンドは `--kv-transfer-config.kv_connector_extra_config.backend $BACKEND` で選択でき、`$BACKEND` には `rdma` または `xgmi` を指定します。既定のバックエンドは RDMA で、複数ノードのデプロイではこちらを使うべきです。

各バックエンドの設定オプションは次のとおりです。

#### RDMA バックエンド { #rdma-backend }

- `qp_per_transfer`: 1 回の転送に使う RDMA の Queue Pair（QP）の数。QP を増やすと 1 回の転送を複数の QP にストライピングして NIC の並行度を高められますが、RDMA のリソース消費は増えます。
- `post_batch_size`: 1 回の `ibv_post_send` のドアベルにまとめる RDMA の Work Request（WR）の数。既定は -1 で、バックエンドの既定値を意味します。バッチを大きくすると WR あたりの発行オーバーヘッドが減ります。
- `num_workers`: 転送の発行と完了ポーリングのために MoRI が使うワーカースレッドの数。

上級者は、`MORI_IO_QP_MAX_SEND_WR`、`MORI_IO_QP_MAX_CQE` などの環境変数で MoRI 自体を設定することもできます。これらは MoRI ライブラリの変数であり、vLLM 自身の `VLLM_MORIIO_*` の設定とは別物です。詳細は [MoRI のリポジトリ](https://github.com/rocm/mori)を参照してください。

#### xGMI バックエンド { #xgmi-backend }

プレフィル側とデコード側が同じ物理ホストで動作する場合は xGMI を使ってください。転送が AMD の GPU ファブリックを通り、NIC を完全にバイパスします。現時点では MoRI 固有の環境変数でのみ設定できます。[MoRI のリポジトリ](https://github.com/rocm/mori)を参照してください。

## 複数ノードでのデプロイ { #multi-node-deployment }

次の例は、2 ノードで 1P1D のデプロイを行う方法を示します。プロキシはプレフィルインスタンスと同じノードで動かします。

### 両方のノードで { #on-both-nodes }

```bash
# Set on both nodes before running any command
export PREFILL_IP=<node1-ip>
export DECODE_IP=<node2-ip>
```

### ノード 1 で { #on-node-1 }

まず[プロキシサーバー](#proxy-server)の説明に従ってプロキシを起動し、続いてプレフィルインスタンスを起動します。

```bash
docker run \
  --name moriio-prefill \
  --init --network host --ipc host --privileged \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size 256G \
  --group-add video --group-add render \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  -e VLLM_ROCM_USE_AITER=1 \
  vllm/vllm-openai-rocm:nightly \
  deepseek-ai/DeepSeek-R1-0528 \
    --port 8100 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config '{
      "kv_connector": "MoRIIOConnector",
      "kv_role": "kv_producer",
      "kv_connector_extra_config": {
        "proxy_ip": "'"${PREFILL_IP}"'",
        "proxy_ping_port": "36367",
        "http_port": "8100",
        "handshake_port": "6301",
        "notify_port": "61005"
      }
    }'
```

### ノード 2 で { #on-node-2 }

デコードインスタンス:

```bash
docker run \
  --name moriio-decode \
  --init --network host --ipc host --privileged \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size 256G \
  --group-add video --group-add render \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  -e VLLM_ROCM_USE_AITER=1 \
  vllm/vllm-openai-rocm:nightly \
  deepseek-ai/DeepSeek-R1-0528 \
    --port 8200 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --enable-expert-parallel \
    --kv-transfer-config '{
      "kv_connector": "MoRIIOConnector",
      "kv_role": "kv_consumer",
      "kv_connector_extra_config": {
        "proxy_ip": "'"${PREFILL_IP}"'",
        "proxy_ping_port": "36367",
        "http_port": "8200",
        "handshake_port": "6301",
        "notify_port": "61005"
      }
    }'
```

## トラブルシューティング { #troubleshooting }

### `availDevices.size() > 0` のアサーション失敗 { #availdevicessize-0-assertion-failure }

**症状:** 次のログを出して vLLM の起動が失敗します。

```bash
libibverbs: Warning: Driver bnxt_re does not support the kernel ABI of 6 (supports 1 to 1) for device /sys/class/infiniband/rdma4
...
ker: /app/mori/src/io/rdma/backend_impl.cpp: mori::io::RdmaManager::RdmaManager(const RdmaBackendConfig, application::RdmaContext *): Assertion `availDevices.size() > 0' failed.
```

**対処:** インストールされている RDMA のユーザー空間ライブラリが、ホストにインストールされているドライバとファームウェアのバージョンと一致していません。RDMA のカーネルモジュールとファームウェアのバージョンに対応する NIC のユーザー空間ライブラリをインストールする必要があります。詳細は [NIC のユーザー空間ライブラリのインストール](#appendix-installing-nic-userspace-libraries)を参照してください。

## 付録: NIC のユーザー空間ライブラリのインストール { #appendix-installing-nic-userspace-libraries }

RDMA で MoRI を動かすには、対応するカーネルモジュールとファームウェアのバージョンに一致する RDMA のユーザー空間ライブラリが環境にインストールされている必要があります。

公式イメージ `vllm/vllm-openai-rocm:nightly` には、次の NIC とカーネルモジュールのバージョン向けのユーザー空間ライブラリがあらかじめインストールされています。

- AINIC（AMD Pensando Pollara）: バージョン `1.117.3-hydra`、`ioinic-dkms=25.11.1.001` で検証済み
- Thor2（Broadcom）: バージョン `235.2.86.0`、`bnxt-en-dkms=1.10.3.235.2.86.0`、`bnxt-re-dkms=235.2.86.0` で検証済み

詳細は [Dockerfile.rocm](../../docker/Dockerfile.rocm) を参照してください。上記以外の NIC、カーネルモジュール、ファームウェアを使っている場合は、各ベンダーのインストール手順を参照してください。

## さらに読む { #further-reading }

- [Next-Level Inference: Why Your Single-Node vLLM Setup Needs Prefill-Decode Disaggregation](https://vllm.ai/blog/2026-04-07-moriio-kv-connector)
