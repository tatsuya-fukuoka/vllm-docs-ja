# エキスパート並列のデプロイ { #expert-parallel-deployment }

vLLM はエキスパート並列（Expert Parallelism、EP）に対応しています。EP を使うと、Mixture-of-Experts（MoE）モデルのエキスパートを別々の GPU に配置でき、局所性・効率・全体のスループットが向上します。

EP は通常、データ並列（Data Parallelism、DP）と組み合わせて使われます。DP は EP なしでも使えますが、EP は DP と併用するほうが効率的です。データ並列については[こちら](data_parallel_deployment.md)を参照してください。

## 前提条件 { #prerequisites }

EP を使う前に、必要な依存関係をインストールする必要があります。これをより簡単にするための作業が進行中です。

1. **DeepEP のインストール**: EP カーネル向けの vLLM のガイド（[こちら](../../tools/ep_kernels)）に従ってホスト環境をセットアップします。
2. **DeepGEMM ライブラリのインストール**: [公式の手順](https://github.com/deepseek-ai/DeepGEMM#installation)に従います。
3. **分離サービングを使う場合**: [`install_gdrcopy.sh`](../../tools/install_gdrcopy.sh) スクリプトを実行して `gdrcopy` をインストールします（例: `install_gdrcopy.sh "${GDRCOPY_OS_VERSION}" "12.8" "x64"`）。利用可能な OS バージョンは[こちら](https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/)で確認できます。

### バックエンドの選び方 { #backend-selection-guide }

vLLM は EP 向けに複数の通信バックエンドを提供しています。`--all2all-backend` で選択します。

| バックエンド | 用途 | 特徴 | 適した場面 |
| ------- | -------- | -------- | -------- |
| `allgather_reducescatter` | 既定のバックエンド | allgather / reducescatter プリミティブを使う標準的な all2all | 汎用。任意の EP+DP 構成で動作します |
| `deepep_high_throughput` | 複数ノードでのプレフィル | 連続レイアウトのグループ化 GEMM。プレフィル向けに最適化 | プレフィルが支配的なワークロード、高スループットが求められる場面 |
| `deepep_low_latency` | 複数ノードでのデコード | CUDA graph 対応、マスク付きレイアウト。デコード向けに最適化 | デコードが支配的なワークロード、低レイテンシが求められる場面 |
| `flashinfer_nvlink_one_sided` | MNNVL システム | 複数ノード NVLink 向けの FlashInfer の片側 A2A 戦略 | 高スループットのワークロード |
| `flashinfer_nvlink_two_sided` | MNNVL システム | 複数ノード NVLink 向けの FlashInfer の両側 A2A 戦略 | ノード間 NVLink を備えたシステム |

## 単一ノードでのデプロイ { #single-node-deployment }

!!! warning
    EP は実験的機能です。引数名や既定値は今後変わる可能性があります。

### 設定 { #configuration }

`--enable-expert-parallel` フラグを指定して EP を有効にします。EP のサイズは次のように自動的に計算されます。

```text
EP_SIZE = TP_SIZE × DP_SIZE
```

各項目の意味は次のとおりです。

- `TP_SIZE`: テンソル並列のサイズ
- `DP_SIZE`: データ並列のサイズ
- `EP_SIZE`: エキスパート並列のサイズ（自動計算されます）

### EP を有効にしたときの層ごとの挙動 { #layer-behavior-with-ep-enabled }

EP を有効にすると、MoE モデル内の層は種別ごとに異なる振る舞いをします。

| 層の種別 | 挙動 | 使われる並列化 |
| ---------- | -------- | ---------------- |
| **エキスパート（MoE）層** | すべての EP ランクに分割配置される | サイズ `TP × DP` のエキスパート並列（EP） |
| **Attention 層** | TP のサイズによって変わる | 下記参照 |

**Attention 層の並列化:**

- **`TP = 1` の場合**: attention の重みはすべての DP ランクに**複製**されます（データ並列）
- **`TP > 1` の場合**: attention の重みは、各 DP グループ内の TP ランクにわたってテンソル並列で**分割**されます

たとえば `TP=2, DP=4`（合計 8 GPU）の場合は次のようになります。

- エキスパート層はサイズ 8 の EP グループを構成し、エキスパートが全 GPU に分散配置されます
- attention 層は 4 つの DP グループそれぞれの内部で TP=2 を使います

!!! note "データ並列デプロイとの主な違い"
    `--enable-expert-parallel` を指定しない場合、MoE 層は dense モデルと同様にテンソル並列（サイズ `TP × DP` の TP グループ）を使います。EP を有効にすると、エキスパート層はエキスパート並列に切り替わり、MoE モデルにとってより高い効率と局所性が得られます。

### コマンド例 { #example-command }

次のコマンドは、テンソル並列 1、（attention の）データ並列 8、エキスパート並列 8 で `DeepSeek-V3-0324` モデルをサービングします。attention の重みは全 GPU に複製され、エキスパートの重みは GPU 間で分割されます。GPU 8 基の H200（または H20）ノードで動作します。H100 の場合は、より小さいモデルを試すか、複数ノードでのデプロイの節を参照してください。

```bash
# Single node EP deployment
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \       # Tensor parallelism across 1 GPU
    --data-parallel-size 8 \         # Data parallelism across 8 processes
    --enable-expert-parallel         # Enable expert parallelism
```

## 複数ノードでのデプロイ { #multi-node-deployment }

複数ノードでのデプロイでは、DeepEP の通信カーネルを 2 つのモードのいずれかで使います（上記の[バックエンドの選び方](#backend-selection-guide)を参照）。

### デプロイ手順 { #deployment-steps }

1. **ノードごとにコマンドを 1 つ実行する** - 各ノードにそれぞれ起動コマンドが必要です
2. **ネットワークを設定する** - IP アドレスとポートの設定が正しいことを確認します
3. **ノードの役割を決める** - 最初のノードがリクエストを処理し、残りのノードは headless モードで動作します

### 例: 2 ノード構成のデプロイ { #example-2-node-deployment }

次の例では、`deepep_low_latency` モードを使って `DeepSeek-V3-0324` を 2 ノードにデプロイします。

```bash
# Node 1 (Primary - handles incoming requests)
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --all2all-backend deepep_low_latency \
    --tensor-parallel-size 1 \               # TP size per node
    --enable-expert-parallel \               # Enable EP
    --data-parallel-size 16 \                # Total DP size across all nodes
    --data-parallel-size-local 8 \           # Local DP size on this node (8 GPUs per node)
    --data-parallel-address 192.168.1.100 \  # Replace with actual IP of Node 1
    --data-parallel-rpc-port 13345 \         # RPC communication port, can be any port as long as reachable by all nodes
    --api-server-count=8                     # Number of API servers for load handling (scaling this out to # local ranks is recommended)

# Node 2 (Secondary - headless mode, no API server)
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --all2all-backend deepep_low_latency \
    --tensor-parallel-size 1 \               # TP size per node
    --enable-expert-parallel \               # Enable EP
    --data-parallel-size 16 \                # Total DP size across all nodes
    --data-parallel-size-local 8 \           # Local DP size on this node
    --data-parallel-start-rank 8 \           # Starting rank offset for this node
    --data-parallel-address 192.168.1.100 \  # IP of primary node (Node 1)
    --data-parallel-rpc-port 13345 \         # Same RPC port as primary
    --headless                               # No API server, worker only
```

### 設定上の重要な注意点 { #key-configuration-notes }

- **headless モード**: 副ノードは `--headless` フラグ付きで動作します。つまり、クライアントからのリクエストはすべて主ノードが処理します
- **ランクの計算**: `--data-parallel-start-rank` には、それ以前のノードのローカル DP サイズの累計を指定します
- **負荷に応じたスケーリング**: リクエスト負荷が高い場合は、主ノードの `--api-server-count` を調整します

### ネットワークの設定 { #network-configuration }

!!! important "InfiniBand クラスタ"
    InfiniBand で接続されたクラスタでは、初期化時のハングを防ぐために次の環境変数を設定してください。
    ```bash
    export GLOO_SOCKET_IFNAME=eth0
    ```
    これにより、torch distributed のグループ探索が初期セットアップ時に InfiniBand ではなく Ethernet を使うようになります。

## エキスパート並列ロードバランサ（EPLB） { #expert-parallel-load-balancer-eplb }

MoE モデルは通常、各エキスパートが同程度の数のトークンを受け取るように学習されますが、実際にはエキスパート間のトークン分布が大きく偏ることがあります。vLLM は、EP ランク間でエキスパートの割り当てを再配置し、エキスパート間の負荷を均す仕組みとして、エキスパート並列ロードバランサ（EPLB）を提供しています。

### 設定 { #configuration_1 }

`--enable-eplb` フラグで EPLB を有効にします。

有効にすると、vLLM は forward pass のたびに負荷統計を収集し、定期的にエキスパートの配置をリバランスします。

### EPLB のパラメータ { #eplb-parameters }

EPLB は `--eplb-config` 引数（JSON 文字列を受け取ります）で設定します。指定できるキーとその説明は次のとおりです。

| パラメータ | 説明 | 既定値 |
| --------- | ----------- | ------- |
| `window_size` | リバランスの判断に使う、追跡対象のエンジンステップ数 | 1000 |
| `step_interval` | リバランスの頻度（N エンジンステップごと） | 3000 |
| `log_balancedness` | 均衡度のメトリクス（エキスパートあたり平均トークン数 ÷ エキスパートあたり最大トークン数）をログ出力するか | `false` |
| `num_redundant_experts` | 均等配分に加えて、EP ランクごとに追加するグローバルエキスパートの数 | `0` |
| `use_async` | レイテンシのオーバーヘッドを減らすため、ノンブロッキングの EPLB を使うか | `true` |
| `policy` | エキスパート並列のロードバランシングに使うポリシーの種類 | `"default"` |
| `communicator` | エキスパートの重み転送に使うバックエンド: `"torch_nccl"`、`"torch_gloo"`、`"pynccl"`、`"nixl"`、または `null`（自動） | `null` |

例:

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --enable-eplb \
  --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2,"log_balancedness":true}'
```

??? tip "JSON ではなく個別の引数で指定したい場合"

    ```bash
    vllm serve Qwen/Qwen3-30B-A3B \
            --enable-eplb \
            --eplb-config.window_size 1000 \
            --eplb-config.step_interval 3000 \
            --eplb-config.num_redundant_experts 2 \
            --eplb-config.log_balancedness true
    ```

### エキスパート配分の計算式 { #expert-distribution-formula }

- **既定**: 各 EP ランクは `NUM_TOTAL_EXPERTS ÷ NUM_EP_RANKS` 個のエキスパートを持ちます
- **冗長エキスパートあり**: 各 EP ランクは `(NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS` 個のエキスパートを持ちます

### メモリ使用量のオーバーヘッド { #memory-footprint-overhead }

EPLB は冗長エキスパートを使うため、その分が GPU メモリに収まる必要があります。したがって、メモリに制約のある環境や KV キャッシュ領域が逼迫している場合には、EPLB は適さないことがあります。

このオーバーヘッドは `NUM_MOE_LAYERS * BYTES_PER_EXPERT * (NUM_TOTAL_EXPERTS + NUM_REDUNDANT_EXPERTS) ÷ NUM_EP_RANKS` に等しくなります。
DeepSeekV3 では、EP ランクあたり冗長エキスパート 1 つで約 `2.4 GB` になります。

### コマンド例 { #example-command_1 }

EPLB を有効にした単一ノードでのデプロイ:

```bash
# Single node with EPLB load balancing
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --tensor-parallel-size 1 \       # Tensor parallelism
    --data-parallel-size 8 \         # Data parallelism
    --enable-expert-parallel \       # Enable EP
    --enable-eplb \                  # Enable load balancer
    --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":2,"log_balancedness":true}'
```

複数ノードでのデプロイでは、これらの EPLB フラグを各ノードのコマンドに追加します。大規模な利用では、人気の高いエキスパートが常に利用可能になるよう `--eplb-config '{"num_redundant_experts":32}'` のように 32 を設定することを推奨します。

## 高度な設定 { #advanced-configuration }

### 性能の最適化 { #performance-optimization }

- **DeepEP カーネル**: `high_throughput` と `low_latency` のカーネルは分離サービング向けに最適化されており、混在ワークロードでは性能が出ない場合があります
- **Dual Batch Overlap**: `--enable-dbo` を使うと all-to-all 通信と計算をオーバーラップできます。詳細は [Dual Batch Overlap](../design/dbo.md) を参照してください。
- **非同期スケジューリング（実験的）**: `--async-scheduling` を試すと、スケジューリングとモデル実行をオーバーラップできます。

### トラブルシューティング { #troubleshooting }

- **`non-zero status: 7 cannot register cq buf`**: Infiniband / RoCE を使う場合、ホスト VM と Pod の両方で `ulimit -l` が "unlimited" になっていることを確認してください。
- **`init failed for transport: IBGDA`**: InfiniBand GDA のカーネルモジュールが不足しています。各 GPU ノードで `tools/ep_kernels/configure_system_drivers.sh` を実行し、再起動してください。`NVSHMEM API called before NVSHMEM initialization has completed` というエラーもこれで解消します。
- **NVSHMEM の peer disconnect**: 通常はネットワーク設定の誤りです。Kubernetes でデプロイしている場合は、Infiniband にアクセスできるよう、すべての Pod が `hostNetwork: true`、`securityContext.privileged: true` で動作していることを確認してください。

### ベンチマーク { #benchmarking }

- シミュレータ用のフラグ `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random` と `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1` を使うと、トークンのルーティングが EP ランク間で均等になります。

## 分離サービング（プレフィル / デコードの分離） { #disaggregated-serving-prefilldecode-split }

TTFT（最初のトークンまでの時間）やトークン間レイテンシに厳密な SLA 保証が求められる本番デプロイでは、分離サービングによってプレフィルとデコードを独立にスケールできます。

### アーキテクチャの概要 { #architecture-overview }

- **プレフィルインスタンス**: 最適なプレフィル性能のために `deepep_high_throughput` バックエンドを使います
- **デコードインスタンス**: デコードレイテンシを最小化するために `deepep_low_latency` バックエンドを使います
- **KV キャッシュの転送**: NIXL やその他の KV コネクタでインスタンス同士を接続します

### セットアップ手順 { #setup-steps }

1. **gdrcopy / ucx / nixl のインストール**: 最大限の性能を得るには、[install_gdrcopy.sh](../../tools/install_gdrcopy.sh) スクリプトを実行して `gdrcopy` をインストールします（例: `install_gdrcopy.sh "${GDRCOPY_OS_VERSION}" "12.8" "x64"`）。利用可能な OS バージョンは[こちら](https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/)で確認できます。`gdrcopy` をインストールしなくても、単に `pip install nixl` するだけで動作しますが、性能は低下します。`nixl` と `ucx` は pip の依存関係としてインストールされます。CUDA 以外のプラットフォームで、CUDA を使わない UCX ビルドとともに nixl をインストールするには、[install_nixl_from_source_ubuntu.py](../../tools/install_nixl_from_source_ubuntu.py) スクリプトを実行します。

2. **両方のインスタンスの設定**: プレフィル側とデコード側の両方に `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}` を追加します。なお、NIXL_Backend を 1 つ以上指定することもできます。例: `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both", "kv_connector_extra_config":{"backends":["UCX", "GDS"]}}'`

3. **クライアント側のオーケストレーション**: 下記のクライアント側スクリプトを使ってプレフィル / デコードの処理を連携させます。ルーティングの仕組みについては現在も開発を進めています。

### クライアント側オーケストレーションの例 { #client-orchestration-example }

```python
from openai import OpenAI
import uuid

try:
    # 1: Set up clients for prefill and decode instances
    openai_api_key = "EMPTY"  # vLLM doesn't require a real API key
    
    # Replace these IP addresses with your actual instance addresses
    prefill_client = OpenAI(
        api_key=openai_api_key,
        base_url="http://192.168.1.100:8000/v1",  # Prefill instance URL
    )
    decode_client = OpenAI(
        api_key=openai_api_key,
        base_url="http://192.168.1.101:8001/v1",  # Decode instance URL  
    )
    
    # Get model name from prefill instance
    models = prefill_client.models.list()
    model = models.data[0].id
    print(f"Using model: {model}")

    # 2: Prefill Phase
    # Generate unique request ID to link prefill and decode operations
    request_id = str(uuid.uuid4())
    print(f"Request ID: {request_id}")
    
    prefill_response = prefill_client.completions.create(
        model=model,
        # Prompt must exceed vLLM's block size (16 tokens) for PD to work
        prompt="Write a detailed explanation of Paged Attention for Transformers works including the management of KV cache for multi-turn conversations",
        max_tokens=1,  # Force prefill-only operation
        extra_body={
            "kv_transfer_params": {
                "do_remote_decode": True,     # Enable remote decode
                "do_remote_prefill": False,   # This is the prefill instance
                "remote_engine_id": None,     # Will be populated by vLLM
                "remote_block_ids": None,     # Will be populated by vLLM
                "remote_host": None,          # Will be populated by vLLM
                "remote_port": None,          # Will be populated by vLLM
            }
        },
        extra_headers={"X-Request-Id": request_id},
    )
    
    print("-" * 50)
    print("✓ Prefill completed successfully")
    print(f"Prefill response: {prefill_response.choices[0].text}")
    
    # 3: Decode Phase
    # Transfer KV cache parameters from prefill to decode instance
    decode_response = decode_client.completions.create(
        model=model,
        prompt="This prompt is ignored during decode",  # Original prompt not needed
        max_tokens=150,  # Generate up to 150 tokens
        extra_body={
            "kv_transfer_params": prefill_response.kv_transfer_params  # Pass KV cache info
        },
        extra_headers={"X-Request-Id": request_id},  # Same request ID
    )
    
    print("-" * 50)
    print("✓ Decode completed successfully")
    print(f"Final response: {decode_response.choices[0].text}")

except Exception as e:
    print(f"❌ Error during disaggregated serving: {e}")
    print("Check that both prefill and decode instances are running and accessible")
```

### ベンチマーク { #benchmarking_1 }

- 分離サービングにおけるデコード側のデプロイをシミュレートするには、`vllm serve` の呼び出しに `--kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both"}'` を渡します。このコネクタは KV キャッシュをランダムな値で埋めるため、デコードを単体でプロファイルできます。

- **CUDAGraph のキャプチャ**: `--compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` を使うと、デコードのみ CUDA graph をキャプチャし、KV キャッシュを節約できます。
