# 並列化とスケーリング { #parallelism-and-scaling }

## 単一モデルレプリカの分散推論戦略 { #distributed-inference-strategies-for-a-single-model-replica }

単一モデルレプリカの分散推論戦略は、次の指針で選びます。

- **単一 GPU（分散推論なし）:** モデルが 1 台の GPU に収まるなら、分散推論はおそらく不要です。その GPU で推論してください。
- **単一ノード・複数 GPU（テンソル並列）:** モデルが 1 台の GPU には大きすぎるが、複数 GPU を持つ 1 ノードには収まる場合は*テンソル並列*を使います。たとえば GPU 4 台のノードでは `tensor_parallel_size=4` を設定します。
- **複数ノード・複数 GPU（テンソル並列＋パイプライン並列）:** モデルが 1 ノードに収まらない場合は、*テンソル並列*と*パイプライン並列*を組み合わせます。`tensor_parallel_size` にノードあたりの GPU 数、`pipeline_parallel_size` にノード数を設定します。たとえば 1 ノード 8 GPU × 2 ノードなら `tensor_parallel_size=8`、`pipeline_parallel_size=2` とします。

モデルに十分な GPU メモリが確保できるまで、GPU 数とノード数を増やしてください。`tensor_parallel_size` にはノードあたりの GPU 数、`pipeline_parallel_size` にはノード数を設定します。

モデルが収まるだけのリソースを用意したら `vllm` を実行し、次のようなログメッセージを確認します。

```text
INFO 07-23 13:56:04 [kv_cache_utils.py:775] GPU KV cache size: 643,232 tokens
INFO 07-23 13:56:04 [kv_cache_utils.py:779] Maximum concurrency for 40,960 tokens per request: 15.70x
```

`GPU KV cache size` の行は、GPU の KV キャッシュに一度に保存できるトークンの総数を示します。`Maximum concurrency` の行は、各リクエストが指定されたトークン数（上の例では 40,960）を必要とする場合に同時に処理できるリクエスト数の目安です。リクエストあたりのトークン数は、モデル設定の最大系列長 `ModelConfig.max_model_len` から取られます。これらの数値が必要なスループットに届かない場合は、クラスタに GPU やノードを追加してください。

!!! note "特殊なケース: GPU 数で均等に割り切れない場合"
    モデルが 1 ノードに収まるものの、GPU 数でモデルサイズを均等に割り切れない場合は、パイプライン並列を有効にしてください。層単位で分割するため不均等な分割にも対応できます。この場合は `tensor_parallel_size=1` とし、`pipeline_parallel_size` に GPU 数を設定します。また、ノードの GPU が NVLINK で接続されていない場合（L40S など）は、テンソル並列よりパイプライン並列を使うほうがスループットが高く、通信のオーバーヘッドも小さくなります。

### *Mixture of Experts*（*MoE*）モデルの分散サービング { #distributed-serving-of-mixture-of-experts-moe-models }

エキスパート層に別の並列化戦略を使い、エキスパートが本来持つ並列性を活かすと有利なことがよくあります。vLLM は、データ並列の Attention と、エキスパート並列またはテンソル並列の MoE 層を組み合わせた大規模なデプロイをサポートしています。詳細は[データ並列のデプロイ](data_parallel_deployment.md)を参照してください。

## 単一ノードでのデプロイ { #single-node-deployment }

vLLM は、テンソル並列とパイプライン並列による分散推論・サービングをサポートしています。実装には [Megatron-LM のテンソル並列アルゴリズム](https://arxiv.org/pdf/1909.08053.pdf)が含まれます。

既定の分散ランタイムは、複数ノードの推論では [Ray](https://github.com/ray-project/ray)、単一ノードの推論では Python 標準の `multiprocessing` です。`LLM` クラスの `distributed_executor_backend`、または API サーバーの `--distributed-executor-backend` で上書きできます。`multiprocessing` なら `mp`、Ray なら `ray` を指定します。

複数 GPU で推論するには、`LLM` クラスの `tensor_parallel_size` に使いたい GPU 数を設定します。たとえば 4 台の GPU で推論する場合:

```python
from vllm import LLM
llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
output = llm.generate("San Francisco is a")
```

複数 GPU でサービングするには、サーバー起動時に `--tensor-parallel-size` を指定します。たとえば 4 台の GPU で API サーバーを動かす場合:

```bash
vllm serve facebook/opt-13b \
     --tensor-parallel-size 4
```

パイプライン並列を有効にするには `--pipeline-parallel-size` を追加します。たとえば 8 台の GPU でパイプライン並列とテンソル並列を併用して API サーバーを動かす場合:

```bash
# Eight GPUs total
vllm serve gpt2 \
     --tensor-parallel-size 4 \
     --pipeline-parallel-size 2
```

## 複数ノードでのデプロイ { #multi-node-deployment }

1 ノードの GPU ではモデルを保持できない場合は、複数ノードに vLLM をデプロイします。モデルのパスや Python パッケージを含め、すべてのノードで実行環境が同一になるようにしてください。環境を揃えやすく、ホスト間の差異も隠せるため、コンテナイメージの利用を推奨します。

### Ray とは { #what-is-ray }

Ray は Python プログラムをスケールさせるための分散コンピューティングのフレームワークです。複数ノードの vLLM デプロイでは、Ray をランタイムエンジンとして利用できます。

vLLM は Ray を使って、複数ノードにまたがるタスクの分散実行を管理し、どこで実行するかを制御します。

Ray は、vLLM をエンジンとして利用できる大規模な[オフラインバッチ推論](https://docs.ray.io/en/latest/data/working-with-llms.html)や[オンラインサービング](https://docs.ray.io/en/latest/serve/llm)の高レベル API も提供しています。これらの API は、vLLM のワークロードに本番品質の耐障害性・スケーリング・分散環境の可観測性を追加します。

Ray は任意の依存パッケージです。Ray ベースの実行を使う前に、明示的にインストールしてください。

```bash
pip install "ray[cgraph]"
```

詳細は [Ray のドキュメント](https://docs.ray.io/en/latest/index.html)（英語）を参照してください。

### コンテナによる Ray クラスタの構築 { #ray-cluster-setup-with-containers }

補助スクリプト [examples/ray_serving/run_cluster.sh](../../examples/ray_serving/run_cluster.sh) は、各ノードでコンテナを起動し Ray を初期化します。既定では管理者権限なしで Docker を実行するため、プロファイリングやトレース時に GPU のパフォーマンスカウンタにアクセスできません。管理者権限を有効にするには、Docker コマンドに `--cap-add=CAP_SYS_ADMIN` を追加してください。

1 台をヘッドノードとして選び、次を実行します。

```bash
bash run_cluster.sh \
                vllm/vllm-openai \
                <HEAD_NODE_IP> \
                --head \
                /path/to/the/huggingface/home/in/this/node \
                -e VLLM_HOST_IP=<HEAD_NODE_IP>
```

各ワーカーノードでは次を実行します。

```bash
bash run_cluster.sh \
                vllm/vllm-openai \
                <HEAD_NODE_IP> \
                --worker \
                /path/to/the/huggingface/home/in/this/node \
                -e VLLM_HOST_IP=<WORKER_NODE_IP>
```

`VLLM_HOST_IP` はワーカーごとに異なる点に注意してください。これらのコマンドを実行したシェルは開いたままにしてください。閉じるとクラスタが終了します。すべてのノードが IP アドレスで相互に通信できることを確認してください。

!!! warning "ネットワークのセキュリティ"
    セキュリティのため、`VLLM_HOST_IP` にはプライベートなネットワークセグメントのアドレスを設定してください。このネットワーク上の通信は暗号化されておらず、攻撃者がネットワークにアクセスできると任意コード実行に悪用されうる形式でデータをやり取りします。信頼できない相手がこのネットワークに到達できないようにしてください。

いずれかのノードでコンテナに入り、`ray status` と `ray list nodes` を実行して、Ray が想定どおりの数のノードと GPU を認識していることを確認してください。

!!! tip
    KubeRay を使って Ray クラスタを構築することもできます。詳細は [KubeRay の vLLM ドキュメント](https://docs.ray.io/en/latest/cluster/kubernetes/examples/rayserve-llm-example.html)（英語）を参照してください。

### Ray クラスタ上での vLLM の実行 { #running-vllm-on-a-ray-cluster }

!!! tip
    Ray をコンテナ内で動かしている場合、このガイドの以降のコマンドはホストではなく*コンテナ内*で実行してください。コンテナ内でシェルを開くには、ノードに接続して `docker exec -it <container_name> /bin/bash` を使います。

Ray クラスタが動き出したら、単一ノードの場合と同じように vLLM を使えます。クラスタ全体のリソースが vLLM から見えるため、1 台のノードで `vllm` コマンドを 1 回実行するだけで十分です。

一般的には、テンソル並列サイズを各ノードの GPU 数、パイプライン並列サイズをノード数に設定します。たとえば 2 ノードに 16 台の GPU がある場合（1 ノードあたり 8 台）、テンソル並列サイズを 8、パイプライン並列サイズを 2 にします。

```bash
vllm serve /path/to/the/model/in/the/container \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray
```

あるいは、`tensor_parallel_size` にクラスタ全体の GPU 数を設定することもできます。

```bash
vllm serve /path/to/the/model/in/the/container \
     --tensor-parallel-size 16 \
     --distributed-executor-backend ray
```

### multiprocessing による vLLM の実行 { #running-vllm-with-multiprocessing }

複数ノードの vLLM デプロイでは、Ray のほかに `multiprocessing` をランタイムエンジンとして使うこともできます。以下は、2 ノード（1 ノードあたり 8 GPU）に `tp_size=8`、`pp_size=2` でモデルをデプロイする例です。

1 台をヘッドノードとして選び、次を実行します。

```bash
vllm serve /path/to/the/model/in/the/container \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank 0 \
  --master-addr <HEAD_NODE_IP>
```

もう一方のワーカーノードでは次を実行します。

```bash
vllm serve /path/to/the/model/in/the/container \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank 1 \
  --master-addr <HEAD_NODE_IP> --headless
```

## テンソル並列のためのネットワーク通信の最適化 { #optimizing-network-communication-for-tensor-parallelism }

テンソル並列を効率よく動かすには、ノード間の高速な通信が必要です。InfiniBand のような高速ネットワークアダプタが望ましいです。
InfiniBand を使うようクラスタを構成するには、補助スクリプト
[examples/ray_serving/run_cluster.sh](../../examples/ray_serving/run_cluster.sh) に `--privileged -e NCCL_IB_HCA=mlx5` のような引数を追加してください。
必要なフラグの詳細はシステム管理者に問い合わせてください。

## GPUDirect RDMA を有効にする { #enabling-gpudirect-rdma }

GPUDirect RDMA（Remote Direct Memory Access）は、ネットワークアダプタが CPU とシステムメモリを介さずに GPU メモリへ直接アクセスできる NVIDIA の技術です。この直接アクセスによりレイテンシと CPU のオーバーヘッドが減るため、ノードをまたぐ GPU 間の大きなデータ転送に有効です。

vLLM で GPUDirect RDMA を有効にするには、次の設定を行います。

- `IPC_LOCK` のセキュリティコンテキスト: メモリページをロックしてディスクへのスワップを防ぐため、コンテナのセキュリティコンテキストに `IPC_LOCK` ケーパビリティを追加します。
- `/dev/shm` による共有メモリ: プロセス間通信 (IPC) 用の共有メモリを確保するため、Pod の spec で `/dev/shm` をマウントします。

Docker を使う場合は、次のようにコンテナを設定します。

```bash
docker run --gpus all \
    --ipc=host \
    --shm-size=16G \
    -v /dev/shm:/dev/shm \
    vllm/vllm-openai
```

Kubernetes を使う場合は、次のように Pod の spec を設定します。

```yaml
...
spec:
  containers:
    - name: vllm
      image: vllm/vllm-openai
      securityContext:
        capabilities:
          add: ["IPC_LOCK"]
      volumeMounts:
        - mountPath: /dev/shm
          name: dshm
      resources:
        limits:
          nvidia.com/gpu: 8
        requests:
          nvidia.com/gpu: 8
  volumes:
    - name: dshm
      emptyDir:
        medium: Memory
...
```

!!! tip "GPUDirect RDMA の動作確認"
    InfiniBand カードが GPUDirect RDMA を使っているか確認するには、NCCL の詳細ログを有効にして vLLM を実行します: `NCCL_DEBUG=TRACE vllm serve ...`。

    そのうえで、NCCL のバージョンと使用されたネットワークを確認します。

    - ログに `[send] via NET/IB/GDRDMA` があれば、NCCL は GPUDirect RDMA 付きの InfiniBand を使っており、効率的です。
    - ログに `[send] via NET/Socket` があれば、NCCL は生の TCP ソケットを使っており、ノードをまたぐテンソル並列には効率的ではありません。 

!!! tip "Hugging Face のモデルを事前にダウンロードする"
    Hugging Face のモデルを使う場合は、vLLM を起動する前にモデルをダウンロードしておくことを推奨します。すべてのノードで同じパスにダウンロードするか、全ノードからアクセスできる分散ファイルシステムに置いてください。そのうえで、リポジトリ ID の代わりにモデルのパスを指定します。そうしない場合は、`run_cluster.sh` に `-e HF_TOKEN=<TOKEN>` を追加して Hugging Face のトークンを渡してください。

## 分散デプロイのトラブルシューティング { #troubleshooting-distributed-deployments }

分散環境のデバッグについては[分散デプロイのトラブルシューティング](distributed_troubleshooting.md)を参照してください。
