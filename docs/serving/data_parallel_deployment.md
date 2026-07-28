# データ並列のデプロイ { #data-parallel-deployment }

vLLM はデータ並列のデプロイをサポートしています。これは、モデルの重みを別々のインスタンス / GPU に複製し、独立したリクエストのバッチを処理する方式です。

dense モデルと MoE モデルのどちらでも動作します。

MoE モデル、特に DeepSeek のように MLA（Multi-head Latent Attention）を採用したモデルでは、Attention 層にデータ並列を、エキスパート層にエキスパート並列またはテンソル並列（EP / TP）を使うと有利なことがあります。

この場合、データ並列のランクは完全には独立していません。forward の実行タイミングを揃える必要があり、処理すべきリクエスト数が DP ランク数より少ない場合でも、すべてのランクのエキスパート層は forward のたびに同期する必要があります。

既定では、エキスパート層は `DP × TP` のサイズのテンソル並列グループを形成します。代わりにエキスパート並列を使うには、CLI 引数 `--enable-expert-parallel` を指定します（複数ノードの場合はすべてのノードで指定）。EP を有効にしたときの Attention 層とエキスパート層の挙動の違いは[エキスパート並列のデプロイ](expert_parallel_deployment.md)を参照してください。

vLLM では、各 DP ランクは独立した「コアエンジン」プロセスとしてデプロイされ、ZMQ ソケットを通じてフロントエンドのプロセスと通信します。データ並列の Attention はテンソル並列の Attention と組み合わせられ、その場合、各 DP エンジンは設定した TP サイズと同じ数の GPU ごとのワーカープロセスを持ちます。

MoE モデルでは、いずれかのランクでリクエストが処理中のとき、リクエストがスケジュールされていない他のすべてのランクでも空の「ダミー」forward を実行する必要があります。これは、すべてのランクと通信する専用の DP コーディネータープロセスと、N ステップごとに実行される集団通信によって処理され、すべてのランクがアイドルになって停止できるタイミングを判定します。DP と TP を併用する場合、エキスパート層は `DP × TP` のサイズのグループを形成します（既定ではテンソル並列、`--enable-expert-parallel` を設定した場合はエキスパート並列）。

いずれの場合も、DP ランク間でリクエストを負荷分散すると効果的です。オンラインのデプロイでは、各 DP エンジンの状態、特にスケジュール済み・待機中（キュー内）のリクエストや KV キャッシュの状態を考慮することで、負荷分散を最適化できます。各 DP エンジンは独立した KV キャッシュを持つため、プロンプトの振り分けを工夫することでプレフィックスキャッシュの効果を最大化できます。

このドキュメントはオンラインのデプロイ（API サーバーを使う形態）に焦点を当てています。DP + EP はオフライン利用（LLM クラス経由）でもサポートされています。例は [examples/features/data_parallel/data_parallel_offline.py](../../examples/features/data_parallel/data_parallel_offline.py) を参照してください。

オンラインのデプロイには 2 つのモードがあります。内部で負荷分散する自己完結型と、ランクごとのプロセスを外部でデプロイ・負荷分散する形態です。

## 内部での負荷分散 { #internal-load-balancing }

vLLM は、単一の API エンドポイントを公開する「自己完結型」のデータ並列デプロイをサポートしています。

設定は、vllm serve のコマンドライン引数に `--data-parallel-size=4` のように追加するだけです。この場合 GPU が 4 台必要になります。テンソル並列と組み合わせることもでき、たとえば `--data-parallel-size=4 --tensor-parallel-size=2` なら GPU が 8 台必要です。DP デプロイのサイジングでは、`--max-num-seqs` が DP ランクごとに適用される点に注意してください。

1 つのデータ並列デプロイを複数ノードにまたがって動かす場合、ノードごとに異なる `vllm serve` を実行し、そのノードで動かす DP ランクを指定します。この場合も HTTP のエントリポイントは 1 つで、API サーバーは 1 ノードでのみ動作します。ただし、そのノードは必ずしも DP ランクと同居している必要はありません。

次は、GPU 8 台の単一ノードで DP=4、TP=2 を動かす例です。

```bash
vllm serve $MODEL --data-parallel-size 4 --tensor-parallel-size 2
```

次は、DP=4 で、DP ランク 0 と 1 をヘッドノード、ランク 2 と 3 を 2 台目のノードで動かす例です。

```bash
# Node 0  (with ip address 10.99.48.128)
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Node 1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-start-rank 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

次は、DP=4 で、1 台目のノードには API サーバーのみを置き、すべてのエンジンを 2 台目のノードで動かす例です。

```bash
# Node 0  (with ip address 10.99.48.128)
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 0 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Node 1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 4 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

この DP モードは、`--data-parallel-backend=ray` を指定して Ray と組み合わせることもできます。

```bash
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-backend=ray
```

Ray を使う場合、いくつか重要な違いがあります。

- いずれかのノードで 1 回コマンドを実行するだけで、ローカルとリモートのすべての DP ランクが起動します。ノードごとに起動するより便利です
- `--data-parallel-address` を指定する必要はありません。コマンドを実行したノードが `--data-parallel-address` として使われます
- `--data-parallel-rpc-port` を指定する必要はありません
- 1 つの DP グループが複数ノードを必要とする場合（1 つのモデルレプリカを 2 ノード以上で動かす必要がある場合など）は、`VLLM_RAY_DP_PACK_STRATEGY="span"` を設定してください。この場合 `--data-parallel-size-local` は無視され、自動的に決定されます
- リモートの DP ランクは、Ray クラスタのノードのリソースに応じて割り当てられます

現時点では、内部の DP の負荷分散は API サーバーのプロセス内で行われ、各エンジンの実行中キューと待機キューにもとづいて判断されます。将来的には KV キャッシュを考慮したロジックを取り込み、より高度にできる可能性があります。

この方式で大きな DP サイズをデプロイすると、API サーバーのプロセスがボトルネックになることがあります。その場合は、独立したオプションである `--api-server-count`（たとえば `--api-server-count=4`）でスケールアウトできます。これは利用者から見て透過的で、公開される HTTP エンドポイント / ポートは 1 つのままです。この API サーバーのスケールアウトは「内部的」なもので、「ヘッド」ノード内に閉じている点に注意してください。

<figure markdown="1">
![DP Internal LB Diagram](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/dp_internal_lb.png)
</figure>

## ハイブリッドな負荷分散 { #hybrid-load-balancing }

ハイブリッドな負荷分散は、内部方式と外部方式の中間に位置します。各ノードが自身の API サーバーを持ち、そのノードに同居するデータ並列エンジンにのみリクエストを流します。上流のロードバランサー（Ingress コントローラーやトラフィックルーターなど）が、ノードごとのエンドポイントにユーザーのリクエストを振り分けます。

このモードは `--data-parallel-hybrid-lb` で有効にします。各ノードの起動時には全体のデータ並列サイズを指定します。内部での負荷分散との主な違いは次のとおりです。

- 各ノードが担当するランクを把握できるよう、`--data-parallel-size-local` と `--data-parallel-start-rank` を指定する必要があります。
- すべてのノードが API エンドポイントを公開するため、`--headless` とは併用できません。
- ノードごとに、ローカルのランク数に応じて `--api-server-count` をスケールさせます

この構成では、各ノードがスケジューリングの判断をローカルに閉じるため、ノード間のトラフィックが減り、DP サイズが大きい場合でも単一ノードがボトルネックになるのを避けられます。

## 外部での負荷分散 { #external-load-balancing }

特に大規模なデプロイでは、データ並列ランクのオーケストレーションと負荷分散を外部で行うほうが合理的な場合があります。

この場合、各 DP ランクを独自のエンドポイントを持つ別々の vLLM デプロイとして扱い、各サーバーからのリアルタイムなテレメトリを活用しながら外部のルーターが HTTP リクエストを振り分けるほうが便利です。

MoE 以外のモデルでは、各サーバーが完全に独立しているため、これは簡単に実現できます。その場合、`--data-parallel-*` の引数を一切付けずに独立した vLLM インスタンスを起動してください。外部 DP 用の CLI オプションは MoE のデプロイでのみサポートされます。

MoE の DP+EP でも同等の構成をサポートしており、次の CLI 引数で設定できます。

DP ランクが同居している（同じノード / IP アドレスの）場合は既定の RPC ポートが使われますが、HTTP サーバーのポートはランクごとに別の値を指定する必要があります。

```bash
# Rank 0
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 \
                                         --port 8000
# Rank 1
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 \
                                         --port 8001
```

複数ノードの場合は、ランク 0 のアドレスとポートも指定する必要があります。

```bash
# Rank 0  (with ip address 10.99.48.128)
vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Rank 1
vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

この構成でもコーディネータープロセスは動作し、DP ランク 0 のエンジンと同居します。

<figure markdown="1">
![DP External LB Diagram](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/dp_external_lb.png)
</figure>

上の図では、点線の各ボックスが `vllm serve` の個別の起動に対応します。たとえば、それぞれ別の Kubernetes Pod にできます。
