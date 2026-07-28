# 最適化とチューニング { #optimization-and-tuning }

このガイドでは、vLLM V1 の最適化戦略と性能チューニングについて説明します。

!!! tip
    メモリが不足していますか？ メモリ節約の方法は[このガイド](./conserving_memory.md)を参照してください。

## 最適化レベル { #optimization-levels }

vLLM には 4 段階の最適化レベル（`-O0`、`-O1`、`-O2`、`-O3`）があり、起動時間と性能のトレードオフを選べます。

- `-O0`: 最適化なし。起動はもっとも速いが、性能はもっとも低い。
- `-O1`: 高速な最適化。単純なコンパイルと軽量な融合、PIECEWISE の CUDA グラフ。
- `-O2`: 既定の最適化。コンパイル範囲と融合が追加され、FULL_AND_PIECEWISE の CUDA グラフを使用。
- `-O3`: 積極的な最適化。現時点では `-O2` と同じだが、将来的に時間のかかる最適化や実験的な最適化が追加される可能性がある。

詳細は[最適化レベルのドキュメント](../design/optimization_levels.md)を参照してください。

## 起動を速くする { #faster-startup }

最適化レベル以外にも、同じ（モデル・設定・ハードウェア）の組み合わせで繰り返し起動する場合に、最初のトークンまでの時間を短縮する仕組みが 3 つあります。

- **コンパイルキャッシュを再利用する。** vLLM は `torch.compile` の成果物を `VLLM_CACHE_ROOT`（既定は `~/.cache/vllm`）に保存します。このキャッシュディレクトリはマシン間でコピーしたり、コンテナイメージに含めたりできます（[torch.compile の設計ドキュメント](../design/torch_compile.md)を参照）。`VLLM_FORCE_AOT_LOAD=1` を設定すると、キャッシュミス時に黙って再コンパイルする代わりに明示的にエラーになります（モデル・設定・関連する `VLLM_*` 環境変数・torch のビルド・GPU の機種のいずれかが変わるとキャッシュは無効になります）。
- **`--kv-cache-memory` でメモリプロファイリングを省略する。** 起動時、vLLM は現在の割り当てを再現する `--kv-cache-memory` の値をログに出力します。次回の起動でその値を渡すと、メモリプロファイリングの計測と CUDA グラフのメモリ見積もりを省略できます。ただし性能への影響がある点に注意してください。KV キャッシュは計測結果ではなく指定した値ちょうどのサイズになるため、保守的な値ではバッチの同時実行数（ひいてはスループット）が制限され、楽観的すぎる値では割り当て時に失敗します。この値は同じ GPU・同じ初期空きメモリでのみ有効です。ハードウェアや同居プロセスが変わって起動時に OOM になる場合は、このフラグを外して再度プロファイリングしてください。
- **`--enforce-eager` で CUDA グラフなしにサービングする。** コンパイルと CUDA グラフのキャプチャの両方を省略し、起動を最速にします。代わりに定常状態の Decode 性能は落ちます。開発サイクルや、起動時間のうちコンパイル・キャプチャがどれだけ占めるかを測るのに便利です。

## プリエンプション { #preemption }

Transformer は自己回帰的に動作するため、バッチ内のすべてのリクエストを処理するには KV キャッシュの容量が足りなくなることがあります。
その場合、vLLM は他のリクエストのために KV キャッシュを空けようとしてリクエストをプリエンプト（横取り）します。プリエンプトされたリクエストは、
十分な KV キャッシュが再び確保できた時点で再計算されます。これが起きると次のような警告が表示されます。

```text
WARNING 05-09 00:49:33 scheduler.py:1057 Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space. This can affect the end-to-end performance. Increase gpu_memory_utilization or tensor_parallel_size to provide more KV cache memory. total_cumulative_preemption_cnt=1
```

この仕組みはシステムの堅牢性を保ちますが、プリエンプションと再計算はエンドツーエンドのレイテンシに悪影響を与えます。
プリエンプションが頻発する場合は、次の対応を検討してください。

- `gpu_memory_utilization` を上げる。vLLM はこの割合のメモリを使って GPU キャッシュを事前確保します。値を上げると KV キャッシュの容量が増えます。
- `max_num_seqs` または `max_num_batched_tokens` を下げる。バッチ内の同時リクエスト数が減り、必要な KV キャッシュ容量が小さくなります。
- `tensor_parallel_size` を上げる。モデルの重みを GPU 間で分割するため、各 GPU で KV キャッシュに使えるメモリが増えます。ただし値を上げすぎると同期のオーバーヘッドが大きくなります。
- `pipeline_parallel_size` を上げる。モデルの層を GPU 間に分散するため、各 GPU で重みに必要なメモリが減り、間接的に KV キャッシュ用のメモリが増えます。ただし値を上げるとレイテンシが悪化することがあります。

プリエンプトされたリクエスト数は、vLLM が公開する Prometheus メトリクスで監視できます。また `disable_log_stats=False` を設定すると、累計のプリエンプション数をログに出力できます。

vLLM V1 では、既定のプリエンプションモードは `SWAP` ではなく `RECOMPUTE` です。V1 のアーキテクチャでは再計算のほうがオーバーヘッドが小さいためです。

## チャンク化 Prefill { #chunked-prefill }

チャンク化 Prefill を使うと、vLLM は大きな Prefill を小さなチャンクに分けて処理し、Decode のリクエストと同じバッチにまとめられます。計算律速（Prefill）とメモリ律速（Decode）の処理のバランスが良くなるため、スループットとレイテンシの両方が改善します。

V1 では、**可能な限りチャンク化 Prefill が既定で有効**になります。チャンク化 Prefill が有効な場合、スケジューリングポリシーは Decode のリクエストを優先します。まず保留中の Decode をすべてバッチにまとめ、その後に Prefill をスケジュールします。`max_num_batched_tokens` の予算に余裕があれば保留中の Prefill をスケジュールし、`max_num_batched_tokens` に収まらない Prefill は自動的にチャンクに分割します。

このポリシーには 2 つの利点があります。

- Decode のリクエストが優先されるため、トークン間レイテンシ (ITL) と生成の Decode が改善します。
- 計算律速（Prefill）とメモリ律速（Decode）のリクエストを同じバッチに入れることで、GPU の使用効率が上がります。

### チャンク化 Prefill による性能チューニング { #performance-tuning-with-chunked-prefill }

`max_num_batched_tokens` を調整することで性能をチューニングできます。

- 小さい値（例: 2048）では Decode を遅くする Prefill が減るため、ITL が良くなります。
- 大きい値では 1 バッチでより多くの Prefill トークンを処理できるため、最初のトークンまでの時間 (TTFT) が良くなります。
- スループットを最大化したい場合、特に大きな GPU で小さめのモデルを動かすときは `max_num_batched_tokens > 8192` を推奨します。
- `max_num_batched_tokens` が `max_model_len` と同じ場合、V0 の既定のスケジューリングポリシーとほぼ同等になります（Decode を優先する点は異なります）。

!!! warning
    チャンク化 Prefill を無効にした場合、`max_num_batched_tokens` は `max_model_len` より大きくする必要があります。  
    `max_num_batched_tokens < max_model_len` の場合、サーバー起動時に vLLM がクラッシュすることがあります。

```python
from vllm import LLM

# Set max_num_batched_tokens to tune performance
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_num_batched_tokens=16384)
```

詳細は関連論文（<https://arxiv.org/pdf/2401.08671> または <https://arxiv.org/pdf/2308.16369>）を参照してください。

## 並列化の戦略 { #parallelism-strategies }

vLLM は複数の並列化戦略をサポートしており、これらを組み合わせてさまざまなハードウェア構成で性能を最適化できます。

### テンソル並列 (TP) { #tensor-parallelism-tp }

テンソル並列は、各層の内部でモデルのパラメータを複数の GPU に分割します。単一ノード内で大きなモデルを推論する際にもっとも一般的な戦略です。

**使いどころ:**

- モデルが大きすぎて 1 つの GPU に収まらない場合
- GPU あたりのメモリ圧迫を減らし、KV キャッシュの容量を増やしてスループットを上げたい場合

```python
from vllm import LLM

# Split model across 4 GPUs
llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=4)
```

70B パラメータ級など、1 つの GPU に収まらないモデルではテンソル並列が不可欠です。

### パイプライン並列 (PP) { #pipeline-parallelism-pp }

パイプライン並列は、モデルの層を複数の GPU に分散します。各 GPU はモデルの異なる部分を順に処理します。

**使いどころ:**

- テンソル並列を効率的に使い切ったうえで、さらにモデルを分散したい場合やノードをまたぎたい場合
- 非常に深く幅の狭いモデルで、テンソル分割より層の分散のほうが効率的な場合

非常に大きなモデルでは、パイプライン並列とテンソル並列を組み合わせられます。

```python
from vllm import LLM

# Combine pipeline and tensor parallelism
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,
    pipeline_parallel_size=2,
)
```

### エキスパート並列 (EP) { #expert-parallelism-ep }

エキスパート並列は Mixture of Experts (MoE) モデルに特化した並列化で、異なるエキスパートのネットワークを GPU 間に分散します。

**使いどころ:**

- MoE モデル（DeepSeekV3、Qwen3MoE、Llama-4 など）を使う場合
- エキスパートの計算負荷を GPU 間で分散したい場合

`enable_expert_parallel=True` を設定するとエキスパート並列が有効になり、MoE 層ではテンソル並列の代わりにエキスパート並列が使われます。
並列度はテンソル並列に設定した値と同じになります。

### データ並列 (DP) { #data-parallelism-dp }

データ並列は、モデル全体を複数の GPU グループに複製し、異なるリクエストのバッチを並列に処理します。

**使いどころ:**

- モデル全体を複製できるだけの GPU がある場合
- モデルサイズではなくスループットをスケールさせたい場合
- リクエストのバッチ間で分離が有効なマルチユーザー環境

データ並列は他の並列化戦略と組み合わせられ、`data_parallel_size=N` で設定します。
MoE 層は、テンソル並列サイズとデータ並列サイズの積に従って分割される点に注意してください。

### マルチソケット GPU ノードでの NUMA バインド { #numa-binding-for-multi-socket-gpu-nodes }

マルチソケットの GPU サーバーでは、GPU ワーカープロセスの CPU 実行やメモリ確保が、その GPU に
もっとも近い NUMA ノードから外れると性能が低下することがあります。vLLM は Python のサブプロセスを
起動する前に `numactl` で各ワーカーを固定できるため、インタプリタ・import・初期のアロケータの状態を
最初から目的の NUMA ポリシーの下で作成できます。

この機能は `--numa-bind` で有効にします。既定では vLLM が GPU と NUMA ノードの対応を自動検出し、
各ワーカーに `--cpunodebind=<node> --membind=<node>` を使います。独自の CPU ポリシーが必要な場合は
`--numa-bind-cpus` を追加すると、vLLM は `--physcpubind=<cpu-list> --membind=<node>` に切り替えます。

これらの `--numa-bind*` オプションは GPU の実行プロセスにのみ適用され、CPU バックエンドが持つ
別のスレッドアフィニティ設定には影響しません。GPU と NUMA の自動検出は、現時点では CUDA/NVML 系と
ROCM 系のプラットフォームで実装されています。その他の GPU バックエンドでこれらのオプションを使う場合は、
バインド先を明示的に指定する必要があります。

`--numa-bind-nodes` には、可視の GPU ごとに 0 以上の NUMA ノード番号を、GPU のインデックスと同じ順で指定します。
`--numa-bind-cpus` には、可視の GPU ごとに `numactl` の CPU リストを、GPU のインデックスと同じ順で指定します。
各 CPU リストは `0-3`、`0,2,4-7`、`16-31,48-63` のように `numactl --physcpubind` の書式で記述します。

```bash
# Auto-detect NUMA nodes for visible GPUs
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 4 \
  --numa-bind

# Explicit NUMA-node mapping
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 4 \
  --numa-bind \
  --numa-bind-nodes 0 0 1 1

# Explicit CPU pinning, useful for PCT or other high-frequency core layouts
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 4 \
  --numa-bind \
  --numa-bind-nodes 0 0 1 1 \
  --numa-bind-cpus 0-3 4-7 48-51 52-55
```

注意点:

- CLI から使う場合、マルチプロセスの起動方式は自動的に `spawn` になります。Python API から NUMA バインドを有効にする場合は、`VLLM_WORKER_MULTIPROC_METHOD=spawn` も設定してください。
- 自動検出はホスト側の NVML と NUMA のサポートに依存します。対応を確実に判定できない場合は `--numa-bind-nodes` を明示的に指定してください。
- 明示的に指定する `--numa-bind-nodes` と `--numa-bind-cpus` の値は、`numactl` が受け付ける形式である必要があります。vLLM も簡単な検証は行いますが、最終的なバインドの意味は `numactl` が決定します。
- 現在の実装では、`EngineCore` やマルチプロセスのワーカーといった GPU 実行プロセスをバインドします。フロントエンドの API サーバープロセスや DP コーディネーターには NUMA バインドは適用されません。
- コンテナ環境では、NUMA ポリシーのシステムコールに追加の権限が必要になることがあります（`docker run` の場合は `--cap-add SYS_NICE` など）。

### CPU バックエンドのスレッドアフィニティ { #cpu-backend-thread-affinity }

CPU バックエンドは `--numa-bind` とは異なる仕組みを使います。CPU での実行は、GPU 向けの
`--numa-bind*` CLI オプションではなく、`VLLM_CPU_OMP_THREADS_BIND`、
`VLLM_CPU_NUM_OF_RESERVED_CPU`、`CPU_VISIBLE_MEMORY_NODES` といった CPU 固有の環境変数で設定します。

既定の `VLLM_CPU_OMP_THREADS_BIND=auto` では、各 CPU ワーカーに対して利用可能な CPU と NUMA の
トポロジから OpenMP の配置を決定します。自動のポリシーを上書きするには、CPU バックエンドの
ドキュメントに記載された CPU リスト形式で `VLLM_CPU_OMP_THREADS_BIND` を明示的に設定するか、
`nobind` を指定してこの動作を無効にします。

現在の CPU バックエンドのセットアップとチューニングについては、次を参照してください。

- [Related runtime environment variables](../getting_started/installation/cpu.md#related-runtime-environment-variables)
- [How to decide `VLLM_CPU_OMP_THREADS_BIND`](../getting_started/installation/cpu.md#how-to-decide-vllm_cpu_omp_threads_bind)

GPU 専用の `--numa-bind`、`--numa-bind-nodes`、`--numa-bind-cpus` オプションは、
CPU ワーカーのアフィニティを設定しません。

### マルチモーダルエンコーダーのバッチ単位 DP { #batch-level-dp-for-multi-modal-encoders }

既定では、各 GPU のメモリと計算負荷を減らすため、言語デコーダーと同様に TP を使って
マルチモーダルエンコーダーの重みを分割します。

しかし、マルチモーダルエンコーダーは言語デコーダーに比べて非常に小さいため、TP による効果は限定的です。
一方で TP は層ごとに all-reduce を行うため、通信のオーバーヘッドが無視できません。

そこで、重みではなくバッチ化された入力データを TP で分割する、実質的にバッチ単位の DP を行うほうが
有利な場合があります。`tensor_parallel_size=8` では、スループットと TTFT がおよそ 10% 改善することが
確認されています。ハードウェア最適化されていない Conv3D 演算を使う画像エンコーダーでは、
通常の TP と比べてさらに 40% ほど改善します。

ただし、マルチモーダルエンコーダーの重みは各 TP ランクに複製されるため、メモリ使用量がわずかに増えます。
すでにぎりぎりでモデルが載っている状況では OOM を引き起こす可能性があります。

バッチ単位の DP は `mm_encoder_tp_mode="data"` を設定すると有効になります。例:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-VL-72B-Instruct",
    tensor_parallel_size=4,
    # When mm_encoder_tp_mode="data",
    # the vision encoder uses TP=4 (not DP=1) to shard the input data,
    # so the TP size becomes the effective DP size.
    # Note that this is independent of the DP size for language decoder which is used in expert parallel setting.
    mm_encoder_tp_mode="data",
    # The language decoder uses TP=4 to shard the weights regardless
    # of the setting of mm_encoder_tp_mode
)
```

!!! important
    バッチ単位の DP は、API のリクエスト単位の DP（こちらは `data_parallel_size` で制御します）とは
    別のものです。混同しないでください。

バッチ単位の DP はモデルごとに実装が必要で、モデルクラスで `supports_encoder_tp_data = True` を
設定することで有効になります。いずれの場合も、この機能を使うにはエンジン引数で
`mm_encoder_tp_mode="data"` を指定する必要があります。

対応が確認されているモデル（対応するベンチマーク付き）:

- dots_ocr (<https://github.com/vllm-project/vllm/pull/25466>)
- GLM-4.1V or above (<https://github.com/vllm-project/vllm/pull/23168>)
- InternVL (<https://github.com/vllm-project/vllm/pull/23909>)
- Kimi-VL (<https://github.com/vllm-project/vllm/pull/23817>)
- Llama4 (<https://github.com/vllm-project/vllm/pull/18368>)
- MiniCPM-V-2.5 or above (<https://github.com/vllm-project/vllm/pull/23327>, <https://github.com/vllm-project/vllm/pull/23948>)
- Qwen2-VL or above (<https://github.com/vllm-project/vllm/pull/22742>, <https://github.com/vllm-project/vllm/pull/24955>, <https://github.com/vllm-project/vllm/pull/25445>)
- Step3 (<https://github.com/vllm-project/vllm/pull/22697>)

## 入力処理 { #input-processing }

### fastokens バックエンド { #fastokens-backend }

既定では、vLLM は fast tokenizer に標準の Hugging Face `tokenizers` ライブラリを使います。
BPE 系のトークナイザー（Qwen、Llama、DeepSeek、GPT-OSS など）では、
[fastokens](https://github.com/crusoecloud/fastokens) の Rust バックエンドに切り替えられます。
これは差し替え可能な実装で、エンコード・デコードやストリーミングのデトークナイズが大幅に高速です。
`VLLM_USE_FASTOKENS` は vLLM v0.23.0 以降で利用できます。インストール済みの vLLM がこの環境変数を
認識しない場合は、有効にする前に vLLM をアップグレードしてください。

```console
VLLM_USE_FASTOKENS=1 vllm serve Qwen/Qwen3-8B
```

オフライン API での同等の指定:

```python
import os
os.environ["VLLM_USE_FASTOKENS"] = "1"

from vllm import LLM
llm = LLM(model="Qwen/Qwen3-8B")
```

`fastokens` の Python パッケージ（0.2.0 以上）がインストールされている必要があります。
入っていない場合、vLLM はトークナイザーの読み込み時に明確な `ImportError` を発生させます。
この上書きは、HF の fast tokenizer を読み込む `--tokenizer-mode`（`hf`、`deepseek_v32`、
`deepseek_v4` など）すべてに適用されます。HF の fast tokenizer を使わないモデル
（`mistral`、`kimi_audio`）ではこのフラグは無視されます。

トークナイザーが律速となるワークロード（長い共通プレフィックス、短いプロンプトのバースト、
バッチのデトークナイズ）でもっとも効果が大きくなります。ボトルネックが GPU の Prefill / Decode に
ある場合、トークナイザーの変更はエンドツーエンドではほとんど体感できません。

### 並列処理 { #parallel-processing }

[API サーバーのスケールアウト](../serving/data_parallel_deployment.md#internal-load-balancing)により、入力処理を並列に実行できます。
これは、（API サーバー内で実行される）入力処理が、（エンジンコア内で実行される）モデルの実行に比べて
ボトルネックになっていて、かつ CPU に余裕がある場合に有効です。

```console
# Run 4 API processes and 1 engine core process
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --api-server-count 4

# Run 4 API processes and 2 engine core processes
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --api-server-count 4 -dp 2
```

!!! note
    API サーバーのスケールアウトはオンライン推論でのみ利用できます。

!!! warning
    既定では、リクエストデータからメディア（画像など）を読み込むために、各 API サーバーで 8 個の CPU スレッドが使われます。

    API サーバーをスケールアウトする場合は、CPU リソースの枯渇を避けるために
    `VLLM_MEDIA_LOADING_THREAD_COUNT` の調整を検討してください。

!!! note
    API サーバーのスケールアウトを行うと、[マルチモーダルの IPC キャッシュ](#ipc-caching)は無効になります。
    このキャッシュは API プロセスとエンジンコアプロセスが 1 対 1 で対応している必要があるためです。

    [マルチモーダルのプロセッサキャッシュ](#processor-caching)には影響しません。

## マルチモーダルキャッシュ { #multi-modal-caching }

マルチモーダルキャッシュは、同じマルチモーダルデータの転送や処理が繰り返されるのを防ぎます。
これはマルチターンの会話でよく発生します。

### プロセッサキャッシュ { #processor-caching }

マルチモーダルのプロセッサキャッシュは自動的に有効になり、`BaseMultiModalProcessor` で
同じマルチモーダル入力を繰り返し処理しないようにします。

### IPC キャッシュ { #ipc-caching }

マルチモーダルの IPC キャッシュは、API プロセス (`P0`) とエンジンコアプロセス (`P1`) が 1 対 1 で
対応している場合に自動的に有効になり、両者の間で同じマルチモーダル入力を繰り返し転送しないようにします。

#### キー複製キャッシュ { #key-replicated-cache }

既定では、IPC キャッシュは**キー複製キャッシュ**を使います。キャッシュのキーは API (`P0`) と
エンジンコア (`P1`) の両プロセスに存在しますが、実際のキャッシュデータは `P1` にのみ置かれます。

#### 共有メモリキャッシュ { #shared-memory-cache }

複数のワーカープロセスが関わる場合（TP > 1 のときなど）は、**共有メモリキャッシュ**のほうが効率的です。
`mm_processor_cache_type="shm"` を設定すると有効になります。このモードではキャッシュのキーは `P0` に置かれ、
キャッシュデータ自体はすべてのプロセスからアクセスできる共有メモリに置かれます。

### 設定 { #configuration }

キャッシュのサイズは `mm_processor_cache_gb`（既定 4 GiB）で調整できます。

キャッシュの効果があまりない場合は、`mm_processor_cache_gb=0` で IPC キャッシュとプロセッサキャッシュの
両方を完全に無効化できます。

例:

```python
# Use a larger cache
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    mm_processor_cache_gb=8,
)

# Use a shared-memory based IPC cache
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    tensor_parallel_size=2,
    mm_processor_cache_type="shm",
    mm_processor_cache_gb=8,
)

# Disable the cache
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    mm_processor_cache_gb=0,
)
```

### キャッシュの配置 { #cache-placement }

設定に応じて、`P0` と `P1` 上のマルチモーダルキャッシュの内容は次のようになります。

| mm_processor_cache_type | キャッシュ種別 | `P0` のキャッシュ | `P1` エンジンのキャッシュ | `P1` ワーカーのキャッシュ | 最大メモリ |
| ----------------- | ----------- | ---------- | ---------- | ----------- | ----------- |
| lru | プロセッサキャッシュ | K + V | N/A | N/A | `mm_processor_cache_gb * data_parallel_size` |
| lru | キー複製キャッシュ | K | K + V | N/A | `mm_processor_cache_gb * api_server_count` |
| shm | 共有メモリキャッシュ | K | N/A | V | `mm_processor_cache_gb * api_server_count` |
| N/A | 無効 | N/A | N/A | N/A | `0` |

K: マルチモーダル項目のハッシュを保持
V: マルチモーダル項目の処理済みテンソルデータを保持

## GPU デプロイにおける CPU リソース { #cpu-resources-for-gpu-deployments }

vLLM V1 はマルチプロセス構成（[V1 のプロセス構成](../design/arch_overview.md#v1-process-architecture)を参照）を採用しており、各プロセスが CPU リソースを必要とします。CPU コアの割り当て不足は、特に仮想化環境において性能低下のよくある原因です。

### 最低限必要な CPU { #minimum-cpu-requirements }

`N` 個の GPU を使うデプロイでは、少なくとも次のプロセスが動きます。

- **API サーバープロセス 1 個** -- HTTP リクエスト、トークナイズ、入力処理を担当
- **エンジンコアプロセス 1 個** -- スケジューラを実行し、GPU ワーカーを統括
- **GPU ワーカープロセス N 個** -- GPU ごとに 1 個で、モデルの forward を実行

つまり、常に少なくとも **`2 + N` 個のプロセス**が CPU 時間を奪い合うことになります。

!!! warning
    物理 CPU コア数がプロセス数より少ないと競合が発生し、スループットとレイテンシが大きく悪化します。エンジンコアプロセスはビジーループで動作するため、CPU 不足の影響を特に受けやすいです。

最低でも `2 + N` 個の物理コア（API サーバーに 1、エンジンコアに 1、GPU ワーカーごとに 1）が必要です。実際には、OS や PyTorch のバックグラウンドスレッド、その他のシステムプロセスも CPU 時間を必要とするため、より多くのコアを割り当てるほど性能が向上します。

!!! important
    Please note we are referring to **physical CPU cores** here. If your system has hyperthreading enabled, then 1 vCPU = 1 hyperthread = 1/2 physical CPU core, so you need `2 x (2 + N)` minimum vCPUs.

### Data Parallel and Multi-API Server Deployments

When using data parallelism or multiple API servers, the CPU requirements increase:

```console
Minimum physical cores = A + DP + N + (1 if DP > 1 else 0)
```

where `A` is the API server count (defaults to `DP`), `DP` is the data parallel size, and `N` is the total number of GPUs. For example, with `DP=4, TP=2` on 8 GPUs:

```console
4 API servers + 4 engine cores + 8 GPU workers + 1 DP coordinator = 17 processes
```

### Performance Impact

CPU underprovisioning particularly impacts:

- **Input processing throughput** -- tokenization, chat template rendering, and multi-modal data loading all run on CPU
- **Scheduling latency** -- the engine core scheduler runs on CPU and directly affects how quickly new tokens are dispatched to the GPU workers
- **Output processing** -- detokenization, networking, and especially streaming token responses use CPU cycles

If you observe that GPU utilization is lower than expected, CPU contention may be the bottleneck. Increasing the number of available CPU cores and even the clock speed can significantly improve end-to-end performance.

## Attention Backend Selection

vLLM supports multiple attention backends optimized for different hardware and use cases. The backend is automatically selected based on your GPU architecture, model type, and configuration, but you can also manually specify one for optimal performance.

For detailed information on available backends, their feature support, and how to configure them, see the [Attention Backend Feature Support](../design/attention_backends.md) documentation.
