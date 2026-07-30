# vLLM のプロファイリング { #profiling-vllm }

!!! warning
    プロファイリングは、vLLM の開発者とメンテナーがコードベースの各部分に費やされる時間の割合を把握するためのものです。
    推論が大幅に遅くなるため、**vLLM のエンドユーザーはプロファイリングを有効にすべきではありません**。

!!! tip "プロファイラの選択"
    - オーバーヘッドが小さく、性能が重要なプロファイリングには **Nsight Systems** を使ってください。
    - スタックトレース、メモリ、形状といったより豊富なデバッグ情報を伴う中程度のオーバーヘッドのプロファイリングには
      **PyTorch Profiler** を使ってください。これらの機能を有効にするとオーバーヘッドが増えるため、
      ベンチマーク用途には推奨されません。

## PyTorch Profiler によるプロファイリング { #profile-with-pytorch-profiler }

vLLM のワーカーは、さまざまなプロファイラでトレースできます。プロファイリングは、サーバー起動時に `--profiler-config` フラグを設定することで有効にできます。

!!! note
    `--profiler-config` フラグは vLLM v0.13.0 以降で利用できます。それより前のバージョンを使っている場合は、
    この機能を使うためにアップグレードしてください。

`torch.profiler` モジュールを使うには、`profiler` の項目を `'torch'` に、`torch_profiler_dir` をトレースの保存先ディレクトリに設定します。さらに、設定で次の追加引数を指定することで、プロファイリングの内容を制御できます。

- `torch_profiler_record_shapes`: テンソルの形状の記録を有効にします。既定は無効。
- `torch_profiler_with_memory`: メモリを記録します。既定は無効。
- `torch_profiler_with_stack`: スタック情報の記録を有効にします。既定は有効。
- `torch_profiler_with_flops`: FLOPs の記録を有効にします。既定は無効。
- `torch_profiler_use_gzip`: プロファイリングのファイルを gzip 圧縮するかを制御します。既定は有効。
- `torch_profiler_dump_cuda_time_total`: 集約された CUDA の self time のテーブルを出力・表示するかを制御します。既定は有効。

`vllm bench serve` を使う場合は、`--profile` フラグを渡すことでプロファイリングを有効にできます。

トレースは <https://ui.perfetto.dev/> で可視化できます。

!!! tip
    `python -m vllm.entrypoints.cli.main bench` を使えば、vLLM をインストールせずに bench モジュールを直接呼び出せます。

!!! tip
    トレースはかなり大きくなることがあるため、プロファイリング時に vLLM へ送るリクエストは少数にとどめてください。
    また、トレースを展開する必要はありません。そのまま閲覧できます。

!!! tip
    プロファイラを停止すると、すべてのプロファイルトレースのファイルがディレクトリに書き出されます。これには時間がかかります。
    たとえば llama 70b で約 100 リクエスト分のデータの場合、H100 では書き出しに約 10 分かかります。
    エンジンクライアントはこの書き出しの完了をタイムアウトせずに待つため、停止の呼び出しが完了するまでそのまま待ってください。

### コマンドと使い方の例 { #example-commands-and-usage }

#### オフライン推論 { #offline-inference }

例は [examples/features/profiling/simple_profiling_offline.py](../../examples/features/profiling/simple_profiling_offline.py) を参照してください。

#### OpenAI サーバー { #openai-server }

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile"}'
```

vllm bench コマンド:

```bash
vllm bench serve \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path sharegpt.json \
    --profile \
    --num-prompts 2
```

あるいは HTTP リクエストを使います。

```shell
# We need first call /start_profile api to start profile.
$ curl -X POST http://localhost:8000/start_profile

# Call model generate.
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [
                        {
                                "role": "user",
                                "content": "San Francisco is a"
                        }
                ]
    }'

# After need call /stop_profile api to stop profile.
$ curl -X POST http://localhost:8000/stop_profile
```

## NVIDIA Nsight Systems によるプロファイリング { #profile-with-nvidia-nsight-systems }

Nsight Systems は、レジスタや共有メモリの使用状況、注釈付きのコード領域、低レベルの CUDA API やイベントなど、より詳細なプロファイリング情報を提供する高度なツールです。

パッケージマネージャで [nsight-systems をインストール](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html)してください。次のブロックは Ubuntu の例です。

```bash
apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install nsight-systems-cli
```

!!! tip
    `nsys` でプロファイリングする際は、環境変数 `VLLM_WORKER_MULTIPROC_METHOD=spawn` を設定することを推奨します。
    既定では `spawn` ではなく `fork` が使われます。詳細は
    [Nsight Systems のリリースノート](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html#general-issues)を参照してください。

Nsight Systems のプロファイラは `nsys profile ...` で起動できます。vLLM では `--trace-fork-before-exec=true --cuda-graph-trace=node` のフラグを推奨します。

### コマンドと使い方の例 { #example-commands-and-usage_1 }

#### オフライン推論 { #offline-inference_1 }

基本的な使い方としては、オフライン推論で実行する既存のスクリプトの前にプロファイリングのコマンドを付けるだけです。

次は `vllm bench latency` スクリプトを使う例です。

```bash
nsys profile  \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
vllm bench latency \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-iters-warmup 5 \
    --num-iters 1 \
    --batch-size 16 \
    --input-len 512 \
    --output-len 8
```

#### OpenAI サーバー { #openai-server_1 }

サーバーをプロファイリングするには、オフライン推論と同様に `vllm serve` コマンドの前に `nsys profile` を付けます。ただし、Torch Profiler と同様の動的なキャプチャを有効にするために、いくつか追加の引数を指定する必要があります。

```bash
# server
nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end repeat \
    vllm serve meta-llama/Llama-3.1-8B-Instruct --profiler-config.profiler cuda

# client
vllm bench serve \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path sharegpt.json \
    --profile \
    --num-prompts 2
```

`--profile` を指定すると、vLLM は `vllm bench serve` の実行ごとにプロファイルをキャプチャします。サーバーを終了すると、すべてのプロファイルが保存されます。

#### 分析 { #analysis }

これらのプロファイルは、`nsys stats [profile-file]` を使って CLI 上でサマリーとして表示するか、[こちらの手順](https://developer.nvidia.com/nsight-systems/get-started)に従ってローカルに Nsight をインストールし、GUI で表示できます。

??? console "CLI の例"

    ```bash
    nsys stats report1.nsys-rep
    ...
    ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

    Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)  Max (ns)   StdDev (ns)                                                  Name
    --------  ---------------  ---------  -----------  -----------  --------  ---------  -----------  ----------------------------------------------------------------------------------------------------
        46.3   10,327,352,338     17,505    589,965.9    144,383.0    27,040  3,126,460    944,263.8  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_of…
        14.8    3,305,114,764      5,152    641,520.7    293,408.0   287,296  2,822,716    867,124.9  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_of…
        12.1    2,692,284,876     14,280    188,535.4     83,904.0    19,328  2,862,237    497,999.9  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x128x64_warpgroupsize1x1x1_execute_segment_k_off…
        9.5    2,116,600,578     33,920     62,399.8     21,504.0    15,326  2,532,285    290,954.1  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x64x64_warpgroupsize1x1x1_execute_segment_k_off_…
        5.0    1,119,749,165     18,912     59,208.4      9,056.0     6,784  2,578,366    271,581.7  void vllm::act_and_mul_kernel<c10::BFloat16, &vllm::silu_kernel<c10::BFloat16>, (bool)1>(T1 *, cons…
        4.1      916,662,515     21,312     43,011.6     19,776.0     8,928  2,586,205    199,790.1  void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMa…
        2.6      587,283,113     37,824     15,526.7      3,008.0     2,719  2,517,756    139,091.1  std::enable_if<T2>(int)0&&vllm::_typeConvert<T1>::exists, void>::type vllm::fused_add_rms_norm_kern…
        1.9      418,362,605     18,912     22,121.5      3,871.0     3,328  2,523,870    175,248.2  void vllm::rotary_embedding_kernel<c10::BFloat16, (bool)1>(const long *, T1 *, T1 *, const T1 *, in…
        0.7      167,083,069     18,880      8,849.7      2,240.0     1,471  2,499,996    101,436.1  void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0…
    ...
    ```

GUI の例:

<img width="1799" alt="Screenshot 2025-03-05 at 11 48 42 AM" src="https://github.com/user-attachments/assets/c7cff1ae-6d6f-477d-a342-bd13c4fc424c" />

## 継続的プロファイリング { #continuous-profiling }

PyTorch のインフラリポジトリには、vLLM 上のさまざまなモデルについて継続的なプロファイリングを行う [GitHub CI ワークフロー](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-profiling.yml)があります。この自動プロファイリングにより、時系列およびモデル構成をまたいだ性能特性を追跡できます。

### 仕組み { #how-it-works }

現在、このワークフローは選定されたモデルについて週次でプロファイリングを実行し、性能のリグレッションや最適化の余地を特定するためにさまざまなツールで分析できる詳細な性能トレースを生成します。GitHub Action のツールを使って手動でトリガーすることもできます。

### 新しいモデルの追加 { #adding-new-models }

継続的プロファイリングの対象にモデルを追加するには、PyTorch の integration testing リポジトリにある [profiling-tests.json](https://github.com/pytorch/pytorch-integration-testing/blob/main/vllm-profiling/cuda/profiling-tests.json) の設定ファイルを変更します。このファイルにモデルの仕様を追加するだけで、自動プロファイリングの実行に含められます。

### プロファイリング結果の閲覧 { #viewing-profiling-results }

継続的プロファイリングのワークフローが生成したプロファイリングトレースは、[vLLM Performance Dashboard](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm) で公開されています。**Profiling traces** の表から、モデルや実行ごとのトレースにアクセスしてダウンロードできます。

## vLLM の Python コードのプロファイリング { #profiling-vllm-python-code }

Python の標準ライブラリには、Python コードをプロファイリングするための [cProfile](https://docs.python.org/3/library/profile.html) が含まれています。

### 使用例 - 関数呼び出し { #example-usage-function-call }

ファイル名を指定すると、プロファイルはそのファイルに保存されます。ファイル名を指定しない場合、プロファイルのデータを標準出力に表示できます。

```python
import cProfile


def expensive_function():
    # some expensive code
    pass


profiler = cProfile.Profile()
profiler.runcall(expensive_function)
profiler.dump_stats("expensive_function.prof")
```

### 使用例 - コンテキストマネージャ形式 { #example-usage-context-manager-style }

```python
import cProfile


def another_function():
    # more expensive code
    pass


profiler = cProfile.Profile()
profiler.enable()
try:
    another_function()
finally:
    profiler.disable()
    profiler.dump_stats("another_function.prof")
```

### プロファイル結果の分析 { #analyzing-profile-results }

プロファイル結果の分析に役立つツールは複数あります。その一例が [snakeviz](https://jiffyclub.github.io/snakeviz/) です。

```bash
pip install snakeviz
snakeviz expensive_function.prof
```

### ガベージコレクションのコストの分析 { #analyzing-garbage-collection-costs }

GC のコストを調べるには、環境変数 VLLM_GC_DEBUG を活用してください。

- VLLM_GC_DEBUG=1: gc.collect の経過時間を出力する GC デバッガを有効にします
- VLLM_GC_DEBUG='{"top_objects":5}': gc.collect ごとに、回収されたオブジェクトの上位 5 件を
  ログ出力する GC デバッガを有効にします
