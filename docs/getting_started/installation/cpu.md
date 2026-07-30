---
toc_depth: 3
---

# CPU

vLLM は、次の CPU 種別に対応する Python ライブラリです。ベンダー固有の手順を見るには、お使いの CPU の種類を選択してください。

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:installation"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:installation"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:installation"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:installation"

## 技術的な議論 { #technical-discussions }

主な議論は [vLLM Slack](https://slack.vllm.ai/) の `#sig-cpu` チャンネルで行われています。

CPU バックエンドに関する GitHub issue を作成する際は、タイトルに `[CPU Backend]` を付けてください。`cpu` ラベルが付き、気づかれやすくなります。

## 要件 { #requirements }

- Python: 3.10 〜 3.13

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:requirements"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:requirements"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:requirements"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:requirements"

## Python を使ったセットアップ { #set-up-using-python }

### 新しい Python 環境を作成する { #create-a-new-python-environment }

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

### ビルド済み wheel { #pre-built-wheels }

インデックス URL を指定する際は、必ず `cpu` 版のサブディレクトリを使ってください。
たとえば nightly ビルドのインデックスは `https://wheels.vllm.ai/nightly/cpu/` です。

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:pre-built-wheels"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:pre-built-wheels"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:pre-built-wheels"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:pre-built-wheels"

### ソースから wheel をビルドする { #build-wheel-from-source }

#### Python のみのビルド（コンパイルなし）でセットアップする {#python-only-build}

この方法には、お使いのプラットフォーム向けの[ビルド済み wheel](#pre-built-wheels) が必要です。

[GPU での Python のみのビルド](./gpu.md#python-only-build)の手順を参照し、ビルドコマンドを次のように置き換えてください。

```bash
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cpu VLLM_TARGET_DEVICE=cpu uv pip install --editable .
```

#### フルビルド（コンパイルあり） {#full-build}

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:build-wheel-from-source"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:build-wheel-from-source"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:build-wheel-from-source"

=== "IBM Z (s390x)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:build-wheel-from-source"

## Docker を使ったセットアップ { #set-up-using-docker }

### ビルド済みイメージ { #pre-built-images }

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:pre-built-images"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:pre-built-images"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:pre-built-images"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:pre-built-images"

### ソースからイメージをビルドする { #build-image-from-source }

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:build-image-from-source"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:build-image-from-source"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:build-image-from-source"

=== "IBM Z (S390X)"
    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:build-image-from-source"

## AMD Zen 最適化 {#amd-zen-optimizations}

--8<-- "docs/getting_started/installation/cpu.x86.inc.md:amd-zen-optimizations"

## 関連する実行時の環境変数 { #related-runtime-environment-variables }

- `VLLM_CPU_KVCACHE_SPACE`: KV キャッシュのサイズを指定します（例: `VLLM_CPU_KVCACHE_SPACE=40` は KV キャッシュに 40 GiB を割り当てることを意味します）。値を大きくすると、vLLM がより多くのリクエストを並行して処理できます。このパラメータは、ハードウェア構成とメモリ管理の方針にもとづいて設定してください。既定値は `0` です。
- `VLLM_CPU_OMP_THREADS_BIND`: OpenMP スレッドを割り当てる CPU コアを指定します。CPU ID のリスト、`auto`（既定）、`nobind`（個々の CPU コアへのバインドを無効にし、ユーザー定義の OpenMP 変数を引き継ぐ）のいずれかを設定できます。たとえば `VLLM_CPU_OMP_THREADS_BIND=0-31` は、CPU コア 0〜31 に 32 個の OpenMP スレッドをバインドすることを意味します。`VLLM_CPU_OMP_THREADS_BIND=0-31|32-63` は、テンソル並列のプロセスが 2 つあり、rank0 の 32 個の OpenMP スレッドが CPU コア 0〜31 に、rank1 の OpenMP スレッドが CPU コア 32〜63 にバインドされることを意味します。`auto` を設定すると、各ランクの OpenMP スレッドがそれぞれの NUMA ノード内の CPU コアにバインドされます。`nobind` を設定した場合、OpenMP スレッド数は標準の `OMP_NUM_THREADS` 環境変数で決まります。
- `VLLM_CPU_NUM_OF_RESERVED_CPU`: ランクごとに、OpenMP スレッド用に割り当てない CPU コアの数を指定します。この変数は VLLM_CPU_OMP_THREADS_BIND が `auto` の場合にのみ有効です。既定値は `None` です。値を設定せずに `auto` のスレッドバインドを使う場合、`world_size == 1` では CPU は予約されず、`world_size > 1` ではランクあたり 1 CPU が予約されます。
- `CPU_VISIBLE_MEMORY_NODES`: ```CUDA_VISIBLE_DEVICES``` と同様に、vLLM の CPU ワーカーから見える NUMA メモリノードを指定します。この変数は VLLM_CPU_OMP_THREADS_BIND が `auto` の場合にのみ有効です。ノードのマスクやバインド順序の変更など、自動スレッドバインド機能をより細かく制御できます。
- `VLLM_CPU_SGL_KERNEL`（x86 のみ、実験的）: 線形層と MoE 層で、小さいバッチ向けに最適化されたカーネルを使うかどうかを指定します。とくにオンラインサービングのような低レイテンシ要件で有効です。これらのカーネルには AMX 命令セット、BFloat16 の重み型、32 で割り切れる重みの形状が必要です。既定値は `0`（False）です。
- `VLLM_ZENTORCH_WEIGHT_PREPACK`（AMD Zen のみ）: `ZenCpuPlatform` が有効な場合に、モデルのロード時に線形層の重みを ZenDNN のブロック化レイアウトへ先読みで prepack し、推論ごとのレイアウト変換のオーバーヘッドをなくします。既定値は `1`（有効）です。[AMD Zen 最適化](#amd-zen-optimizations)を参照してください。

## FAQ { #faq }

### どの `dtype` を使うべきか { #which-dtype-should-be-used }

- 現在、vLLM CPU は `dtype` としてモデルの既定設定を使います。ただし torch CPU の float16 サポートが不安定なため、性能や精度に問題がある場合は `dtype=bfloat16` を明示的に指定することを推奨します。
- AMD Zen CPU（`ZenCpuPlatform`）では `float16` は**サポートされません**。受け付けられるのは `bfloat16` と `float32` のみで、`float16` と宣言されたモデルはロード時に自動的に `bfloat16` へダウンキャストされます。[AMD Zen 最適化](#amd-zen-optimizations)を参照してください。

### CPU 上で vLLM のサービスを起動するには { #how-to-launch-a-vllm-service-on-cpu }

- オンラインサービングを使う場合、CPU の過剰割り当てを避けるため、サービングのフレームワーク用に 1〜2 個の CPU コアを予約することを推奨します。たとえば物理 CPU コアが 32 個のプラットフォームで、CPU 31 をフレームワーク用に予約し、CPU 0〜30 を推論スレッドに使う場合は次のようにします。

```bash
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_OMP_THREADS_BIND=0-30
vllm serve facebook/opt-125m --dtype=bfloat16
```

 あるいは、既定の自動スレッドバインドを使う場合は次のようにします。

```bash
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_NUM_OF_RESERVED_CPU=1
vllm serve facebook/opt-125m --dtype=bfloat16
```

なお、`world_size == 1` の場合は、vLLM のフロントエンドプロセス用に手動で CPU を 1 つ予約することを推奨します。

### CPU でサポートされているモデルは { #what-are-supported-models-on-cpu }

CPU プラットフォームで検証済みのモデルの完全かつ最新の一覧は、公式ドキュメントの [CPU でサポートされるモデル](../../models/hardware_supported_models/cpu.md)を参照してください。

### サポート対象の CPU モデル向けのベンチマーク設定例はどこにあるか { #how-to-find-benchmark-configuration-examples-for-supported-cpu-models }

[CPU でサポートされるモデル](../../models/hardware_supported_models/cpu.md)に掲載されているモデルについては、vLLM Benchmark Suite の CPU テストケース（cpu テストケースの serving-tests-cpu.json で定義）に最適化された実行時設定が用意されています。テキストのみのモデル、マルチモーダルモデル、埋め込みモデルの完全なテストケースは、それぞれ cpu Text-Only テストケースの serving-tests-cpu-text.json、cpu Multi-Modal テストケースの serving-tests-cpu-multimodal.json、cpu Embedded テストケースの serving-tests-cpu-embed.json にあります。
これらの最適化された設定がどのように決められているかについては、[performance-benchmark-details](../../../.buildkite/performance-benchmarks/README.md#performance-benchmark-details) を参照してください。
これらの最適化された設定でサポート対象モデルのベンチマークを実行するには、[vLLM Benchmark Suite を手動で実行する](../../benchmarking/dashboard.md#manually-trigger-the-benchmark)の手順に従い、CPU 環境で Benchmark Suite を実行してください。

以下は、最適化された設定を使って CPU 対応の全モデルをベンチマークするコマンドの例です。

```bash
ON_CPU=1 bash .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
```

ベンチマーク結果は `./benchmark/results/` に保存されます。
このディレクトリには、生成された `.commands` ファイルがあり、ベンチマーク用のコマンド例がすべて含まれています。

tensor-parallel-size は、システムの NUMA ノード数に合わせて設定することを推奨します。なお、現在のリリースでは tensor-parallel-size=6 はサポートされていません。
利用可能な NUMA ノード数は、次のコマンドで確認できます。

```bash
lscpu | grep "NUMA node(s):" | awk '{print $3}'
```

性能の参考として、[vLLM Performance Dashboard](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm&deviceName=cpu)
も参照できます。ここでは、同じ Benchmark Suite を使って得られた既定モデルの CPU 結果が公開されています。

#### Dry-Run { #dry-run }

ベンチマークを実行せずに最適化された実行時設定だけを取得したい場合のために、Dry-Run モードが用意されています。
run-performance-benchmarks.sh に環境変数 DRY_RUN=1 を渡すと、すべてのコマンドが `./benchmark/results/` 配下に生成されます。

```bash
ON_CPU=1 DRY_RUN=1 bash .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
```

別の JSON ファイルを指定すれば、埋め込みモデルなど別種のモデル向けの実行時設定も取得できます。

```bash
ON_CPU=1 SERVING_JSON=serving-tests-cpu-embed.json DRY_RUN=1 bash .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
```

MODEL_FILTER と DTYPE_FILTER を指定すると、該当するモデル ID とデータ型のコマンドだけが生成されます。

```bash
ON_CPU=1 SERVING_JSON=serving-tests-cpu-text.json DRY_RUN=1 MODEL_FILTER=meta-llama/Llama-3.1-8B-Instruct DTYPE_FILTER=bfloat16  bash .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
```

### AMD Zen 最適化を有効にするには？ {#how-do-i-enable-amd-zen-optimizations}

AMD Zen 4 / Zen 5 の CPU では、`zen` エクストラを付けて CPU 版 wheel をインストールします。これにより、そのリリースで検証済みの `zentorch` のバージョンが取得されます。

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
uv pip install "vllm[zen]" --extra-index-url https://wheels.vllm.ai/${VLLM_VERSION}/cpu --index-strategy first-index --torch-backend cpu
```

vLLM はプラットフォームを自動検出し、線形層を ZenDNN 最適化カーネル経由でルーティングします。フラグの指定は不要です。有効になっているかを確認するには、サーバー起動時のログでプラットフォーム選択の行を探してください。

```bash
vllm serve Qwen/Qwen3-0.6B 2>&1 | grep "AMD Zen CPU detected with zentorch installed"
```

バックエンドごとのディスパッチの詳細（各線形層がどのカーネルにバインドされたか）を見るには、`VLLM_LOGGING_LEVEL=DEBUG` を付けて再実行し、`CPU unquantized GEMM dispatch` を grep してください。

判定ルール、サポートされる dtype、`VLLM_ZENTORCH_WEIGHT_PREPACK` の設定については [AMD Zen 最適化](#amd-zen-optimizations)を参照してください。

### `VLLM_CPU_OMP_THREADS_BIND` はどう決めればよいか { #how-to-decide-vllm_cpu_omp_threads_bind }

- ほとんどの場合、既定の `auto` によるスレッドバインドを推奨します。理想的には、各 OpenMP スレッドがそれぞれ専用の物理コアにバインドされ、各ランクのスレッドが同じ NUMA ノードにバインドされ、`world_size > 1` の場合はランクあたり 1 CPU が他の vLLM コンポーネント用に予約されます。性能の問題や意図しないバインドの挙動がある場合は、次のようにスレッドをバインドしてみてください。

- ハイパースレッディングが有効で、論理 CPU コア 16 個 / 物理 CPU コア 8 個のプラットフォームの場合:

??? console "コマンド"

    ```console
    $ lscpu -e # check the mapping between logical CPU cores and physical CPU cores

    # The "CPU" column means the logical CPU core IDs, and the "CORE" column means the physical core IDs. On this platform, two logical cores are sharing one physical core.
    CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ      MHZ
    0    0      0    0 0:0:0:0          yes 2401.0000 800.0000  800.000
    1    0      0    1 1:1:1:0          yes 2401.0000 800.0000  800.000
    2    0      0    2 2:2:2:0          yes 2401.0000 800.0000  800.000
    3    0      0    3 3:3:3:0          yes 2401.0000 800.0000  800.000
    4    0      0    4 4:4:4:0          yes 2401.0000 800.0000  800.000
    5    0      0    5 5:5:5:0          yes 2401.0000 800.0000  800.000
    6    0      0    6 6:6:6:0          yes 2401.0000 800.0000  800.000
    7    0      0    7 7:7:7:0          yes 2401.0000 800.0000  800.000
    8    0      0    0 0:0:0:0          yes 2401.0000 800.0000  800.000
    9    0      0    1 1:1:1:0          yes 2401.0000 800.0000  800.000
    10   0      0    2 2:2:2:0          yes 2401.0000 800.0000  800.000
    11   0      0    3 3:3:3:0          yes 2401.0000 800.0000  800.000
    12   0      0    4 4:4:4:0          yes 2401.0000 800.0000  800.000
    13   0      0    5 5:5:5:0          yes 2401.0000 800.0000  800.000
    14   0      0    6 6:6:6:0          yes 2401.0000 800.0000  800.000
    15   0      0    7 7:7:7:0          yes 2401.0000 800.0000  800.000

    # On this platform, it is recommended to only bind openMP threads on logical CPU cores 0-7 or 8-15
    $ export VLLM_CPU_OMP_THREADS_BIND=0-7
    $ python examples/basic/offline_inference/basic.py
    ```

- NUMA を備えたマルチソケットマシンで vLLM の CPU バックエンドをデプロイし、テンソル並列やパイプライン並列を有効にすると、各 NUMA ノードが 1 つの TP / PP ランクとして扱われます。したがって、NUMA ノードをまたぐメモリアクセスを避けるため、1 つのランクの CPU コアは同じ NUMA ノード上に設定するよう注意してください。

### `VLLM_CPU_KVCACHE_SPACE` はどう決めればよいか { #how-to-decide-vllm_cpu_kvcache_space }

この値は既定で 4GB です。大きくすると、より多くの同時リクエストや長いコンテキスト長に対応できます。ただし、各 NUMA ノードのメモリ容量に注意が必要です。各 TP ランクのメモリ使用量は `weight shard size` と `VLLM_CPU_KVCACHE_SPACE` の合計であり、これが単一 NUMA ノードの容量を超えると、TP ワーカーはメモリ不足により `exitcode 9` で強制終了されます。

### vLLM CPU の性能チューニングはどう行うか { #how-to-do-performance-tuning-for-vllm-cpu }

まず、スレッドバインドと KV キャッシュ領域が正しく設定され、実際に効いていることを確認してください。スレッドバインドは、vLLM のベンチマークを実行して `htop` で CPU コアの使用状況を観察すれば確認できます。

`--block-size` には 32 の倍数を使ってください（既定値は 128 です）。

推論のバッチサイズは性能にとって重要なパラメータです。バッチが大きいほど通常スループットは高くなり、小さいほどレイテンシは低くなります。既定値から始めて最大バッチサイズを調整し、スループットとレイテンシのバランスを取ることは、特定のプラットフォームで vLLM CPU の性能を高める有効な手段です。vLLM には関連する重要なパラメータが 2 つあります。

- `--max-num-batched-tokens`: 1 バッチあたりのトークン数の上限を定めます。最初のトークンの性能により大きく影響します。既定値は次のとおりです。
    - オフライン推論: `4096 * world_size`
    - オンラインサービング: `2048 * world_size`
- `--max-num-seqs`: 1 バッチあたりのシーケンス数の上限を定めます。出力トークンの性能により大きく影響します。
    - オフライン推論: `256 * world_size`
    - オンラインサービング: `128 * world_size`

vLLM CPU は、複数の CPU ソケットとメモリノードを活用するために、データ並列（DP）、テンソル並列（TP）、パイプライン並列（PP）をサポートしています。DP、TP、PP のチューニングの詳細は[最適化とチューニング](../../configuration/optimization.md)を参照してください。vLLM CPU では、CPU ソケットとメモリノードが十分にある場合、DP・TP・PP を組み合わせて使うことを推奨します。

### vLLM CPU がサポートする量子化の設定は { #which-quantization-configs-does-vllm-cpu-support }

- vLLM CPU がサポートする量子化は次のとおりです。
    - AWQ（x86 のみ）
    - GPTQ（x86 のみ）
    - compressed-tensor INT8 W8A8（x86、s390x）

### Docker で実行すると `get_mempolicy: Operation not permitted` が出るのはなぜか { #why-do-i-see-get_mempolicy-operation-not-permitted-when-running-in-docker }

Docker などの一部のコンテナ環境では、vLLM が使う NUMA 関連のシステムコール（`get_mempolicy`、`migrate_pages` など）が、ランタイムの既定の seccomp / capabilities 設定によってブロック・拒否されます。その結果、`get_mempolicy: Operation not permitted` のような警告が出ることがあります。機能自体に影響はありませんが、NUMA のメモリバインド / マイグレーションの最適化が効かず、性能が最適でなくなる可能性があります。

最小限の権限で Docker 内でこれらの最適化を有効にするには、次の方法を参考にしてください。

```bash
docker run ... --cap-add SYS_NICE --security-opt seccomp=unconfined  ...

# 1) `--cap-add SYS_NICE` is to address `get_mempolicy` EPERM issue.

# 2) `--security-opt seccomp=unconfined` is to enable `migrate_pages` for `numa_migrate_pages()`.
# Actually, `seccomp=unconfined` bypasses the seccomp for container,
# if it's unacceptable, you can customize your own seccomp profile,
# based on docker/runtime default.json and add `migrate_pages` to `SCMP_ACT_ALLOW` list.

# reference : https://docs.docker.com/engine/security/seccomp/
```

代替として `--privileged=true` で実行しても動作しますが、権限が広すぎるため一般には推奨しません。

Kubernetes では、ワークロードの yaml に次の設定を追加すると、上記と同じ効果が得られます。

```yaml
securityContext:
  seccompProfile:
    type: Unconfined
  capabilities:
    add:
    - SYS_NICE
```
