# パラメータスイープ { #parameter-sweeps }

`vllm bench sweep` は、複数の構成でベンチマークを実行し、結果を可視化して比較するためのコマンド群です。

## オンラインベンチマーク { #online-benchmark }

### 基本 { #basic }

`vllm bench sweep serve` は `vllm serve` を起動し、各サーバー構成について `vllm bench serve` を繰り返し実行します。

!!! tip
    単一のサーバー構成でベンチマークを実行するだけなら、[GuideLLM](https://github.com/vllm-project/guidellm) の利用を検討してください。進捗のライブ更新とレポートの自動生成を備えた実績のある性能ベンチマークフレームワークです。データセットの読み込み、リクエストの整形、ワークロードのパターンの面でも `vllm bench serve` より柔軟です。

スクリプトを実行するには次の手順に従います。

1. `vllm serve` の基本コマンドを組み立て、`--serve-cmd` オプションに渡します。
2. `vllm bench serve` の基本コマンドを組み立て、`--bench-cmd` オプションに渡します。
3. （任意）`vllm serve` の設定を変化させたい場合は、新しい JSON ファイルを作成し、試したいパラメータの組み合わせを記述します。そのファイルパスを `--serve-params` に渡します。

    - 例: `--max-num-seqs` と `--max-num-batched-tokens` を調整する場合:

    ```json
    [
        {
            "max_num_seqs": 32,
            "max_num_batched_tokens": 1024
        },
        {
            "max_num_seqs": 64,
            "max_num_batched_tokens": 1024
        },
        {
            "max_num_seqs": 64,
            "max_num_batched_tokens": 2048
        },
        {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 2048
        },
        {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 4096
        },
        {
            "max_num_seqs": 256,
            "max_num_batched_tokens": 4096
        }
    ]
    ```

4. （任意）`vllm bench serve` の設定を変化させたい場合は、新しい JSON ファイルを作成し、試したいパラメータの組み合わせを記述します。そのファイルパスを `--bench-params` に渡します。

    - 例: random データセットで異なる入出力長を使う場合:

    ```json
    [
        {
            "_benchmark_name": "scenario_A",
            "random_input_len": 128,
            "random_output_len": 32
        },
        {
            "_benchmark_name": "scenario_B",
            "random_input_len": 256,
            "random_output_len": 64
        },
        {
            "_benchmark_name": "scenario_C",
            "random_input_len": 512,
            "random_output_len": 128
        }
    ]
    ```

5. 結果の保存先を指定するため、`--output-dir` と（必要に応じて）`--experiment-name` を設定します。

コマンドの例:

```bash
vllm bench sweep serve \
    --serve-cmd 'vllm serve meta-llama/Llama-2-7b-chat-hf' \
    --bench-cmd 'vllm bench serve --model meta-llama/Llama-2-7b-chat-hf --backend vllm --endpoint /v1/completions --dataset-name sharegpt --dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json' \
    --serve-params benchmarks/serve_hparams.json \
    --bench-params benchmarks/bench_hparams.json \
    --output-dir benchmarks/results \
    --experiment-name demo
```

既定では、結果の信頼性を高めるため、各パラメータの組み合わせを 3 回ベンチマークします。実行回数は `--num-runs` で調整できます。

!!! important
    `--serve-params` と `--bench-params` の両方を渡した場合、スクリプトは両者のデカルト積を順に処理します。
    実行されるコマンドは `--dry-run` でプレビューできます。

    サーバーは `--serve-params` ごとに 1 回だけ起動し、複数の `--bench-params` の間は起動したままにします。
    各ベンチマーク実行の間には、次の実行をクリーンな状態から始めるためにすべての `/reset_*_cache`
    エンドポイントを呼び出します。独自の `--serve-cmd` を使っている場合は、`--after-bench-cmd` を設定して
    状態のリセットに使うコマンドを上書きできます。

!!! note
    変数が多いパラメータの組み合わせには、人間が読みやすい名前を付けるために `_benchmark_name` を
    設定してください。ファイル名がファイルシステムの最大パス長を超えてしまう場合、これは必須になります。

!!! tip
    HF Hub への接続タイムアウトなど、予期しないエラーが起きた場合は `--resume` オプションで
    パラメータスイープを再開できます。

### ワークロードエクスプローラー { #workload-explorer }

`vllm bench sweep serve_workload` は `vllm bench sweep serve` の派生で、レイテンシとスループットのトレードオフを見つけるために、さまざまなワークロードの水準を探索します。結果は[可視化](#visualization)して、実現可能な SLA を判断することもできます。

ワークロードはリクエストレートまたは並行度で表現できます（`--workload-var` で選択します）。

コマンドの例:

```bash
vllm bench sweep serve_workload \
    --serve-cmd 'vllm serve meta-llama/Llama-2-7b-chat-hf' \
    --bench-cmd 'vllm bench serve --model meta-llama/Llama-2-7b-chat-hf --backend vllm --endpoint /v1/completions --dataset-name sharegpt --dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 100' \
    --workload-var max_concurrency \
    --serve-params benchmarks/serve_hparams.json \
    --bench-params benchmarks/bench_hparams.json \
    --num-runs 1 \
    --output-dir benchmarks/results \
    --experiment-name demo
```

さまざまなワークロード水準を探索するアルゴリズムは次のとおりです。

1. リクエストを 1 件ずつ送ってベンチマークを実行する（逐次推論、最も低いワークロード）。これにより、達成しうる最小のレイテンシとスループットが得られます。
2. すべてのリクエストを一度に送ってベンチマークを実行する（バッチ推論、最も高いワークロード）。これにより、達成しうる最大のレイテンシとスループットが得られます。
3. ステップ 2 に対応する `workload_var` の値を推定します。
4. 残りの反復回数を使って、`workload_var` の中間の値を均等に取りながらベンチマークを実行します。

このアルゴリズムの反復回数は `--workload-iters` で上書きできます。

!!! tip
    これは [GuideLLM の `--profile sweep`](https://github.com/vllm-project/guidellm/blob/v0.5.3/src/guidellm/benchmark/profiles.py#L575) に相当する機能です。

    一般に、`--workload-var max_concurrency` は vLLM エンジンにかかるワークロードを直接制御するため、
    より信頼できる結果が得られます。それでも、GuideLLM と似た挙動を保つために既定は
    `--workload-var request_rate` にしています。

## 起動ベンチマーク { #startup-benchmark }

`vllm bench sweep startup` は、パラメータの組み合わせにわたって `vllm bench startup` を実行し、エンジン設定ごとのコールド / ウォーム起動時間を比較します。

スクリプトを実行するには次の手順に従います。

1. （任意）`vllm bench startup` の基本コマンドを組み立て、`--startup-cmd` に渡します（既定: `vllm bench startup`）。
2. （任意）エンジン設定を変化させるために、`vllm bench sweep serve` の `--serve-params` の JSON を再利用します。`vllm bench startup` がサポートするパラメータのみが適用されます。
3. （任意）反復回数など、起動固有のオプションを変化させるための `--startup-params` の JSON を作成します。
4. 結果の保存先を決め、`--output-dir` に渡します。

`--serve-params` の例:

```json
[
    {
        "_benchmark_name": "tp1",
        "model": "Qwen/Qwen3-0.6B",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9
    },
    {
        "_benchmark_name": "tp2",
        "model": "Qwen/Qwen3-0.6B",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.9
    }
]
```

`--startup-params` の例:

```json
[
    {
        "_benchmark_name": "qwen3-0.6",
        "num_iters_cold": 2,
        "num_iters_warmup": 1,
        "num_iters_warm": 2
    }
]
```

Example command:

```bash
vllm bench sweep startup \
    --startup-cmd 'vllm bench startup --model Qwen/Qwen3-0.6B' \
    --serve-params benchmarks/serve_hparams.json \
    --startup-params benchmarks/startup_hparams.json \
    --output-dir benchmarks/results \
    --experiment-name demo
```

!!! important
    既定では、`--serve-params` や `--startup-params` に含まれる未対応のパラメータは警告とともに無視されます。
    未知のキーで即座に失敗させたい場合は `--strict-params` を使ってください。

## 可視化 { #visualization }

### 基本 { #basic_1 }

`vllm bench sweep plot` を使うと、パラメータスイープの結果から性能曲線をプロットできます。

プロットする変数は `--var-x` と `--var-y` で指定し、必要に応じて値に `--filter-by` や `--bin-by` を適用します。プロットの構成は `--fig-by`、`--row-by`、`--col-by`、`--curve-by` で決まります。

[ワークロードエクスプローラー](#workload-explorer)の結果を可視化するコマンドの例:

```bash
EXPERIMENT_DIR=${1:-"benchmarks/results/demo"}

# Latency increases as the workload increases
vllm bench sweep plot $EXPERIMENT_DIR \
    --var-x max_concurrency \
    --var-y median_ttft_ms \
    --col-by _benchmark_name \
    --curve-by max_num_seqs,max_num_batched_tokens \
    --fig-name latency_curve

# Throughput saturates as workload increases
vllm bench sweep plot $EXPERIMENT_DIR \
    --var-x max_concurrency \
    --var-y total_token_throughput \
    --col-by _benchmark_name \
    --curve-by max_num_seqs,max_num_batched_tokens \
    --fig-name throughput_curve

# Tradeoff between latency and throughput
vllm bench sweep plot $EXPERIMENT_DIR \
    --var-x total_token_throughput \
    --var-y median_ttft_ms \
    --col-by _benchmark_name \
    --curve-by max_num_seqs,max_num_batched_tokens \
    --fig-name latency_throughput
```

!!! tip
    プロットされる図は `--dry-run` でプレビューできます。

### パレート図 { #pareto-chart }

`vllm bench sweep plot_pareto` は、ユーザーあたりと GPU あたりのスループットのバランスが取れた構成を選ぶのに役立ちます。

並行度やバッチサイズを上げると GPU の効率（GPU あたり）は高まりますが、ユーザーあたりのレイテンシは増える可能性があります。逆に並行度を下げるとユーザーあたりのレートは改善しますが、GPU を十分に使い切れません。パレートフロンティアは、実行結果の中で達成可能な最良の組み合わせを示します。

- x 軸: tokens/s/user = `output_throughput` ÷ 並行度（`--user-count-var`、既定は `max_concurrency`、フォールバックは `max_concurrent_requests`）。
- y 軸: tokens/s/GPU = `output_throughput` ÷ GPU 数（`--gpu-count-var` を設定した場合はその値。設定しない場合、gpu_count は TP×PP×DP）。
- 出力: `OUTPUT_DIR/pareto/PARETO.png` に 1 枚の図が出力されます。
- 各データ点で使われた構成を表示するには `--label-by` を使います（既定: `max_concurrency,gpu_count`）。

例:

```bash
EXPERIMENT_DIR=${1:-"benchmarks/results/demo"}

vllm bench sweep plot_pareto $EXPERIMENT_DIR \
  --label-by max_concurrency,tensor_parallel_size,pipeline_parallel_size
```

!!! tip
    プロットされる図は `--dry-run` でプレビューできます。
