# 性能ダッシュボード { #performance-dashboard }

性能ダッシュボードは、新しい変更がさまざまなワークロードで性能を改善するか、あるいは悪化させるかを確認するために使います。`perf-benchmarks` と `ready` の両方のラベルが付いたコミット、および PR が vLLM にマージされたタイミングでベンチマークの実行がトリガーされ、ダッシュボードが更新されます。

結果は公開されている [vLLM Performance Dashboard](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm) に自動的に公開されます。

## ベンチマークを手動でトリガーする { #manually-trigger-the-benchmark }

vLLM のベンチマークスイートと一緒に [vllm-ci-test-repo のイメージ](https://gallery.ecr.aws/q9t5s3a7/vllm-ci-test-repo)を使ってください。x86 CPU 環境では末尾が "-cpu" のイメージを、AArch64 CPU 環境では末尾が "-arm64-cpu" のイメージを使います。

次は CPU 向けの docker run コマンドの例です。GPU の場合は `ON_CPU` 環境変数の設定を省略してください。

```bash
export VLLM_COMMIT=7f42dc20bb2800d09faa72b26f25d54e26f1b694 # use full commit hash from the main branch
export HF_TOKEN=<valid Hugging Face token>
if [[ "$(uname -m)" == aarch64 || "$(uname -m)" == arm64 ]]; then
  IMG_SUFFIX="arm64-cpu"
else
  IMG_SUFFIX="cpu"
fi
docker run -it --entrypoint /bin/bash -v /data/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF_TOKEN -e ON_CPU=1 --shm-size=16g --name vllm-cpu-ci public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:${VLLM_COMMIT}-${IMG_SUFFIX}
```

次に、docker インスタンス内で以下のコマンドを実行します。

```bash
bash .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
```

実行すると、ベンチマークスクリプトは **benchmark/results** フォルダ以下に結果を生成し、benchmark_results.md と benchmark_results.json も出力します。

### 実行時の環境変数 { #runtime-environment-variables }

- `ON_CPU`: Intel® Xeon® および Arm® Neoverse™ プロセッサでは '1' を設定します。既定値は 0。
- `SERVING_JSON`: サービングテストに使う JSON ファイル。既定値は空文字列（既定のファイルを使用）。
- `LATENCY_JSON`: レイテンシテストに使う JSON ファイル。既定値は空文字列（既定のファイルを使用）。
- `THROUGHPUT_JSON`: スループットテストに使う JSON ファイル。既定値は空文字列（既定のファイルを使用）。
- `REMOTE_HOST`: ベンチマーク対象のリモート vLLM サービスの IP。既定値は空文字列。
- `REMOTE_PORT`: ベンチマーク対象のリモート vLLM サービスのポート。既定値は空文字列。
- `PROMPTS_PER_CONCURRENCY`: サービングテストの `num_prompts` を計算するための倍率（`num_prompts = max_concurrency × 値`）。JSON の `num_prompts` を上書きします。既定値は NULL。
- `ENABLE_ADAPTIVE_CONCURRENCY`: '1' を設定すると、静的なサービングの max_concurrency スイープのあとに、SLA にもとづく適応的な並行度探索が有効になります。既定値は 0。
- `SLA_TTFT_MS`: 適応的な並行度探索における既定の TTFT の SLA しきい値（ミリ秒）。既定値は 3000。
- `SLA_TPOT_MS`: 適応的な並行度探索における既定の TPOT の SLA しきい値（ミリ秒）。既定値は 100。
- `ADAPTIVE_MAX_PROBES`: 適応的探索で追加するプローブの最大回数。既定値は 8。
- `ADAPTIVE_MAX_CONCURRENCY`: 適応的探索で許容する最大の並行度。既定値は 1024。

### 可視化 { #visualization }

`convert-results-json-to-markdown.py` を使うと、実際のベンチマーク結果を Markdown の表にまとめられます。結果の表は `buildkite/performance-benchmark` のジョブページで確認できます。表が表示されない場合は、ベンチマークの実行が終わるまで待ってください。表の JSON 版（およびベンチマークの JSON 版）も Markdown ファイルに添付されます。生のベンチマーク結果（JSON ファイル）は、ベンチマークの `Artifacts` タブにあります。

#### 性能結果の比較 { #performance-results-comparison }

`compare-json-results.py` を使うと、`convert-results-json-to-markdown.py` で変換したベンチマーク結果の JSON ファイルを比較できます。実行すると、ベンチマークスクリプトは `benchmark/results` フォルダ以下に結果を生成し、`benchmark_results.md` と `benchmark_results.json` も出力します。`compare-json-results.py` は 2 つの `benchmark_results.json` を比較し、Output Tput、Median TTFT、Median TPOT などの性能比を出力します。  
benchmark_results.json を 1 つだけ渡した場合、`compare-json-results.py` は代わりにその中の異なる TP / PP 構成を比較します。

次は、同じモデル・データセット名・入出力長について、最大並行度と qps ごとに result_a と result_b を比較する例です。
`python3 compare-json-results.py -f results_a/benchmark_results.json -f results_b/benchmark_results.json`

***Output Tput (tok/s) — Model : [ meta-llama/Llama-3.1-8B-Instruct ] , Dataset Name : [ random ] , Input Len : [ 2048.0 ] , Output Len : [ 2048.0 ]***

| | # of max concurrency | qps | results_a/benchmark_results.json | results_b/benchmark_results.json | perf_ratio |
| | -------------------- | --- | -------------------------------- | -------------------------------- | ---------- |
| 0 | 12 | inf | 24.98 | 186.03 |  7.45 |
| 1 | 16 | inf |  25.49 | 246.92 | 9.69 |
| 2 | 24 | inf | 27.74 | 293.34 |  10.57 |
| 3 | 32 | inf | 28.61 |306.69 | 10.72 |

***compare-json-results.py – コマンドラインパラメータ***  

compare-json-results.py には、1 つ以上の benchmark_results.json を比較してサマリー表とグラフを生成するための設定可能なパラメータがあります。多くの場合、対象のベンチマーク結果を読み込むために `--file` を指定するだけで十分です。

| パラメータ              | 型               | 既定値           | 説明                                                                                           |
| ---------------------- | ------------------ | ----------------------- | ----------------------------------------------------------------------------------------------------- |
| `--file`               | `str`（複数指定可） | *なし*                  | 入力となる JSON 結果ファイル。複数回指定して複数のベンチマーク結果を比較できます。     |
| `--debug`              | `bool`             | `False`                 | デバッグモードを有効にします。設定すると、トラブルシューティングと検証に役立つ情報をすべて出力します。 |
| `--plot` / `--no-plot` | `bool`             | `True`                  | 性能グラフを生成するかどうかを制御します。グラフ生成を無効にするには `--no-plot` を使います。        |
| `--xaxis`              | `str`              | `# of max concurrency.` | 比較グラフの X 軸に使う列名（並行度やバッチサイズなど）。          |
| `--latency`            | `str`              | `p99`                   | TTFT / TPOT に使うレイテンシの集計方法。指定可能な値: `median` または `p99`。                   |
| `--ttft-max-ms`        | `float`            | `3000.0`                | TTFT グラフの参照上限（ミリ秒）。通常は SLA のしきい値を可視化するために使います。      |
| `--tpot-max-ms`        | `float`            | `100.0`                 | TPOT グラフの参照上限（ミリ秒）。通常は SLA のしきい値を可視化するために使います。      |

***有効な最大並行度のサマリー***  

設定された TTFT と TPOT の SLA しきい値にもとづき、compare-json-results.py は各ベンチマーク結果について有効な最大並行度を計算します。  
「Max # of max concurrency. (Both)」の列は、TTFT と TPOT の両方の制約を同時に満たす最も高い並行度を表します。  
この値は通常、キャパシティプランニングやサイジングのガイドに使われます。  

| # | Configuration  | Max # of max concurrency. (TTFT ≤ 10000 ms) | Max # of max concurrency. (TPOT ≤ 100 ms) | Max # of max concurrency. (Both) | Output Tput @ Both (tok/s) | TTFT @ Both (ms) | TPOT @ Both (ms) |
| - | -------------- | ------------------------------------------- | ----------------------------------------- | -------------------------------- | -------------------------- | ---------------- | ---------------- |
| 0 | results-a      | 128.00                                      | 12.00                                     | 12.00                            | 127.76                     | 3000.82          | 93.24            |
| 1 | results-b      | 128.00                                      | 32.00                                     | 32.00                            | 371.42                     | 2261.53          | 81.74            |

性能ベンチマークとそのパラメータの詳細は、[ベンチマークの README](https://github.com/intel-ai-tce/vllm/blob/more_cpu_models/.buildkite/nightly-benchmarks/README.md) と[性能ベンチマークの説明](../../.buildkite/performance-benchmarks/performance-benchmarks-descriptions.md)を参照してください。

## 継続的ベンチマーク { #continuous-benchmarking }

継続的ベンチマークは、さまざまなモデルと GPU デバイスにわたる vLLM の性能を自動的に監視します。これにより、vLLM の性能特性を時系列で追跡し、性能のリグレッションや改善を把握できます。

### 仕組み { #how-it-works }

継続的ベンチマークは、PyTorch のインフラリポジトリにある [GitHub ワークフロー CI](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-benchmark.yml) からトリガーされ、4 時間ごとに自動実行されます。このワークフローは 3 種類の性能テストを実行します。

- **サービングテスト**: リクエスト処理と API の性能を測定します
- **スループットテスト**: トークン生成のレートを評価します
- **レイテンシテスト**: 応答時間の特性を評価します

### ベンチマークの設定 { #benchmark-configuration }

現在、ベンチマークは [vllm-benchmarks ディレクトリ](https://github.com/pytorch/pytorch-integration-testing/tree/main/vllm-benchmarks/benchmarks)で設定された、あらかじめ定義されたモデル群に対して実行されます。ベンチマーク対象のモデルを追加するには次のようにします。

1. ベンチマーク設定の中の該当する GPU ディレクトリに移動する
2. 対応する設定ファイルにモデルの仕様を追加する
3. 追加したモデルは、次回のスケジュール実行に含まれます
