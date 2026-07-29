# メトリクス { #metrics }

vLLM は、V1 エンジンの可観測性とキャパシティプランニングを支えるため、豊富なメトリクスを公開しています。

## 目的 { #objectives }

- 本番環境の監視に役立つよう、エンジンレベルとリクエストレベルのメトリクスを網羅的に提供すること。
- 本番環境で使われると想定されるため、Prometheus との統合を優先すること。
- 場当たり的なテスト、デバッグ、開発、試行的な用途のために、ログ出力（メトリクスを info ログへ出力する）もサポートすること。

## 背景 { #background }

vLLM のメトリクスは次のように分類できます。

1. サーバーレベルのメトリクス: LLM エンジンの状態と性能を追跡するグローバルなメトリクスです。通常、Prometheus では Gauge または Counter として公開されます。
2. リクエストレベルのメトリクス: 個々のリクエストの特性（サイズやタイミングなど）を追跡するメトリクスです。通常、Prometheus では Histogram として公開され、vLLM を監視する SRE が追跡する SLO になることが多いものです。

サーバーレベルのメトリクスがリクエストレベルのメトリクスの値を説明する、というのが基本的な考え方です。

### メトリクスの概要 { #metrics-overview }

### v1 のメトリクス { #v1-metrics }

v1 では、`vllm:` プレフィックスを用いた Prometheus 互換の `/metrics` エンドポイント経由で、幅広いメトリクスが公開されます。例:

- `vllm:num_requests_running`（Gauge）- 現在実行中のリクエスト数。
- `vllm:kv_cache_usage_perc`（Gauge）- 使用中の KV キャッシュブロックの割合（0〜1）。
- `vllm:prefix_cache_queries`（Counter）- プレフィックスキャッシュへの問い合わせ回数。
- `vllm:prefix_cache_hits`（Counter）- プレフィックスキャッシュのヒット数。
- `vllm:prompt_tokens_total`（Counter）- 処理したプロンプトトークンの総数。
- `vllm:generation_tokens_total`（Counter）- 生成したトークンの総数。
- `vllm:request_success_total`（Counter）- 完了したリクエスト数（完了理由別）。
- `vllm:request_prompt_tokens`（Histogram）- 入力プロンプトのトークン数のヒストグラム。
- `vllm:request_generation_tokens`（Histogram）- 生成トークン数のヒストグラム。
- `vllm:time_to_first_token_seconds`（Histogram）- 最初のトークンまでの時間（TTFT）。
- `vllm:inter_token_latency_seconds`（Histogram）- トークン間レイテンシ。
- `vllm:e2e_request_latency_seconds`（Histogram）- エンドツーエンドのリクエストレイテンシ。
- `vllm:request_prefill_time_seconds`（Histogram）- リクエストのプレフィル時間。
- `vllm:request_decode_time_seconds`（Histogram）- リクエストのデコード時間。

これらは [推論とサービング -> 本番向けメトリクス](../usage/metrics.md)に記載されています。

### Grafana ダッシュボード { #grafana-dashboard }

vLLM は、これらのメトリクスを Prometheus で収集・保存し、Grafana ダッシュボードで可視化する方法の[リファレンス例](https://docs.vllm.ai/en/v0.26.0/examples/observability/prometheus_grafana/)も提供しています。

Grafana ダッシュボードで公開されているメトリクスの一部を見ると、とくに重要なメトリクスが分かります。

- `vllm:e2e_request_latency_seconds_bucket` - 秒単位のエンドツーエンドのリクエストレイテンシ。
- `vllm:prompt_tokens` - プロンプトトークン。
- `vllm:generation_tokens` - 生成トークン。
- `vllm:inter_token_latency_seconds` - 秒単位のトークン間レイテンシ（出力トークンあたりの時間、TPOT）。
- `vllm:time_to_first_token_seconds` - 秒単位の最初のトークンまでのレイテンシ（TTFT）。
- `vllm:num_requests_running`（`_swapped` と `_waiting` も）- RUNNING、WAITING、SWAPPED の各状態にあるリクエスト数。
- `vllm:kv_cache_usage_perc` - vLLM が使用中のキャッシュブロックの割合。
- `vllm:request_prompt_tokens` - リクエストのプロンプト長。
- `vllm:request_generation_tokens` - リクエストの生成長。
- `vllm:request_success` - 完了理由（EOS トークンの生成、または最大シーケンス長への到達）別の完了リクエスト数。
- `vllm:request_queue_time_seconds` - キュー滞在時間。
- `vllm:request_prefill_time_seconds` - リクエストのプレフィル時間。
- `vllm:request_decode_time_seconds` - リクエストのデコード時間。

ここでの選択の背景については、[このダッシュボードを追加した PR](https://github.com/vllm-project/vllm/pull/2316) が興味深く参考になります。

### Prometheus クライアントライブラリ { #prometheus-client-library }

Prometheus のサポートは当初 [aioprometheus ライブラリを使って](https://github.com/vllm-project/vllm/pull/1890)追加されましたが、ほどなく [prometheus_client](https://github.com/vllm-project/vllm/pull/2730) へ切り替えられました。その理由は、いずれのリンク先の PR でも議論されています。

これらの移行の途中で HTTP メトリクスを追跡する `MetricsMiddleware` が一時的に失われましたが、[prometheus_fastapi_instrumentator を使って](https://github.com/vllm-project/vllm/pull/15657)復活しました。

```bash
$ curl http://0.0.0.0:8000/metrics 2>/dev/null  | grep -P '^http_(?!.*(_bucket|_created|_sum)).*'
http_requests_total{handler="/v1/completions",method="POST",status="2xx"} 201.0
http_request_size_bytes_count{handler="/v1/completions"} 201.0
http_response_size_bytes_count{handler="/v1/completions"} 201.0
http_request_duration_highr_seconds_count 201.0
http_request_duration_seconds_count{handler="/v1/completions",method="POST"} 201.0
```

### マルチプロセスモード { #multi-process-mode }

歴史的には、メトリクスはエンジンコアのプロセスで収集され、API サーバーのプロセスでそれを利用可能にするためにマルチプロセスモードが使われていました。<https://github.com/vllm-project/vllm/pull/7279> を参照してください。

より最近では、メトリクスは API サーバーのプロセスで収集され、マルチプロセスモードが使われるのは `--api-server-count > 1` の場合のみです。<https://github.com/vllm-project/vllm/pull/17546> と [API サーバーのスケールアウト](../serving/data_parallel_deployment.md#internal-load-balancing)の詳細を参照してください。

### 組み込みの Python / プロセスのメトリクス { #built-in-pythonprocess-metrics }

次のメトリクスは `prometheus_client` が既定でサポートしていますが、マルチプロセスモードでは公開されません。

- `python_gc_objects_collected_total`
- `python_gc_objects_uncollectable_total`
- `python_gc_collections_total`
- `python_info`
- `process_virtual_memory_bytes`
- `process_resident_memory_bytes`
- `process_start_time_seconds`
- `process_cpu_seconds_total`
- `process_open_fds`
- `process_max_fds`

したがって、`--api-server-count > 1` の場合これらのメトリクスは利用できません。vLLM インスタンスを構成するすべてのプロセスにわたって統計を集約するわけではないため、これらがどれほど有用かは疑問の余地があります。

## メトリクスの設計 { #metrics-design }

メトリクス設計の多くは [「Even Better Observability」](https://github.com/vllm-project/vllm/issues/3616)の機能として計画されました。たとえば、[詳細なロードマップが示された箇所](https://github.com/vllm-project/vllm/issues/3616#issuecomment-2030858781)を参照してください。

### 過去の PR { #legacy-prs }

メトリクス設計の背景を理解する助けとして、当初の（現在はレガシーとなった）メトリクスを追加した主な PR を挙げます。

- <https://github.com/vllm-project/vllm/pull/1890>
- <https://github.com/vllm-project/vllm/pull/2316>
- <https://github.com/vllm-project/vllm/pull/2730>
- <https://github.com/vllm-project/vllm/pull/4464>
- <https://github.com/vllm-project/vllm/pull/7279>

### メトリクス実装の PR { #metrics-implementation-prs }

背景として、メトリクスの実装（<https://github.com/vllm-project/vllm/issues/10582>）に関連する PR を挙げます。

- <https://github.com/vllm-project/vllm/pull/11962>
- <https://github.com/vllm-project/vllm/pull/11973>
- <https://github.com/vllm-project/vllm/pull/10907>
- <https://github.com/vllm-project/vllm/pull/12416>
- <https://github.com/vllm-project/vllm/pull/12478>
- <https://github.com/vllm-project/vllm/pull/12516>
- <https://github.com/vllm-project/vllm/pull/12530>
- <https://github.com/vllm-project/vllm/pull/12561>
- <https://github.com/vllm-project/vllm/pull/12579>
- <https://github.com/vllm-project/vllm/pull/12592>
- <https://github.com/vllm-project/vllm/pull/12644>

### メトリクスの収集 { #metrics-collection }

v1 では、forward pass 同士の間隔を最小化するため、計算とオーバーヘッドをエンジンコアのプロセスの外へ移したいと考えています。

V1 の EngineCore 設計の全体的な考え方は次のとおりです。

- EngineCore は内側のループです。ここでは性能が最も重要です。
- AsyncLLM は外側のループです。これは（理想的には）GPU の実行とオーバーラップするため、可能な限り「オーバーヘッド」はここに置くべきです。したがって、可能であれば AsyncLLM.output_handler_loop がメトリクス管理の理想的な場所です。

これを実現するため、メトリクスはフロントエンドの API サーバーで収集し、エンジンコアのプロセスがフロントエンドへ返す `EngineCoreOutputs` から得られる情報にもとづくものにします。

### 時間間隔の計算 { #interval-calculations }

vLLM のメトリクスの多くは、リクエスト処理における各種イベント間の時間間隔です。間隔の計算には、「壁時計時刻」（`time.time()`）ではなく「単調時刻」（`time.monotonic()`）にもとづくタイムスタンプを使うのが定石です。前者は（NTP などによる）システムクロックの変更の影響を受けないためです。

また、単調クロックはプロセスごとに異なる（各プロセスが独自の基準点を持つ）点も重要です。したがって、異なるプロセスの単調タイムスタンプ同士を比較しても意味がありません。

そのため、間隔を計算するには、同じプロセスの 2 つの単調タイムスタンプを比較する必要があります。

### スケジューラの統計 { #scheduler-stats }

エンジンコアのプロセスは、スケジューラからいくつかの主要な統計（直近のスケジューラパスの後にスケジュールされた、あるいは待機中のリクエスト数など）を収集し、それらを `EngineCoreOutputs` に含めます。

### エンジンコアのイベント { #engine-core-events }

エンジンコアは、フロントエンドがイベント間の間隔を計算できるよう、リクエストごとの特定のイベントのタイムスタンプも記録します。

イベントは次のとおりです。

- `QUEUED` - リクエストがエンジンコアに受け取られ、スケジューラのキューに追加された時点。
- `SCHEDULED` - リクエストが最初に実行のためスケジュールされた時点。
- `PREEMPTED` - 他のリクエストを完了させる余地を作るため、リクエストが待機キューへ戻された時点。将来的に再スケジュールされ、プレフィルのフェーズをやり直します。
- `NEW_TOKENS` - `EngineCoreOutput` に含まれる出力が生成された時点。これはある反復におけるすべてのリクエストに共通するため、このイベントの記録には `EngineCoreOutputs` 上の単一のタイムスタンプを使います。

そして計算される間隔は次のとおりです。

- キュー間隔 - `QUEUED` から直近の `SCHEDULED` まで。
- プレフィル間隔 - 直近の `SCHEDULED` から、それに続く最初の `NEW_TOKENS` まで。
- デコード間隔 - （直近の `SCHEDULED` 以降の）最初の `NEW_TOKENS` から最後の `NEW_TOKENS` まで。
- 推論間隔 - 直近の `SCHEDULED` から最後の `NEW_TOKENS` まで。
- トークン間の間隔 - 連続する `NEW_TOKENS` のあいだ。

言い換えると次のようになります。

![時間間隔の計算 - 通常のケース](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/metrics/intervals-1.png)

フロントエンドから見えるイベントのタイミングを使って、フロントエンド側でこれらの間隔を計算する可能性も検討しました。しかし、フロントエンドは `QUEUED` と `SCHEDULED` のイベントのタイミングを把握できず、また間隔は同じプロセスの単調タイムスタンプにもとづいて計算する必要があるため、これらすべてのイベントのタイムスタンプはエンジンコアが記録する必要があります。

#### 時間間隔の計算とプリエンプション { #interval-calculations-vs-preemptions }

デコード中にプリエンプションが発生した場合、すでに生成されたトークンは再利用されるため、プリエンプションはトークン間・デコード・推論の各間隔に影響するものとみなします。

![時間間隔の計算 - デコード中のプリエンプション](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/metrics/intervals-2.png)

プレフィル中にプリエンプションが発生した場合（そうしたことが起こり得ると仮定して）、プリエンプションは TTFT とプレフィルの間隔に影響するものとみなします。

![時間間隔の計算 - プレフィル中のプリエンプション](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/metrics/intervals-3.png)

### フロントエンドでの統計収集 { #frontend-stats-collection }

フロントエンドは 1 つの `EngineCoreOutputs`（すなわちエンジンコアの 1 回の反復の出力）を処理する際、その反復に関する各種統計を収集します。

- この反復で新たに生成されたトークンの総数。
- この反復で完了したプレフィルが処理したプロンプトトークンの総数。
- この反復でスケジュールされたリクエストのキュー間隔。
- この反復でプレフィルが完了したリクエストのプレフィル間隔。
- この反復に含まれるすべてのリクエストのトークン間の間隔（出力トークンあたりの時間、TPOT）。
- この反復でプレフィルが完了したリクエストの TTFT（最初のトークンまでの時間）。ただし、入力処理の時間も含めるため、この間隔はフロントエンドがリクエストを最初に受け取った時点（`arrival_time`）を基準に計算します。現在、`arrival_time` はトークナイズの開始時点から始まります。

ある反復で完了したリクエストについては、次も記録します。

- 推論間隔とデコード間隔 - 上記のとおり、スケジュールと最初のトークンのイベントを基準にしたもの。
- エンドツーエンドのレイテンシ - フロントエンドの `arrival_time` から、フロントエンドが最後のトークンを受け取るまでの間隔。

### KV キャッシュの滞在時間メトリクス { #kv-cache-residency-metrics }

サンプリングされた KV キャッシュブロックがどれだけ滞在し、どれだけ再利用されるかを示す一連のヒストグラムも出力します。サンプリング（`--kv-cache-metrics-sample`）によりオーバーヘッドはごくわずかに抑えられます。ブロックが選ばれると、次を記録します。

- `lifetime` – 確保 ⟶ 追い出し
- `idle before eviction` – 最後のアクセス ⟶ 追い出し
- `reuse gaps` – ブロックが再利用される際のアクセス間の間隔

これらは Prometheus のメトリクスに直接対応します。

- `vllm:kv_block_lifetime_seconds` – サンプリングされた各ブロックが存在した時間。
- `vllm:kv_block_idle_before_evict_seconds` – 最後のアクセス以降のアイドル時間。
- `vllm:kv_block_reuse_gap_seconds` – 連続するアクセスのあいだの時間。

エンジンコアは `SchedulerStats` を通じて生の追い出しイベントを送るだけで、フロントエンドがそれを取り出して Prometheus の観測値に変換し、ログ出力が有効な場合は `LLM.get_metrics()` からも同じデータを公開します。lifetime とアイドル時間を 1 つのチャートで見ると、放置されたキャッシュや、長いデコードのあいだプロンプトを保持し続けるワークロードを見つけやすくなります。

### メトリクスの公開 - ログ { #metrics-publishing-logging }

`LoggingStatLogger` のメトリクス公開機能は、5 秒ごとに主要なメトリクスを含む `INFO` レベルのログメッセージを出力します。

- 現在の実行中 / 待機中のリクエスト数
- 現在の GPU キャッシュ使用量
- 直近 5 秒間の 1 秒あたりの処理済みプロンプトトークン数
- 直近 5 秒間の 1 秒あたりの新規生成トークン数
- 直近 1000 回の KV キャッシュブロックの問い合わせにおけるプレフィックスキャッシュのヒット率

### メトリクスの公開 - Prometheus { #metrics-publishing-prometheus }

`PrometheusStatLogger` のメトリクス公開機能は、Prometheus 互換の形式で `/metrics` の HTTP エンドポイントを通じてメトリクスを利用可能にします。Prometheus のインスタンスは、このエンドポイントを（たとえば 1 秒ごとに）ポーリングして値を時系列データベースへ記録するよう設定できます。Prometheus は Grafana と組み合わせて使われることが多く、これらのメトリクスを時系列でグラフ化できます。

Prometheus は次のメトリクス種別をサポートします。

- Counter: 時間とともに増加し、減ることはなく、vLLM インスタンスの再起動時に一般にゼロへリセットされる値です。たとえば、インスタンスの稼働期間中に生成されたトークン数です。
- Gauge: 増減する値です。たとえば、現在実行のためにスケジュールされているリクエスト数です。
- Histogram: バケットごとに記録されたメトリクスのサンプル数です。たとえば、TTFT が 1ms 未満、5ms 未満、10ms 未満、20ms 未満……だったリクエスト数です。

Prometheus のメトリクスにはラベルを付けることもでき、一致するラベルに応じてメトリクスを組み合わせられます。vLLM では、すべてのメトリクスに、そのインスタンスがサービングしているモデル名を含む `model_name` ラベルを付けています。

出力例:

```bash
$ curl http://0.0.0.0:8000/metrics
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="meta-llama/Llama-3.1-8B-Instruct"} 8.0
...
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 27453.0
...
# HELP vllm:request_success_total Count of successfully processed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
vllm:request_success_total{finished_reason="length",model_name="meta-llama/Llama-3.1-8B-Instruct"} 131.0
vllm:request_success_total{finished_reason="abort",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
...
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.001",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
vllm:time_to_first_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
vllm:time_to_first_token_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
vllm:time_to_first_token_seconds_bucket{le="0.02",model_name="meta-llama/Llama-3.1-8B-Instruct"} 13.0
vllm:time_to_first_token_seconds_bucket{le="0.04",model_name="meta-llama/Llama-3.1-8B-Instruct"} 97.0
vllm:time_to_first_token_seconds_bucket{le="0.06",model_name="meta-llama/Llama-3.1-8B-Instruct"} 123.0
vllm:time_to_first_token_seconds_bucket{le="0.08",model_name="meta-llama/Llama-3.1-8B-Instruct"} 138.0
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="meta-llama/Llama-3.1-8B-Instruct"} 140.0
vllm:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 140.0
```

!!! note
    幅広いユースケースにわたって利用者にとって最も有用なヒストグラムのバケットを
    選ぶのは簡単ではなく、時間をかけて改善していく必要があります。

### キャッシュ設定の情報 { #cache-config-info }

`prometheus_client` は [Info メトリクス](https://prometheus.github.io/client_python/instrumenting/info/)をサポートしています。これは値が常に 1 に設定された `Gauge` に相当しますが、ラベルを通じて有用なキー / 値の情報を公開します。これは、変化しないインスタンスの情報（したがって起動時に一度観測すればよい情報）に使われ、Prometheus 上でインスタンス間の比較を可能にします。

vLLM ではこの考え方を `vllm:cache_config_info` メトリクスに使っています。

```text
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",cache_dtype="auto",calculate_kv_scales="False",cpu_offload_gb="0",enable_prefix_caching="False",gpu_memory_utilization="0.9",...} 1.0
```

ただし `prometheus_client` は、[不明確な理由](gh-pr:7279#discussion_r1710417152)により、
[マルチプロセスモードでの Info メトリクスを一度もサポートしたことがありません](https://github.com/prometheus/client_python/pull/300)。
そこで vLLM では代わりに、値を 1 に設定し `multiprocess_mode="mostrecent"` とした `Gauge` メトリクスを単純に使っています。

### LoRA のメトリクス { #lora-metrics }

`vllm:lora_requests_info` の `Gauge` もこれに似ていますが、値が現在の壁時計時刻であり、反復ごとに更新される点が異なります。

使われるラベル名は次のとおりです。

- `running_lora_adapters`: そのアダプタを使って実行中のリクエスト数をアダプタごとに数えたもので、カンマ区切りの文字列として整形されます。
- `waiting_lora_adapters`: 同様ですが、スケジュール待ちのリクエストを数えます。
- `max_lora` - 「単一バッチ内の LoRA の最大数」という静的な設定値。

複数アダプタの実行中 / 待機中の件数をカンマ区切りの文字列にエンコードするのは、かなり筋の悪い方法に見えます。アダプタごとの件数の区別にはラベルを使えるはずで、これは見直すべきです。

`multiprocess_mode="livemostrecent"` が使われている点に注意してください。最新のメトリクスが使われますが、現在実行中のプロセスのものに限られます。

これは <https://github.com/vllm-project/vllm/pull/9477> で追加され、[少なくとも 1 つの既知の利用者](https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/54)がいます。
この設計を見直して古いメトリクスを非推奨にする場合は、削除前に移行できるよう下流の利用者と調整すべきです。

### プレフィックスキャッシュのメトリクス { #prefix-cache-metrics }

プレフィックスキャッシュのメトリクス追加に関する <https://github.com/vllm-project/vllm/issues/10582> の議論からは、今後のメトリクスへの取り組み方にも関わる興味深い論点が得られました。

プレフィックスキャッシュに問い合わせるたびに、問い合わせたトークン数と、そのうちキャッシュに存在したトークン数（すなわちヒット数）を記録します。

しかし、関心があるのはヒット率、つまり問い合わせあたりのヒット数です。

ログ出力の場合、直近の一定件数の問い合わせにわたってヒット率を計算するのが利用者にとって最も有用だと考えています（現時点では直近 1000 件に固定されています）。

一方 Prometheus の場合は、その時系列としての性質を活かし、利用者が任意の区間でヒット率を計算できるようにすべきです。たとえば、直近 5 分間のヒット率を計算する PromQL クエリは次のようになります。

```text
rate(cache_query_hit[5m]) / rate(cache_query_total[5m])
```

これを実現するには、ヒット率を gauge として記録するのではなく、問い合わせ数とヒット数を Prometheus の counter として記録すべきです。

## 非推奨のメトリクス { #deprecated-metrics }

### 非推奨化の進め方 { #how-to-deprecate }

メトリクスの非推奨化を軽々しく行うべきではありません。利用者はメトリクスが非推奨になったことに気づかないかもしれず、代替となる同等のメトリクスがあったとしても、（利用者から見れば）突然削除されると大きな不便を被る可能性があります。

例として、`vllm:avg_prompt_throughput_toks_per_s` が（コード中のコメント付きで）[非推奨になり](https://github.com/vllm-project/vllm/pull/2764)、[削除され](https://github.com/vllm-project/vllm/pull/12383)、その後[利用者に気づかれた](https://github.com/vllm-project/vllm/issues/13218)経緯を参照してください。

一般に次のようにすべきです。

1. とくに利用者への影響を予測しにくいことから、メトリクスの非推奨化には慎重であるべきです。
2. `/metrics` の出力に含まれるヘルプ文字列に、目立つ形で非推奨の告知を含めるべきです。
3. 非推奨のメトリクスは、利用者向けのドキュメントとリリースノートに列挙すべきです。
4. 削除するまでのあいだ管理者に[逃げ道](https://kubernetes.io/docs/concepts/cluster-administration/system-metrics/#show-hidden-metrics)を与えるため、非推奨のメトリクスを CLI 引数の背後に隠すことを検討すべきです。

プロジェクト全体の非推奨化ポリシーについては[非推奨化ポリシー](../contributing/deprecation_policy.md)を参照してください。

### 未実装 - `vllm:tokens_total` { #unimplemented-vllmtokens_total }

<https://github.com/vllm-project/vllm/pull/4464> で追加されましたが、実装されないままのようです。これは単に削除して構いません。

### 重複 - キュー滞在時間 { #duplicated-queue-time }

`vllm:time_in_queue_requests` の Histogram メトリクスは <https://github.com/vllm-project/vllm/pull/9659> で追加され、その計算は次のとおりです。

```python
    self.metrics.first_scheduled_time = now
    self.metrics.time_in_queue = now - self.metrics.arrival_time
```

その 2 週間後、<https://github.com/vllm-project/vllm/pull/4464> が `vllm:request_queue_time_seconds` を追加し、次のような状態になりました。

```python
if seq_group.is_finished():
    if (seq_group.metrics.first_scheduled_time is not None and
            seq_group.metrics.first_token_time is not None):
        time_queue_requests.append(
            seq_group.metrics.first_scheduled_time -
            seq_group.metrics.arrival_time)
    ...
    if seq_group.metrics.time_in_queue is not None:
        time_in_queue_requests.append(
            seq_group.metrics.time_in_queue)
```

これは重複しており、どちらか一方は削除すべきです。後者は Grafana ダッシュボードで使われているため、前者を非推奨にするか削除すべきです。

### プレフィックスキャッシュのヒット率 { #prefix-cache-hit-rate }

前述のとおり、現在は「ヒット率」の gauge ではなく「問い合わせ数」と「ヒット数」の counter を公開しています。

### KV キャッシュのオフロード { #kv-cache-offloading }

レガシーのメトリクスのうち 2 つは、v1 ではもはや関係のない「swapped」なプリエンプションモードに関するものです。

- `vllm:num_requests_swapped`
- `vllm:cpu_cache_usage_perc`

このモードでは、リクエストがプリエンプトされたとき（他のリクエストを完了させるために KV キャッシュに空きを作る場合など）、KV キャッシュのブロックが CPU メモリへスワップアウトされていました。V1 ではこの機能が使われなくなったため、`--swap-space` フラグは削除されました。

歴史的に、[vLLM は長らくビームサーチをサポートしてきました](https://github.com/vllm-project/vllm/issues/6226)。SequenceGroup は、同じプロンプトの KV ブロックを共有する N 個の Sequence という考え方をカプセル化したものでした。これによりリクエスト間で KV キャッシュブロックを共有し、コピーオンライトで分岐できました。CPU へのスワップは、こうしたビームサーチのようなケースを想定したものでした。

その後、KV キャッシュブロックを暗黙的に共有できるようにするプレフィックスキャッシュの概念が導入されました。ブロックは必要に応じて徐々に追い出せ、追い出されたプロンプトの部分は再計算できるため、これは CPU へのスワップより優れた選択肢であることが分かりました。

SequenceGroup は V1 で削除されましたが、「並列サンプリング」（`n>1`）のためには代替が必要になります。[ビームサーチはコアの外へ移されました](https://github.com/vllm-project/vllm/issues/8306)。ごく限られた用途の機能に対して、複雑なコードが多く存在していました。

V1 では、プレフィックスキャッシュがより優れており（オーバーヘッドがゼロ）既定で有効になっているため、プリエンプションと再計算の戦略はより良く機能するはずです。

## 今後の作業 { #future-work }

### 並列サンプリング { #parallel-sampling }

レガシーのメトリクスの一部は「並列サンプリング」の文脈でのみ意味を持ちます。これは、リクエストの `n` パラメータを使って同じプロンプトから複数の補完を要求する場合です。

<https://github.com/vllm-project/vllm/pull/10980> で並列サンプリングのサポートを追加する一環として、これらのメトリクスも追加すべきです。

- `vllm:request_params_n`（Histogram）

  完了した各リクエストの 'n' パラメータの値を観測します。

- `vllm:request_max_num_generation_tokens`（Histogram）

  完了した各シーケンスグループ内の全シーケンスの最大出力長を観測します。並列サンプリングがない場合、これは `vllm:request_generation_tokens` と等価です。

### 投機的デコーディング { #speculative-decoding }

レガシーのメトリクスの一部は「投機的デコーディング」に固有のものです。これは、より高速で近似的な手法やモデルで候補トークンを生成し、それらをより大きなモデルで検証する方式です。

- `vllm:spec_decode_draft_acceptance_rate`（Gauge）
- `vllm:spec_decode_efficiency`（Gauge）
- `vllm:spec_decode_num_accepted_tokens`（Counter）
- `vllm:spec_decode_num_draft_tokens`（Counter）
- `vllm:spec_decode_num_emitted_tokens`（Counter）

v1 に「prompt lookup（ngram）」の投機的デコーディングを追加する PR（<https://github.com/vllm-project/vllm/pull/12193>）がレビュー中です。他の手法も続く予定です。この文脈でこれらのメトリクスを見直すべきです。

!!! note
    プレフィックスキャッシュのヒット率と同様に、受理率も受理数とドラフト数の
    別々の counter として公開すべきでしょう。efficiency についても
    同様の扱いが必要と思われます。

### オートスケーリングと負荷分散 { #autoscaling-and-load-balancing }

vLLM のメトリクスのよくあるユースケースの 1 つは、vLLM インスタンスの自動スケーリングを支えることです。

[Kubernetes Serving Working Group](https://github.com/kubernetes/community/tree/master/wg-serving) の関連する議論としては、次を参照してください。

- [Standardizing Large Model Server Metrics in Kubernetes](https://docs.google.com/document/d/1SpSp1E6moa4HSrJnS4x3NpLuj88sMXr2tbofKlzTZpk)
- [Benchmarking LLM Workloads for Performance Evaluation and Autoscaling in Kubernetes](https://docs.google.com/document/d/1k4Q4X14hW4vftElIuYGDu5KDe2LtV1XammoG-Xi3bbQ)
- [Inference Perf](https://github.com/kubernetes-sigs/wg-serving/tree/main/proposals/013-inference-perf)
- <https://github.com/vllm-project/vllm/issues/5041> and <https://github.com/vllm-project/vllm/pull/12726>.

これは簡単な話ではありません。Rob の次のコメントを見てみましょう。

> I think this metric should focus on trying to estimate what the max
> concurrency that will cause the average request length > queries per
> second ... since this is really what will "saturate" the server.

明確な目標は、管理者がそれにもとづいて自動スケーリングのルールを実装できるよう、この飽和点を検出するために必要なメトリクスを公開することです。ただしそのためには、管理者（および自動監視システム）がインスタンスが飽和に近づいていることをどう判断すべきか、明確な見通しが必要です。

> To identify, what is the saturation point for model server compute
> (the inflection point where we cannot get more throughput with a
> higher request rate, but start to incur additional latency) so we
> can autoscale effectively?

### メトリクスの命名 { #metric-naming }

メトリクスの命名方針は、おそらく見直す価値があります。

1. メトリクス名にコロンを使うのは、[「コロンはユーザー定義の recording rule のために予約されている」](https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels)という方針に反しているように見えます。
2. ほとんどのメトリクスは末尾に単位を付ける慣例に従っていますが、すべてではありません。
3. 一部のメトリクス名は `_total` で終わります。

    メトリクス名に `_total` のサフィックスがある場合、それは取り除かれます。counter の時系列を公開する際には `_total` のサフィックスが付加されます。これは、OpenMetrics が `_total` のサフィックスを要求するため、OpenMetrics と Prometheus のテキスト形式のあいだの互換性を保つためです。

### メトリクスの追加 { #adding-more-metrics }

新しいメトリクスのアイデアには事欠きません。

- [TGI](https://github.com/IBM/text-generation-inference?tab=readme-ov-file#metrics) のような他プロジェクトの例
- 上記の Kubernetes による自動スケーリングのような、特定のユースケースから生じる提案
- [OpenTelemetry の生成 AI 向けセマンティック規約](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai)のような標準化の取り組みから生じ得る提案

新しいメトリクスの追加には慎重であるべきです。メトリクスの追加は比較的簡単なことが多い一方で、次の点があります。

1. 削除するのは難しいことがあります。上記の非推奨化の節を参照してください。
2. 有効にすると無視できない性能への影響が生じ得ます。またメトリクスは、既定かつ本番で有効にできない限り、有用性がかなり限られます。
3. プロジェクトの開発と保守に影響します。これまでに追加されてきたメトリクスは、その都度この労力を増やしてきました。すべてのメトリクスが、その継続的な保守コストに見合うとは限らないかもしれません。

## トレーシング - OpenTelemetry { #tracing-opentelemetry }

メトリクスは、システムの性能と健全性に関する時系列の集約されたビューを提供します。一方トレーシングは、個々のリクエストが異なるサービスやコンポーネントを通過する様子を追跡します。どちらも、より一般的な「可観測性」という見出しの下に位置づけられます。

vLLM は OpenTelemetry のトレーシングをサポートしています。

- Added by <https://github.com/vllm-project/vllm/pull/4687> and reinstated by <https://github.com/vllm-project/vllm/pull/20372>
- Configured with `--oltp-traces-endpoint` and `--collect-detailed-traces`
- [OpenTelemetry blog post](https://opentelemetry.io/blog/2024/llm-observability/)
- [User-facing docs](https://docs.vllm.ai/en/v0.26.0/examples/observability/opentelemetry/)
- [Blog post](https://medium.com/@ronen.schaffer/follow-the-trail-supercharging-vllm-with-opentelemetry-distributed-tracing-aa655229b46f)
- [IBM product docs](https://www.ibm.com/docs/en/instana-observability/current?topic=mgaa-monitoring-large-language-models-llms-vllm-public-preview)

OpenTelemetry には [Gen AI Working Group](https://github.com/open-telemetry/community/blob/main/projects/gen-ai.md) があります。

メトリクスだけでも十分に大きなテーマであるため、トレーシングについてはメトリクスとはかなり別のテーマとして扱います。

### OpenTelemetry のモデル forward 時間と execute 時間 { #opentelemetry-model-forward-vs-execute-time }

現在の実装では、次の 2 つのメトリクスを公開しています。

- `vllm:model_forward_time_milliseconds`（Histogram）- そのリクエストがバッチに含まれていたときの、モデルの forward pass に費やされた時間。
- `vllm:model_execute_time_milliseconds`（Histogram）- モデルの execute 関数に費やされた時間。モデルの forward、ワーカー間のブロック / 同期、CPU-GPU の同期時間、サンプリング時間を含みます。

これらのメトリクスは、OpenTelemetry のトレーシングが有効で、かつ `--collect-detailed-traces=all/model/worker` が指定された場合にのみ有効になります。このオプションのドキュメントには次のように記載されています。

> collect detailed traces for the specified modules. This involves
> use of possibly costly and or blocking operations and hence might
> have a performance impact.

これらのメトリクスは <https://github.com/vllm-project/vllm/pull/7089> で追加され、OpenTelemetry のトレースには次のように現れます。

```text
-> gen_ai.latency.time_in_scheduler: Double(0.017550230026245117)
-> gen_ai.latency.time_in_model_forward: Double(3.151565277099609)
-> gen_ai.latency.time_in_model_execute: Double(3.6468167304992676)
```

すでに `inference_time` と `decode_time` のメトリクスがあるため、より細かい粒度の計測がオーバーヘッドに見合うほど一般的なユースケースがあるかどうかが論点になります。

OpenTelemetry のサポートについては別テーマとして扱うため、これらのメトリクスもそのテーマの中に含めることにします。
